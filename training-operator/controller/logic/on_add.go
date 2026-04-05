package logic

import (
	"context"
	"fmt"
	"log"

	"training-operator/controller/pipeline"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
)

// OnAdd 用户创建了一个 TrainingJob → 用 Pipeline 链式创建子资源
func OnAdd(ctx context.Context, kubeClient kubernetes.Interface, u *unstructured.Unstructured) {
	name := u.GetName()
	namespace := u.GetNamespace()

	// 从 CR 里读 spec
	image, _, _ := unstructured.NestedString(u.Object, "spec", "image")
	workers, _, _ := unstructured.NestedInt64(u.Object, "spec", "workers")
	command, _, _ := unstructured.NestedStringSlice(u.Object, "spec", "command")
	args, _, _ := unstructured.NestedStringSlice(u.Object, "spec", "args")
	checkpointPVC, _, _ := unstructured.NestedString(u.Object, "spec", "checkpointPVC")

	log.Printf("   image=%s, workers=%d, command=%v\n", image, workers, command)

	// 用 Pipeline 链式执行，任意一步失败自动停止
	pipeline.New("create-"+name).
		Step("create-headless-service", func() error {
			return ensureService(ctx, kubeClient, namespace, name)
		}).
		Step("create-statefulset", func() error {
			return ensureStatefulSet(ctx, kubeClient, namespace, name, image, int32(workers), command, args, checkpointPVC)
		}).
		OnError(func(stepName string, err error) {
			log.Printf("⚠️  任务 %s 创建中断，已完成的资源需要清理\n", name)
		}).
		Run()
}

// ensureService 确保 Headless Service 存在 (幂等)
func ensureService(ctx context.Context, kubeClient kubernetes.Interface, namespace, jobName string) error {
	svcName := jobName + "-svc"

	_, err := kubeClient.CoreV1().Services(namespace).Get(ctx, svcName, metav1.GetOptions{})
	if err == nil {
		log.Printf("   Service %s 已存在，跳过\n", svcName)
		return nil
	}
	if !errors.IsNotFound(err) {
		return fmt.Errorf("查询 Service 失败: %w", err)
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: svcName, Namespace: namespace},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Selector:  map[string]string{"job": jobName},
			Ports: []corev1.ServicePort{{
				Port: 29500, TargetPort: intstr.FromInt(29500), Name: "torchrun",
			}},
		},
	}
	_, err = kubeClient.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{})
	return err
}

// ensureStatefulSet 确保 StatefulSet 存在 (幂等)
func ensureStatefulSet(ctx context.Context, kubeClient kubernetes.Interface, namespace, jobName, image string, workers int32, command, args []string, checkpointPVC string) error {
	stsName := jobName + "-worker"

	_, err := kubeClient.AppsV1().StatefulSets(namespace).Get(ctx, stsName, metav1.GetOptions{})
	if err == nil {
		log.Printf("   StatefulSet %s 已存在，跳过\n", stsName)
		return nil
	}
	if !errors.IsNotFound(err) {
		return fmt.Errorf("查询 StatefulSet 失败: %w", err)
	}

	checkpointPath := "/checkpoints/" + jobName
	container := corev1.Container{
		Name:  "trainer",
		Image: image,
		Command: []string{"/bin/bash", "-c",
			"mkdir -p " + checkpointPath + " && " +
				"export RANK=${POD_NAME##*-} && export LOCAL_RANK=0 && exec " +
				joinStrings(command) + " " + joinStrings(args),
		},
		Env: []corev1.EnvVar{
			{Name: "WORLD_SIZE", Value: fmt.Sprintf("%d", workers)},
			{Name: "MASTER_ADDR", Value: fmt.Sprintf("%s-0.%s-svc", stsName, jobName)},
			{Name: "MASTER_PORT", Value: "29500"},
			{Name: "CHECKPOINT_DIR", Value: checkpointPath},
			{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
			}},
		},
	}

	var volumes []corev1.Volume
	if checkpointPVC != "" {
		container.VolumeMounts = []corev1.VolumeMount{
			{Name: "checkpoint-storage", MountPath: "/checkpoints"},
		}
		volumes = []corev1.Volume{{
			Name: "checkpoint-storage",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: checkpointPVC,
				},
			},
		}}
	}

	replicas := workers
	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: stsName, Namespace: namespace},
		Spec: appsv1.StatefulSetSpec{
			ServiceName:         jobName + "-svc",
			Replicas:            &replicas,
			PodManagementPolicy: appsv1.ParallelPodManagement,
			Selector:            &metav1.LabelSelector{MatchLabels: map[string]string{"job": jobName}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"job": jobName}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{container}, Volumes: volumes},
			},
		},
	}

	_, err = kubeClient.AppsV1().StatefulSets(namespace).Create(ctx, sts, metav1.CreateOptions{})
	return err
}

func joinStrings(ss []string) string {
	result := ""
	for i, s := range ss {
		if i > 0 {
			result += " "
		}
		result += s
	}
	return result
}
