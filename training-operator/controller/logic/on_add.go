package logic

import (
	"context"
	"fmt"
	"log"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
)

// OnAdd 用户创建了一个 TrainingJob → 创建 Headless Service + StatefulSet
func OnAdd(ctx context.Context, kubeClient kubernetes.Interface, u *unstructured.Unstructured) {
	name := u.GetName()
	namespace := u.GetNamespace()

	// 从 CR 里读 spec
	image, _, _ := unstructured.NestedString(u.Object, "spec", "image")
	workers, _, _ := unstructured.NestedInt64(u.Object, "spec", "workers")
	command, _, _ := unstructured.NestedStringSlice(u.Object, "spec", "command")
	args, _, _ := unstructured.NestedStringSlice(u.Object, "spec", "args")

	log.Printf("   image=%s, workers=%d, command=%v\n", image, workers, command)

	svcName := name + "-svc"
	stsName := name + "-worker"

	// --- Step 1: 创建 Headless Service (Pod 间 DNS 互相发现) ---
	_, err := kubeClient.CoreV1().Services(namespace).Get(ctx, svcName, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		svc := &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: svcName, Namespace: namespace},
			Spec: corev1.ServiceSpec{
				ClusterIP: "None",
				Selector:  map[string]string{"job": name},
				Ports: []corev1.ServicePort{{
					Port: 29500, TargetPort: intstr.FromInt(29500), Name: "torchrun",
				}},
			},
		}
		if _, err := kubeClient.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{}); err != nil {
			log.Printf("❌ 创建 Service 失败: %v\n", err)
			return
		}
		log.Printf("✅ 创建 Headless Service: %s\n", svcName)
	}

	// --- Step 2: 创建 StatefulSet ---
	_, stsErr := kubeClient.AppsV1().StatefulSets(namespace).Get(ctx, stsName, metav1.GetOptions{})
	if stsErr == nil {
		log.Printf("   StatefulSet %s 已存在，跳过\n", stsName)
		return
	}
	if !errors.IsNotFound(stsErr) {
		log.Printf("❌ 查询失败: %v\n", stsErr)
		return
	}

	// 读取 checkpointPVC
	checkpointPVC, _, _ := unstructured.NestedString(u.Object, "spec", "checkpointPVC")

	// 构建容器
	// Checkpoint 路径: /checkpoints/{job-name}，每个任务隔离
	checkpointPath := "/checkpoints/" + name

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
			{Name: "MASTER_ADDR", Value: fmt.Sprintf("%s-0.%s-svc", stsName, name)},
			{Name: "MASTER_PORT", Value: "29500"},
			{Name: "CHECKPOINT_DIR", Value: checkpointPath},
			{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
			}},
		},
	}

	// 如果指定了 PVC，挂载到 /checkpoints
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

	// 创建 StatefulSet
	replicas := int32(workers)
	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      stsName,
			Namespace: namespace,
		},
		Spec: appsv1.StatefulSetSpec{
			ServiceName:         name + "-svc",
			Replicas:            &replicas,
			PodManagementPolicy: appsv1.ParallelPodManagement,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"job": name},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"job": name},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{container},
					Volumes:    volumes,
				},
			},
		},
	}

	_, err = kubeClient.AppsV1().StatefulSets(namespace).Create(ctx, sts, metav1.CreateOptions{})
	if err != nil {
		log.Printf("❌ 创建 StatefulSet 失败: %v\n", err)
	} else {
		log.Printf("✅ 创建 StatefulSet: %s (%d workers)\n", stsName, workers)
	}
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
