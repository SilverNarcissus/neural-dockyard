package controller

import (
	"context"
	"fmt"
	"log"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

// ============================================================
// 最简 Controller — 只有 3 件事:
//   1. Watch:     监听 TrainingJob CR 的增删事件
//   2. Reconcile: CR 创建时 → 创建 StatefulSet
//   3. Cleanup:   CR 删除时 → 删除 StatefulSet
// ============================================================

type Controller struct {
	kubeClient    kubernetes.Interface        // 操作原生资源 (StatefulSet, Service)
	dynamicClient dynamic.Interface           // 操作 CRD 资源 (TrainingJob)
	tjResource    schema.GroupVersionResource // CRD 的 GVR 标识
}

func New(kubeClient kubernetes.Interface, dynamicClient dynamic.Interface) *Controller {
	return &Controller{
		kubeClient:    kubeClient,
		dynamicClient: dynamicClient,
		tjResource: schema.GroupVersionResource{
			Group:    "training.megatron.io",
			Version:  "v1",
			Resource: "trainingjobs",
		},
	}
}

// Run 启动控制循环 — Operator 的 main loop
func (c *Controller) Run(ctx context.Context) error {
	log.Println("🚀 Controller 启动，开始监听 TrainingJob")

	// Informer: K8s 的 Watch 机制封装
	// 它会: 1)首次 List 所有CR  2)之后 Watch 增量变化  3)本地缓存
	factory := dynamicinformer.NewDynamicSharedInformerFactory(c.dynamicClient, 30*time.Second)
	informer := factory.ForResource(c.tjResource).Informer()

	// 注册 3 个回调: 新增 / 更新 / 删除
	informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			u := obj.(*unstructured.Unstructured)
			log.Printf("📥 [ADD] TrainingJob: %s\n", u.GetName())
			c.onAdd(ctx, u)
		},
		DeleteFunc: func(obj interface{}) {
			u := obj.(*unstructured.Unstructured)
			log.Printf("🗑️  [DELETE] TrainingJob: %s\n", u.GetName())
			c.onDelete(ctx, u)
		},
	})

	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())
	log.Println("✅ 就绪，等待事件...")

	<-ctx.Done()
	return nil
}

// ============================================================
// onAdd — 用户创建了一个 TrainingJob，我们创建对应的 StatefulSet
// ============================================================
func (c *Controller) onAdd(ctx context.Context, u *unstructured.Unstructured) {
	name := u.GetName()
	namespace := u.GetNamespace()

	// 从 CR 里读 spec
	image, _, _ := unstructured.NestedString(u.Object, "spec", "image")
	workers, _, _ := unstructured.NestedInt64(u.Object, "spec", "workers")

	log.Printf("   image=%s, workers=%d\n", image, workers)

	// 幂等检查: StatefulSet 是否已存在？
	stsName := name + "-worker"
	_, err := c.kubeClient.AppsV1().StatefulSets(namespace).Get(ctx, stsName, metav1.GetOptions{})
	if err == nil {
		log.Printf("   StatefulSet %s 已存在，跳过\n", stsName)
		return
	}
	if !errors.IsNotFound(err) {
		log.Printf("❌ 查询失败: %v\n", err)
		return
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
					Containers: []corev1.Container{{
						Name:    "trainer",
						Image:   image,
						Command: []string{"python", "train.py"},
						Env: []corev1.EnvVar{
							{Name: "WORLD_SIZE", Value: fmt.Sprintf("%d", workers)},
							{Name: "MASTER_ADDR", Value: fmt.Sprintf("%s-0.%s-svc", stsName, name)},
						},
					}},
				},
			},
		},
	}

	_, err = c.kubeClient.AppsV1().StatefulSets(namespace).Create(ctx, sts, metav1.CreateOptions{})
	if err != nil {
		log.Printf("❌ 创建 StatefulSet 失败: %v\n", err)
	} else {
		log.Printf("✅ 创建 StatefulSet: %s (%d workers)\n", stsName, workers)
	}
}

// ============================================================
// onDelete — 用户删除了 TrainingJob，我们清理 StatefulSet
// ============================================================
func (c *Controller) onDelete(ctx context.Context, u *unstructured.Unstructured) {
	name := u.GetName()
	namespace := u.GetNamespace()
	stsName := name + "-worker"

	err := c.kubeClient.AppsV1().StatefulSets(namespace).Delete(ctx, stsName, metav1.DeleteOptions{})
	if err != nil && !errors.IsNotFound(err) {
		log.Printf("❌ 删除 StatefulSet 失败: %v\n", err)
	} else {
		log.Printf("✅ 已删除 StatefulSet: %s\n", stsName)
	}
}
