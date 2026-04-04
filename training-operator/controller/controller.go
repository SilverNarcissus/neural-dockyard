package controller

import (
	"context"
	"log"
	"time"

	"training-operator/controller/logic"
	"training-operator/controller/watcher"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

// Controller 核心结构 — 持有 K8s 客户端，串联各模块
type Controller struct {
	KubeClient    kubernetes.Interface
	DynamicClient dynamic.Interface
	TjResource    schema.GroupVersionResource
}

func New(kubeClient kubernetes.Interface, dynamicClient dynamic.Interface) *Controller {
	return &Controller{
		KubeClient:    kubeClient,
		DynamicClient: dynamicClient,
		TjResource: schema.GroupVersionResource{
			Group:    "training.megatron.io",
			Version:  "v1",
			Resource: "trainingjobs",
		},
	}
}

// Run 启动控制循环
func (c *Controller) Run(ctx context.Context) error {
	log.Println("🚀 Controller 启动，开始监听 TrainingJob")

	// Informer: 监听 TrainingJob CR 的增删事件
	factory := dynamicinformer.NewDynamicSharedInformerFactory(c.DynamicClient, 30*time.Second)
	informer := factory.ForResource(c.TjResource).Informer()

	// 注册事件回调 → 分发到 logic 包
	informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			u := obj.(*unstructured.Unstructured)
			log.Printf("📥 [ADD] TrainingJob: %s\n", u.GetName())
			logic.OnAdd(ctx, c.KubeClient, u)
		},
		DeleteFunc: func(obj interface{}) {
			u := obj.(*unstructured.Unstructured)
			log.Printf("🗑️  [DELETE] TrainingJob: %s\n", u.GetName())
			logic.OnDelete(ctx, c.KubeClient, u)
		},
	})

	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())
	log.Println("✅ 就绪，等待事件...")

	// 启动完成检测循环
	go watcher.WatchCompletion(ctx, c.KubeClient, c.DynamicClient, c.TjResource)

	<-ctx.Done()
	return nil
}
