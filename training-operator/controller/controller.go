package controller

import (
	"context"
	"log"
	"sync"
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

// Controller 核心结构
type Controller struct {
	KubeClient    kubernetes.Interface
	DynamicClient dynamic.Interface
	TjResource    schema.GroupVersionResource
	locks         sync.Map // per-job 锁: 同一个 Job 不会并发 reconcile
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

// lockFor 获取指定 job 的互斥锁
func (c *Controller) lockFor(name string) *sync.Mutex {
	v, _ := c.locks.LoadOrStore(name, &sync.Mutex{})
	return v.(*sync.Mutex)
}

// Run 启动控制循环
func (c *Controller) Run(ctx context.Context) error {
	log.Println("🚀 Controller 启动，开始监听 TrainingJob")

	factory := dynamicinformer.NewDynamicSharedInformerFactory(c.DynamicClient, 30*time.Second)
	informer := factory.ForResource(c.TjResource).Informer()

	informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			u := obj.(*unstructured.Unstructured)
			name := u.GetName()
			log.Printf("📥 [ADD] TrainingJob: %s\n", name)

			// goroutine 异步执行 + per-job 锁防并发
			go func() {
				mu := c.lockFor(name)
				mu.Lock()
				defer mu.Unlock()
				logic.OnAdd(ctx, c.KubeClient, u)
			}()
		},
		DeleteFunc: func(obj interface{}) {
			u := obj.(*unstructured.Unstructured)
			name := u.GetName()
			log.Printf("🗑️  [DELETE] TrainingJob: %s\n", name)

			go func() {
				mu := c.lockFor(name)
				mu.Lock()
				defer mu.Unlock()
				logic.OnDelete(ctx, c.KubeClient, u)
				c.locks.Delete(name) // 清理锁
			}()
		},
	})

	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())
	log.Println("✅ 就绪，等待事件...")

	go watcher.WatchCompletion(ctx, c.KubeClient, c.DynamicClient, c.TjResource)

	<-ctx.Done()
	return nil
}
