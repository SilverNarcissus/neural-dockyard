package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"training-operator/controller"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	log.Println("🎯 Training Operator 启动")

	// 连接 K8s: Pod 里用 ServiceAccount，本地开发用 ~/.kube/config
	config, err := rest.InClusterConfig()
	if err != nil {
		home, _ := os.UserHomeDir()
		config, err = clientcmd.BuildConfigFromFlags("", home+"/.kube/config")
	}
	if err != nil {
		log.Fatalf("❌ 连接K8s失败: %v\n", err)
	}

	kubeClient, _ := kubernetes.NewForConfig(config)
	dynamicClient, _ := dynamic.NewForConfig(config)

	// 启动 Controller
	ctrl := controller.New(kubeClient, dynamicClient)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Ctrl+C 优雅退出
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("收到退出信号")
		cancel()
	}()

	ctrl.Run(ctx)
}
