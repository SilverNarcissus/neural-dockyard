package watcher

import (
	"context"
	"log"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// WatchCompletion 定期检查训练任务是否完成，完成后自动清理
func WatchCompletion(ctx context.Context, kubeClient kubernetes.Interface, dynamicClient dynamic.Interface, tjResource schema.GroupVersionResource) {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			log.Println("🔍 检查训练任务完成状态...")
			checkAllJobs(ctx, kubeClient, dynamicClient, tjResource)
		}
	}
}

func checkAllJobs(ctx context.Context, kubeClient kubernetes.Interface, dynamicClient dynamic.Interface, tjResource schema.GroupVersionResource) {
	// 列出所有 TrainingJob
	tjList, err := dynamicClient.Resource(tjResource).Namespace("default").List(ctx, metav1.ListOptions{})
	if err != nil {
		log.Printf("⚠️  列出 TrainingJob 失败: %v\n", err)
		return
	}

	for _, tj := range tjList.Items {
		name := tj.GetName()
		namespace := tj.GetNamespace()
		stsName := name + "-worker"
		checkpointPVC, _, _ := unstructured.NestedString(tj.Object, "spec", "checkpointPVC")

		// 获取该任务的所有 Pod
		pods, err := kubeClient.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: "job=" + name,
		})
		if err != nil {
			log.Printf("⚠️  列出 Pod 失败: %v\n", err)
			continue
		}
		if len(pods.Items) == 0 {
			continue
		}

		// 打印 Pod 状态便于调试
		for _, pod := range pods.Items {
			restarts := int32(0)
			exitCode := int32(-1)
			for _, cs := range pod.Status.ContainerStatuses {
				restarts = cs.RestartCount
				if cs.LastTerminationState.Terminated != nil {
					exitCode = cs.LastTerminationState.Terminated.ExitCode
				}
			}
			log.Printf("   Pod %s: phase=%s, restarts=%d, lastExitCode=%d\n",
				pod.Name, pod.Status.Phase, restarts, exitCode)
		}

		// 检查是否所有 Pod 都已完成
		// StatefulSet 的 restartPolicy=Always，Pod 不会变成 Succeeded
		// 所以检查容器的上次退出码: exit 0 = 训练成功完成后被重启了
		allCompleted := true
		for _, pod := range pods.Items {
			if pod.Status.Phase == corev1.PodSucceeded {
				continue // Job 类型的 Pod 会直接 Succeeded
			}
			// StatefulSet Pod: 检查容器是否曾经成功退出过
			completed := false
			for _, cs := range pod.Status.ContainerStatuses {
				// 容器上次退出码为 0 = 训练正常完成
				if cs.LastTerminationState.Terminated != nil && cs.LastTerminationState.Terminated.ExitCode == 0 {
					completed = true
					break
				}
			}
			if !completed {
				allCompleted = false
				break
			}
		}

		if !allCompleted {
			continue
		}

		// 🎉 训练完成!
		log.Println("============================================================")
		log.Printf("🎉 训练任务完成: %s\n", name)
		log.Printf("   Workers: %d, 全部 Succeeded\n", len(pods.Items))
		if checkpointPVC != "" {
			log.Printf("   📁 Checkpoint 存储: PVC=%s, 路径=/checkpoints/%s/\n", checkpointPVC, name)
		}
		log.Println("============================================================")

		// 清理子资源
		kubeClient.AppsV1().StatefulSets(namespace).Delete(ctx, stsName, metav1.DeleteOptions{})
		log.Printf("   🧹 已清理 StatefulSet: %s\n", stsName)

		svcName := name + "-svc"
		kubeClient.CoreV1().Services(namespace).Delete(ctx, svcName, metav1.DeleteOptions{})
		log.Printf("   🧹 已清理 Service: %s\n", svcName)

		// 删除 TrainingJob CR 本身
		dynamicClient.Resource(tjResource).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
		log.Printf("   🧹 已清理 TrainingJob CR: %s\n", name)
	}
}
