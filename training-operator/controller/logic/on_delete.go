package logic

import (
	"context"
	"log"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// OnDelete 用户删除了 TrainingJob → 清理 StatefulSet + Service
func OnDelete(ctx context.Context, kubeClient kubernetes.Interface, u *unstructured.Unstructured) {
	name := u.GetName()
	namespace := u.GetNamespace()

	stsName := name + "-worker"
	kubeClient.AppsV1().StatefulSets(namespace).Delete(ctx, stsName, metav1.DeleteOptions{})
	log.Printf("✅ 已删除 StatefulSet: %s\n", stsName)

	svcName := name + "-svc"
	kubeClient.CoreV1().Services(namespace).Delete(ctx, svcName, metav1.DeleteOptions{})
	log.Printf("✅ 已删除 Service: %s\n", svcName)
}
