package logic

import (
	"context"
	"log"

	"training-operator/controller/pipeline"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/kubernetes"
)

// OnDelete 用户删除了 TrainingJob → 用 Pipeline 链式清理子资源
func OnDelete(ctx context.Context, kubeClient kubernetes.Interface, u *unstructured.Unstructured) {
	name := u.GetName()
	namespace := u.GetNamespace()

	pipeline.New("delete-"+name).
		Step("delete-statefulset", func() error {
			return kubeClient.AppsV1().StatefulSets(namespace).Delete(ctx, name+"-worker", metav1.DeleteOptions{})
		}).
		Step("delete-service", func() error {
			return kubeClient.CoreV1().Services(namespace).Delete(ctx, name+"-svc", metav1.DeleteOptions{})
		}).
		OnError(func(stepName string, err error) {
			log.Printf("⚠️  清理 %s 时步骤 [%s] 失败，可能资源已被删除\n", name, stepName)
		}).
		Run()
}
