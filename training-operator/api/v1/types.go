package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ============================================================
// CRD 类型定义: TrainingJob
// ============================================================
// 用户通过提交 TrainingJob CR 来描述一个分布式训练任务
// Controller 会根据 spec 自动创建 Headless Service + StatefulSet
//
// 对应 JD: "参与AI平台本身各项功能以及用户训练任务"
// 类似 Kubeflow 的 PyTorchJob，但简化为学习用途
// ============================================================

const (
	// CRD 元信息
	Group    = "training.megatron.io"
	Version  = "v1"
	Kind     = "TrainingJob"
	Plural   = "trainingjobs"
	Singular = "trainingjob"
	ShortName = "tj"
)

// TrainingJobSpec 用户提交训练任务时填写的配置
type TrainingJobSpec struct {
	// 镜像地址 (包含训练代码和依赖)
	Image string `json:"image"`

	// Worker 数量 (即 world_size，每个 worker 对应一个 rank)
	Workers int32 `json:"workers"`

	// 启动命令 (如 "python train.py")
	Command []string `json:"command"`

	// 启动参数
	Args []string `json:"args,omitempty"`

	// GPU 需求 (每个 worker 几张卡)
	GPUsPerWorker int32 `json:"gpusPerWorker,omitempty"`

	// 训练框架端口 (默认 29500, PyTorch rendezvous)
	MasterPort int32 `json:"masterPort,omitempty"`

	// Checkpoint 存储路径 (PVC 名称)
	CheckpointPVC string `json:"checkpointPVC,omitempty"`

	// 资源配置
	Resources *ResourceSpec `json:"resources,omitempty"`
}

// ResourceSpec 每个 worker 的资源限制
type ResourceSpec struct {
	CPURequest    string `json:"cpuRequest,omitempty"`
	CPULimit      string `json:"cpuLimit,omitempty"`
	MemoryRequest string `json:"memoryRequest,omitempty"`
	MemoryLimit   string `json:"memoryLimit,omitempty"`
}

// TrainingJobPhase 训练任务的生命周期阶段
type TrainingJobPhase string

const (
	TrainingJobPending   TrainingJobPhase = "Pending"   // 刚创建，等待调度
	TrainingJobCreating  TrainingJobPhase = "Creating"  // 正在创建子资源 (Service, StatefulSet)
	TrainingJobRunning   TrainingJobPhase = "Running"   // 训练进行中
	TrainingJobSucceeded TrainingJobPhase = "Succeeded" // 训练完成
	TrainingJobFailed    TrainingJobPhase = "Failed"    // 训练失败
)

// TrainingJobStatus Controller 维护的任务状态
type TrainingJobStatus struct {
	// 当前阶段
	Phase TrainingJobPhase `json:"phase,omitempty"`

	// 就绪的 worker 数
	ReadyWorkers int32 `json:"readyWorkers,omitempty"`

	// 开始时间
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// 完成时间
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// 状态消息 (供用户查看)
	Message string `json:"message,omitempty"`
}

// TrainingJob CRD 主结构
type TrainingJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TrainingJobSpec   `json:"spec"`
	Status TrainingJobStatus `json:"status,omitempty"`
}

// TrainingJobList 列表
type TrainingJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []TrainingJob `json:"items"`
}

// SetDefaults 填充默认值
func (tj *TrainingJob) SetDefaults() {
	if tj.Spec.MasterPort == 0 {
		tj.Spec.MasterPort = 29500
	}
	if tj.Spec.GPUsPerWorker == 0 {
		tj.Spec.GPUsPerWorker = 0 // CPU 训练
	}
	if tj.Spec.Resources == nil {
		tj.Spec.Resources = &ResourceSpec{
			CPURequest:    "250m",
			CPULimit:      "500m",
			MemoryRequest: "512Mi",
			MemoryLimit:   "1Gi",
		}
	}
}
