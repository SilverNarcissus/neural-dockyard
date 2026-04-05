package pipeline

import (
	"fmt"
	"log"
	"time"
)

// ============================================================
// Pipeline — Go 版 Future 链
// ============================================================
// 类似 Strimzi 的 Future.compose() 链:
//   reconcile()
//       .compose(reconcileCas)
//       .compose(reconcileKafka)
//       .recover(handleError)
//
// Go 版:
//   pipeline.New("create-job").
//       Step("create-service", createService).
//       Step("create-statefulset", createStatefulSet).
//       OnError(handleError).
//       Run()
// ============================================================

// StepFunc 单个步骤的函数签名
// 返回 error 表示失败，nil 表示成功
type StepFunc func() error

// step 一个步骤
type step struct {
	name string
	fn   StepFunc
}

// Pipeline 步骤链
type Pipeline struct {
	name    string
	steps   []step
	onError func(stepName string, err error) // 统一错误处理
}

// New 创建 pipeline
func New(name string) *Pipeline {
	return &Pipeline{name: name}
}

// Step 添加一个步骤
func (p *Pipeline) Step(name string, fn StepFunc) *Pipeline {
	p.steps = append(p.steps, step{name: name, fn: fn})
	return p
}

// OnError 设置统一错误处理（类似 .recover()）
func (p *Pipeline) OnError(fn func(stepName string, err error)) *Pipeline {
	p.onError = fn
	return p
}

// Run 执行 pipeline，任意一步失败则停止
func (p *Pipeline) Run() error {
	log.Printf("▶️  Pipeline [%s] 开始 (%d 步)\n", p.name, len(p.steps))
	start := time.Now()

	for i, s := range p.steps {
		log.Printf("   [%d/%d] %s ...\n", i+1, len(p.steps), s.name)

		if err := s.fn(); err != nil {
			log.Printf("   ❌ [%d/%d] %s 失败: %v\n", i+1, len(p.steps), s.name, err)

			if p.onError != nil {
				p.onError(s.name, err)
			}

			return fmt.Errorf("pipeline [%s] 在步骤 [%s] 失败: %w", p.name, s.name, err)
		}

		log.Printf("   ✅ [%d/%d] %s 完成\n", i+1, len(p.steps), s.name)
	}

	elapsed := time.Since(start)
	log.Printf("▶️  Pipeline [%s] 全部完成 (耗时 %v)\n", p.name, elapsed)
	return nil
}
