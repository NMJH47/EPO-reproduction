# EPO 实验驱动开发流程

## 基线定义

- 算法基线固定为提交 `a0c5ec0`：上游当前设计、GPU 环境引导和 W&B 可靠性修复。
- CI、安全配置和文档可以进入 `main`，但不得改变算法基线行为。
- 每个实验必须记录代码提交、种子、环境数量、帧预算、配置、GPU、W&B run URL 和终止状态。
- 论文与代码都只作为待验证假设；合并依据是可重复实验结果。

## GPU 约束

- Isaac Gym、训练、推理和算法测试只允许 NVIDIA CUDA，不允许 CPU fallback。
- GitHub 托管 runner 仅运行语法、仓库策略和 W&B 控制面测试，不作为算法正确性证据。
- GPU smoke workflow 只运行在带有 `self-hosted, linux, x64, gpu, nvidia` 标签的 runner。

## 分支和 PR

1. 从最新验证过的 `main` 创建 `agent/<problem>` 分支。
2. 一个分支只处理一个问题，不合并无关重构。
3. 修改前记录能复现缺陷的 baseline 指标。
4. 使用相同种子、环境数量、帧预算和评估方法运行 candidate。
5. 在 PR 中填写 baseline/candidate 的提交和 W&B 链接。
6. 只有缺陷被证实、修复有效且没有回归时才合并。
7. 合并后将远端 `main` 同步回本地，再开始下一个问题。

## 优先顺序

1. 评估 gene 选择和结果聚合口径。
2. gene 参数与优化器生命周期。
3. GA 适应度来源。
4. master 保护。
5. 演化与 rollout 的时序。
6. 论文 Eq.(4) 的偏策略修正。
7. LSTM 偏策略隐藏状态。
8. MLP/LSTM 顺序、GA 算子和 critic target 消融。

## 合并门禁

- `CI`：Python 编译、shell 语法、仓库策略和 W&B 单元测试。
- `CodeQL`：Python 扩展安全查询。
- `Dependency Review`：拒绝 PR 新增的高危依赖。
- `Secret Scan`：扫描提交历史中的凭据。
- GPU 算法结果：由本机或自托管 NVIDIA runner 提供，并通过 W&B 保存完整证据。
