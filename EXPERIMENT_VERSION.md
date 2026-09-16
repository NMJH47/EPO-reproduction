# 历史最高完整评估快照：frontier 2B

分支：`archive/historical-best-frontier-2b`。保存日期：2026-09-16。

本分支保存已完成实验 `112_long_frontier_seed36001_N20480` 的实际源码和配置。
在已核验、可比较的完整947对象评估中，固定master63的两个评估seed均值为
**41.5042238648**，对应control为 **41.0491024287**。这是历史观测最高快照；
仅有一个共同E51祖先和一个continuation seed，不能据此确立跨训练seed的算法优势。

## 源码与行为

- 继承E51：F1全种群fitness、niche-local8父代、off-policy actor support门、off-policy critic系数0、原位gene更新和原AMP裁剪行为。
- frontier课程开启：3×3目标位置区域，warmup306轮，cohort64轮，maturity600步，原分布与自适应提案各占0.5。
- actor保持原ratio裁剪区间`[0.9, 1.1]`；没有启用Clip-Higher。
- 训练入口：`scripts/research/run_long_budget_training.py`。
- 课程实现：`scripts/research/frontier_curriculum.py`。

## 训练与结果身份

- E51祖先：frame5,000,069,120，训练seed43。
- continuation seed36001，连续6104轮，新增2,000,158,720帧，终点frame7,000,227,840。
- 训练W&B：[f18cf118ea3744d3](https://wandb.ai/leo-ucbm-university-of-california-berkeley/EPO-reproduction/runs/f18cf118ea3744d3)。
- 评估：master63、全947对象首episode、随机动作、tolerance0.01、cap50，eval seed17001/17002，全部无删失。
- 最终checkpoint SHA256：`29f3b33d2b0d80daa923aa9dc65538d3f1220e204f91e05d253878277b85276c`。

[版本清单](experiments/snapshots/historical-best/version.json)绑定实际源码、执行配置、评估摘要和原始记录位置；
[checkpoint清单](experiments/snapshots/historical-best/checkpoints.json)保存四个里程碑身份；
[终点artifact记录](experiments/snapshots/historical-best/terminal_model_artifact.json)给出权重归档身份。
模型权重、原始episode CSV和训练日志保留在既有本地/W&B归档，不加入Git。

## 使用范围

这是可审查的源码与实验身份快照，保存动作不启动训练。
归档计划保留当时的绝对路径、run ID和输出目录；直接重跑旧计划会被入口的已有输出检查拒绝。
若安排新实验，应在新运行身份下重绑定路径和SHA，使用`epo`环境与CUDA GPU，
并先取得checkpoint清单指向的E51祖先权重。重绑定后的实验不能冒充历史原运行。

## 验证

535份训练时源码全部与原`source_manifest.json`逐字节相同。
额外归档配置、摘要和来源，共560份文件通过逐字节核验；两组各两个seed的原始CSV
均通过SHA验证并重新计算均值，与保存摘要一致。本次没有重新训练或改写历史结果。
