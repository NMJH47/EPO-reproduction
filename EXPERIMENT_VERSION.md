# DAPO Clip-Higher 长预算实验快照

分支：`exp/dapo-clip-higher-2b`。保存日期：2026-09-16。

本分支保存实验 `118_structural_clip_higher_seed36001_N20480` 的实际执行源码和配置。
训练已于2026-09-16 09:20北京时间按用户要求停止；原父进程记录实际退出码`-15`。
它是尚未完成的实验版本，当前没有完整2B效果结论。此次保存不恢复训练。

## 与历史最高快照的配置区别

| 项目 | 历史最高frontier 2B | 当前DAPO Clip-Higher |
|---|---|---|
| actor ratio下界 | 0.9 | 0.9 |
| actor ratio上界 | 1.1 | 1.2 |
| frontier课程 | 开启 | 关闭，仅被动记录 |
| critic clip与LF support门 | 原实现 | 原实现 |
| 梯度裁剪参数集合 | 全model | 全model |
| 同次稀疏norm探针 | 无本轮新增hook | 开启 |
| 训练起点 | E51 5B checkpoint | 同一E51 5B checkpoint |

Git分支以历史快照提交为代码父节点，便于查看新增执行入口和配置差异；
**训练血缘仍是E51，而非frontier终点权重**。Clip-Higher与owned-only clipping
是原队列的独立实验，本分支选中的`clip_higher`臂没有同时启用owned-only clipping。

## 源码位置

- actor实现：`scripts/research/dapo_actor.py`的`actor_loss_asymmetric`。
- 长训练入口：`scripts/research/run_structural_training.py`。
- 同次稀疏norm探针：`scripts/research/clip_scope_hooks.py`。
- 原队列入口与收尾：`scripts/research/run_structural_study.py`、`_research_support/monitor_job_consistent.py`。
- [实际执行计划](experiments/snapshots/dapo-clip-higher/plan.json)与[Hydra配置](experiments/snapshots/dapo-clip-higher/config.yaml)。

以上入口与隔离运行目录中实际执行的SHA绑定副本逐字节相同。
`experiments/snapshots/dapo-clip-higher/bound/`保留实际执行位置对应的原件，
[版本清单](experiments/snapshots/dapo-clip-higher/version.json)给出原路径、提交路径与SHA映射。

## 训练与停止证据

- W&B：[e1e2f5f3f82e41a9](https://wandb.ai/leo-ucbm-university-of-california-berkeley/EPO-reproduction/runs/e1e2f5f3f82e41a9)。
- E51祖先frame5,000,069,120，祖先训练seed43，continuation seed36001。
- 原计划：连续6104轮，新增2,000,158,720帧。
- 实际返回1623个完整epoch；1624轮已进入，环境step累计返回532,152,320帧。
- 已保存100,270,080和500,039,680新增帧的里程碑；没有2B终点checkpoint。
- [checkpoint身份](experiments/snapshots/dapo-clip-higher/checkpoints.json)、[实际训练摘要](experiments/snapshots/dapo-clip-higher/training_summary.json)、[用户停止收据](experiments/snapshots/dapo-clip-higher/user_stop_receipt.json)。

权重和大体积原始训练记录保留在既有本地/W&B归档，不加入Git。
归档计划保留旧绝对路径、run ID和输出目录；直接重跑会被已有输出检查拒绝。
新运行必须使用新身份、明确的起点权重与重新绑定的源码/配置SHA。
本次仅进行版本保存、文件身份核验和语法检查，没有安排新的训练或评估。

## 核验结果

包含535份原核心源码的562个快照文件逐字节通过。
新增入口、norm hook和监控helper与真实执行副本逐字节匹配；两个已保存checkpoint
及E51祖先的SHA通过。历史最高结果依据仍在`experiments/snapshots/historical-best/`。
