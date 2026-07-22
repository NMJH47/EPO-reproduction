# EPO Reorientation 复现实验

## 目标实验

仓库中 `result/reorientation/evolution_64` 对应 EPO-64 的 Reorientation 结果。参考运行采用：

- `AllegroKukaLSTM` 与 `reorientation`
- 24,576 个并行环境
- 64 个探索系数基因，每个基因 384 个环境
- minibatch 98,304，即启动器默认的 `4 * num_envs`
- entropy intrinsic reward，scale 为 0.005
- LSTM、`mixed_expl_learn_param`、`use_others_experience=lf`
- 图表在 50 亿 environment frames 截止
- 本次选择参考结果中已有的 seed 42

## 当前机器的缩放配置

RTX 4090 Laptop 只有 16,376 MiB 显存。24,576 环境甚至在使用更小 minibatch 的保守容量探针中也已于首轮训练触发 CUDA OOM；原始 98,304 minibatch 配置只会需要更多显存，因此不能在这张卡上做完全同规模复现。

保持算法、64 个基因、每个 epoch 的 minibatch 比例及其余超参数不变后，20,480 环境配置通过 GPU 容量测试：

- 20,480 个环境，为原规模的 83.3%
- 每个基因 320 个环境
- minibatch 81,920
- 实测首轮峰值显存 15,086 MiB
- Isaac Gym PhysX、simulation pipeline 与 RL device 均为 `cuda:0`

因此该运行属于硬件缩放复现，不应表述为 24,576 环境的完全等价复现。若换用至少 24GB 显存的 GPU，可设置 `EPO_NUM_ENVS=24576` 恢复原规模。

## 启动与监控

后台启动并在线同步到 W&B：

```bash
./scripts/reproduce_reorientation_epo64.sh start
```

查看 GPU、进程和最新训练日志：

```bash
./scripts/reproduce_reorientation_epo64.sh status
```

停止训练：

```bash
./scripts/reproduce_reorientation_epo64.sh stop
```

前台调试：

```bash
./scripts/reproduce_reorientation_epo64.sh foreground
```

默认 W&B 位置：

- entity: `leo-ucbm-university-of-california-berkeley`
- project: `EPO-reproduction`
- run: `uid_00_repro_epo64_seed42_N20480`

日志默认写入 `logs/reproduction/`。每 200 epochs 更新可续训的 `runs/<experiment>/last/model.pth`；脚本再次启动时会自动检测并加载该 checkpoint，同时续接相同 W&B run。

可通过环境变量覆盖参数，例如：

```bash
EPO_NUM_ENVS=16384 EPO_SEED=12345 WANDB_PROJECT=EPO-reproduction \
  ./scripts/reproduce_reorientation_epo64.sh start
```
