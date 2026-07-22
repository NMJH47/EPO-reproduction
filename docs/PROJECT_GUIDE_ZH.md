# EPO 项目与论文导览

本文档对应仓库 `YifanSu1301/EPO` 的 `main` 分支提交
`17c48efab01bb73ee2d359f3eab7b6268070f599`，并对照 arXiv 论文
《Evolutionary Policy Optimization》当前的 v3 版本。仓库提交时间早于论文 v3，二者并非完全一致。

## 1. 项目来源与组成

仓库不是从零实现的单一算法包，而是四层代码的组合：

1. **IsaacGymEnvs**：任务、GPU 物理仿真、Hydra 配置、域随机化和向量环境。
2. **rl_games 1.6.1**：PPO/A2C、SAC、网络构造、经验缓冲、数据集和推理器。
3. **SAPG**：把并行环境分成多个策略块，并让主策略复用追随者的当轮轨迹。
4. **EPO 增量**：可学习 latent gene、适应度统计，以及对 gene 的选择和交叉。

与 SAPG 上游逐文件比较后，EPO 的主要新增算法代码只有两处：

- `isaacgymenvs/utils/rlgames_utils.py` 中的 `EPOObserver`。
- `rl_games/rl_games/algos_torch/network_builder.py` 中 gene 的合并、复制和可视化方法。

训练数据聚合、PPO、条件方差和 latent-conditioned policy 的主体来自 SAPG/rl_games。

## 2. 目录职责

| 路径 | 职责 |
| --- | --- |
| `isaacgymenvs/cfg/` | Hydra 根配置、53 个任务配置、45 个训练配置及 PBT 配置 |
| `isaacgymenvs/tasks/` | 任务实现、机器人/物体状态张量、奖励、重置、地形和 AMP 动作数据工具 |
| `isaacgymenvs/tasks/base/vec_task.py` | 所有 Isaac Gym 任务的统一向量环境生命周期 |
| `isaacgymenvs/learning/` | AMP、HRL 及其模型、数据集、回放缓冲和 player |
| `isaacgymenvs/pbt/` | 独立多进程 PBT、变异规则、checkpoint 交换和本地/Slurm/NGC launcher |
| `isaacgymenvs/utils/` | rl_games 适配、EPO observer、域随机化、四元数/JIT 数学和 WandB |
| `rl_games/rl_games/` | PPO/A2C/SAC 核心、模型/网络工厂、经验缓冲、数据集、调度器和环境适配器 |
| `assets/` | 机器人、物体、地形、AMP motion、URDF/MJCF/mesh 及其许可证 |
| `scripts/` | 论文任务的启动脚本；最终都会调用 PBT launcher |
| `result/` | 仓库附带的训练曲线 CSV，不是运行时依赖 |

`rl_games/rl_games/envs/` 中的 Brax、EnvPool、SMAC、Atari 等适配器是 vendored
rl_games 的通用能力，不属于 EPO 复现实验的必需依赖。

## 3. 从命令到梯度更新

完整调用链如下：

1. `scripts/train_*.sh` 用 `nvidia-smi` 计算 GPU 数，并调用
   `python -m isaacgymenvs.pbt.launcher.run`。
2. `experiment_args.py` 把普通参数转换为 Hydra override。`--epo` 会强制设置：
   - `expl_type=mixed_expl_learn_param`
   - `use_others_experience=lf`
   - `fixed_sigma=coef_cond`
   - `train.params.config.epo=True`
3. `isaacgymenvs/train.py` 组合 `cfg/config.yaml`、任务配置和训练配置，注册
   `RLGPU` 向量环境，并在 EPO 模式下注入 `EPOObserver`。
4. `isaacgymenvs.make()` 根据 `isaacgym_task_map` 实例化任务。
5. `VecTask.step()` 依次执行动作裁剪、`pre_physics_step()`、PhysX 仿真、
   `post_physics_step()`，再返回 observation/reward/done/info 张量。
6. `rl_games.torch_runner.Runner` 创建连续动作 `A2CAgent`。
7. `A2CBase.play_steps()` 收集一个 horizon，计算 GAE/return，并把轨迹交给
   `PPODataset`。
8. `A2CAgent.calc_gradients()` 计算 clipped PPO actor loss、critic loss、entropy
   和 action-bound loss，再由 Adam 更新共享网络与 latent gene。

## 4. EPO/SAPG 数据布局

令并行环境数为 `N`，每个 gene 的环境块大小为 `B=expl_coef_block_size`，则
agent 数为 `K=N/B`。论文默认 `N=24,576`、`K=64`、`B=384`。

初始化时，每个连续环境块得到一个标识值。`A2CBuilder.Network` 不把这个标识
直接作为特征，而是查表取得对应的可学习向量 `extra_params[k]`，再将 gene 与
原 observation 拼接后送入共享 actor-critic。动作均值共享同一网络，动作标准差
在 `coef_cond` 模式下按 agent 单独查表。

一次更新的数据由两部分构成：

- **所有 agent 的 on-policy 数据**：原始 rollout 全部保留，每个 agent 因而使用
  自己的轨迹更新共享参数和自己的 gene。
- **master 的 follower 数据**：随机选取 follower 块，把 observation 中的标识
  重标记为 master，然后附加到 batch。off-policy return 使用一步 bootstrap；
  actor 仍通过保存的 behavior log-prob 与当前 log-prob 进入 PPO ratio/clipping。

代码没有持久化论文伪码中的 `K` 个 cyclic replay buffer；它复用的是当前 horizon
中的 follower 数据。

## 5. 仿真任务族

- **基础控制**：Cartpole、Ant、Humanoid、Anymal、Quadcopter、Ingenuity 等直接
  继承 `VecTask`，实现建模、观测、奖励和 reset。
- **灵巧操作**：Allegro Hand、Shadow Hand、Trifinger，以及单/双臂
  Allegro-Kuka。Kuka 的共同控制、状态和课程学习位于 base 类，regrasping、
  throwing、reorientation 子类定义目标与 `true_objective`。
- **Factory/IndustReal**：使用 base/env/task 三层多继承，将机器人资产与具体装配
  目标分离，并支持 asymmetric critic。
- **Dextreme**：在 Allegro Hand 上增加 ADR worker mode、参数边界扩张和手工 DR。
- **AMP/HRL**：Humanoid AMP 使用 motion library 和 discriminator reward；HRL
  让高层策略驱动已训练的低层控制器。
- **Parkour**：由 A1 机器人、程序化 terrain、深度相机/history observation、
  课程学习及大量 reward term 组成，并额外依赖 `pydelatin`、`pyfqmr`、`kornia`。

## 6. 论文方法

论文维护共享参数 `(theta, psi)` 和每个 agent 的 gene `phi_k`，指定 `pi_1` 为
master。其主要步骤为：

1. 每个 agent 在自己的 `N/K` 个环境上获得 fitness。
2. 当 fitness spread 满足
   `max(f)-min(f) > gamma * median(f)`（或固定间隔）时执行遗传操作。
3. 保留 top-`x` elites；随机选择两个 elite，以
   `phi'=(phi_i+phi_j)/2` 交叉，再加高斯噪声 `epsilon` 变异。
4. 所有 agent 用自身 on-policy loss 更新；master 额外使用 follower 数据的
   importance-corrected off-policy loss，权重为 `lambda`。
5. 论文写明 on-policy critic 使用 3-step target，off-policy critic使用 1-step
   target。

论文在 5 个 manipulation、ANYmal、A1 Parkour 和 DMC Humanoid 共 8 个任务上报告
5 seeds。默认使用 24,576 个环境和 64 个 agent，并在固定每 agent 384 个环境时，
把规模从 16/6,144 扩展到 128/49,152。论文报告 EPO-64 约 6,764 FPS、约 24 GB
显存；该数字意味着标准配置不适合显存小于 24 GB 的单卡直接复现。

## 7. 论文与公开代码的差异

这些差异直接来自当前 v3 论文和当前 `main` 源码的逐项核对：

| 论文描述 | 当前代码 |
| --- | --- |
| top-`x` elite 生成完整下一代，实验中 `x=K-2` | 固定取最优两个 gene，只替换一个最差 gene |
| 交叉后添加高斯 mutation | 只有精确算术平均，没有 mutation |
| 用 Eq.3 fitness spread 触发，或固定间隔 | 只用固定 200M warm-up + 每 20M steps |
| `K` 个 cyclic replay buffers | 只重用当前 rollout，没有独立 cyclic buffer |
| on-policy critic 为 3-step target | 实现使用 PPO 的 GAE；off-policy 才显式为 1-step |
| 8 个任务的完整复现 | README 仅给 5 个 manipulation 命令；launcher 没有 ANYmal 入口 |

另外还有下列上游算法实现风险；本次环境兼容补丁没有改变这些行为：

1. `merge_extra_params()` 通过重新赋值 `nn.Parameter` 替换整个 gene 表。Adam 在此前
   已经持有旧 Parameter，因此第一次遗传替换后，新 gene 会获得梯度但不会被
   optimizer step 更新。最小实测结果为 `new_in_optimizer=False`、
   `new_changed_by_step=False`。
2. mixed exploration 把 `ignore_env_boundary` 设为 `N-B`。`EPOObserver` 因而先丢弃
   前 63 个块，只保留最后 384 个环境，再将这 384 个值平均切成 64 份（每份 6 个
   环境）当作 64 个 agent 的 fitness；这不对应 `64 x 384` 的真实块布局。
3. `finished_agents` 和旧 objective 在遗传替换后没有清空，后续代可能继续混入旧值。
4. observer 初始化直接访问 `infos['true_objective']`，因此后来写出的 reward fallback
   实际无法覆盖不提供该字段的任务；原始 Anymal/Humanoid 就没有该字段。
5. launcher 固定生成 `headless=False`；`train.py` 中 WandB observer 被注释，所以
   README 的 WandB 参数不会真正启用记录。
6. `--sapg` 参数仍被解析，但条件已改成只检查 `--epo`，因此单独传 `--sapg` 不会再
   自动设置 SAPG 的三个关键 override。

本机分支只额外修复了两个运行层问题：CUDA 11.1 在 RTX 4090 上生成 TorchScript
融合内核时不认识 `sm_89`，以及普通训练结束返回的二元组被入口误判为 PBT 重启。
两项修复都位于 `isaacgymenvs/train.py`，不修改 EPO/SAPG 损失或遗传逻辑。

因此，公开代码足以说明 EPO 原型如何在 SAPG 上加入 latent evolution，也能运行其
公开的 manipulation 配置；但不能把它视为论文 v3 所有算法细节和 8 个任务的完整、
无差异发布版。若目标是严格复现实验，应先修复上述问题并向作者确认最终训练代码。

## 8. 关键入口索引

- 训练入口：`isaacgymenvs/train.py`
- 环境注册：`isaacgymenvs/tasks/__init__.py`
- 仿真生命周期：`isaacgymenvs/tasks/base/vec_task.py`
- EPO observer：`isaacgymenvs/utils/rlgames_utils.py`
- SAPG rollout 聚合：`rl_games/rl_games/common/a2c_common.py`
- PPO loss：`rl_games/rl_games/algos_torch/a2c_continuous.py`
- gene 与共享网络：`rl_games/rl_games/algos_torch/network_builder.py`
- runner：`rl_games/rl_games/torch_runner.py`
- PBT（独立基线）：`isaacgymenvs/pbt/pbt.py`

论文：https://arxiv.org/abs/2503.19037
项目页：https://yifansu1301.github.io/EPO/
代码：https://github.com/YifanSu1301/EPO
