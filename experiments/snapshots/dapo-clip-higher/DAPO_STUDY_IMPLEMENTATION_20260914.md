# DAPO 向 EPO 的迁移：实现与完整实验安排

本记录描述本次实现和事前固定的实验，不宣称已经提高 success。50 是任务实现的逐回合计分上限；它不构成当前策略类别、观测条件和固定评估分布下期望值可达50的证明。

## 文献与迁移判断

DAPO 的 Dynamic Sampling 在同一问题的多个完整回答间寻找非退化的奖励组，重采样补足有效训练批次；它与组内中心化优势相配套。EPO 使用连续控制、GAE、共享价值网络、RNN、GA population 和 LF 行为数据复用。同一个物体类别不等于同一个初始物理状态、行为gene与RNN历史；16步片段的success相同也不意味着优势为零。原地删除这些片段会改变状态访问与目标权重，无法从DAPO直接推出梯度等价性。

本次实现选择 DAPO 的另一项明确机制 Clip-Higher，先用固定策略同一次前向的影子探针核验是否确实存在被截断的有效正优势信号。完整推导、三种迁移路径及局限见 `artifacts/research/20260914-dapo-transfer/literature/REPORT.md` 和种子补充 `ADDENDUM.md`。

一手来源：[DAPO论文v2](https://arxiv.org/abs/2503.14476v2)、[作者官方代码](https://github.com/BytedTsinghua-SIA/DAPO)。连续控制中的效果仍需本项目实验验证；现有方法迁移本身不是新的算法贡献。

## 修改前与修改后

原生产核心文件保持本次任务开始时的数学实现。新入口 `scripts/research/run_dapo_continuation.py` 对候选agent实例安装actor函数；对照继续使用原函数对象。完整新增文件可通过本次交付的补丁复制，不能通过修改共享 `e_clip` 达到同一干预。

修改前，actor 使用原 `common_losses.actor_loss`，等价核心为：

```python
ratio = torch.exp(old_neglogp - new_neglogp)
loss = torch.maximum(-advantage * ratio,
                     -advantage * torch.clamp(ratio, 0.9, 1.1))
```

修改后，候选入口调用 `scripts/research/dapo_actor.py` 的函数：

```python
loss = actor_loss_asymmetric(torch, old_neglogp, new_neglogp,
                             advantage, is_ppo, clip_low=0.1, clip_high=0.2)
```

该函数保留原操作顺序、dtype、`torch.maximum`相等分支行为及clamp边界导数，候选actor区间成为 `[0.9, 1.2]`。原critic裁剪、LF支持门 `[0.9,1.1]`、AMP、全模型梯度裁剪、优势归一化、RNN、GA和学习率调度保持原值。放宽上界只让一部分正优势行继续贡献actor叶导数；这不是完整参数梯度贡献或性能增益的证明。

## 已完成的机制证据

95影子运行使用原损失，不应用候选；固定一轮327680物理帧。8次actor尝试中4次实际Adam更新，终端参数与优化器状态有限。655360次on-policy行访问中有3568次叶系数被候选恢复，分布在3次尝试，满足事前“至少两次尝试存在恢复”的进入筛查条件。off-policy加权叶系数改变数和on-policy丢失数均为0。51次CUDA autograd调用覆盖原生数值及混合dtype边界；这些是数学资格测试，不是训练seed。

每个新增训练入口还需通过E51候选、fresh对照、fresh候选各一轮的真实资格检查；评估入口需通过一次独立seed14999的E51全947对象评估。资格结果及实际退出、W&B回读收据保留在隔离执行目录。

## 固定实验

权威事前规则是 `DAPO_TRANSFER_PROTOCOL_20260914.md`。E51祖先训练seed43，学习筛查重新构建模拟器使用seed35001；这不恢复历史物理轨迹。筛查两臂从同一E51完整checkpoint各训练306整轮，即每臂100270080新帧，候选先运行。只保存固定终点；随后每臂以15001和15002各评估一次master63、全部947对象的首回合、随机动作、容差0.01、最多30000步。

进入完整fresh阶段要求：无行政删失；两个eval seed的配对均值差均为正；两次平均至少+0.5 success；每次10000次CUDA配对对象bootstrap的95%区间下界均大于0。对象bootstrap仅是给定checkpoint/eval seed下的对象组成敏感性，不能冒充训练seed置信区间。负结果保留并结束本候选队列，不扩大预算或改门槛。

通过后自动完成fresh训练seed35011、35012、35013各两臂，每臂15259整轮=5000069120物理帧，六次共30000414720帧。两臂运行顺序跨seed交替；每个终点以16001和16002全947评估。先在每个训练seed内平均两个eval结果，再报告三个配对训练seed差值与小样本t区间，明确正态性假设及n=3局限。

## 资源、归档与停止规则

实际执行使用detached worktree `/home/ubuntu22/EPO-experiments/dapo-20260914`，保留本机原Python3.8/Torch1.8.1/CUDA11.1环境。单GPU串行，训练/评估各自持有GPU锁；调度器不占GPU。训练、模型、bootstrap均在CUDA，CPU只做控制、文本、文件及元数据处理。

每个阶段有独立W&B身份，开跑前等待计划artifact提交，结束后等待原始结果提交；调度器取得真实子进程退出码后，再用fresh API核对远端run与完整artifact清单/MD5/大小/本地SHA。资源监控记录采样显存、功率、温度、RAM、磁盘和实际进程身份，不把离散采样峰值当作连续最大值。

技术失败停止本候选队列并保留原始日志，不自动重跑；科学筛查失败结束完整fresh分配。GPU锁和启动资源门不放宽。预计完整六次5B阶段为数天量级，实际时间以稳定阶段帧速和资源日志为准。
