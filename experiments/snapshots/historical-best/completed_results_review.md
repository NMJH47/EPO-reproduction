# 2026-09-16：完整长预算复核与下一轮依据

本轮逐项核对了已执行的两条连续训练、原11次评估及其资源和W&B记录，补齐5次缺失评估，完成四个固定节点的配对分析。**延长训练确有明显收益；九格课程在2B终点出现小幅正向信号，但没有足够证据确立稳定算法增益，也不能把50当作已经证明可达的最优期望。**

## 完成状态与恢复

原 controller `beb2f48c07044934` 并未跑完整个三臂队列。frontier/control各6104轮、2,000,158,720新增环境控制帧均成功；Clip-Higher没有启动。第11次评估的仿真、947首episode、上传及finish均成功，monitor在结束后的一次性远端状态/清单断言失败，导致旧controller只计10次。断言没有保存两个操作数，不能再细分是run状态还是artifact枚举暂未可见。

独立 fresh API 核对原27个run、49个artifact及879条manifest记录，MD5/size与本地SHA一致。原monitor现在远端finished并不抹去其实际wait1；原controller依然failed。新恢复 `44d8e248644148b0` 只补5个未运行control评估、导入已验证第11项，并生成4个比较，实际wait0、远端finished，plan136文件和result58文件均逐项核验。两条旧训练没有重跑。

[原运行逐项审计](/home/ubuntu22/EPO/artifacts/research/20260916-review/runtime/REPORT.md)、[恢复方案和闭合结果](/home/ubuntu22/EPO-experiments/frontier-20260914/artifacts/research/20260916-recovery/controller/summary.json)、[恢复W&B](https://wandb.ai/leo-ucbm-university-of-california-berkeley/EPO-reproduction/runs/44d8e248644148b0)。各专项报告保留其恢复前的时间截面；本文件及完整16项汇总给出恢复后的结果。

## 每条训练及所有固定节点

两组都从同一E51、同一continuation seed36001启动。每节点分别对全947对象、两个eval seed17001/17002、master63做首episode冻结评估，均无删失。下表是两个eval seed等权均值，不是多个独立训练重复。

| 新增训练帧 | control | frontier | frontier−control |
|---|---:|---:|---:|
| 100,270,080 | 38.362196 | 38.362196 | 0.000000 |
| 500,039,680 | 40.108765 | 39.973073 | −0.135692 |
| 1,000,079,360 | 40.869060 | 40.834741 | −0.034319 |
| 2,000,158,720 | 41.049102 | 41.504224 | +0.455121 |

100M点仍处在课程warmup内，候选首次在第321轮改变提案。前320轮2560次actor返回的11个记录字段逐项相同，100M全部1894个对象配对值相同，符合预定隔离时点；不能用该节点判课程无效。

2B两个eval seed的差值分别为+0.517423、+0.392819；对应10,000次对象配对bootstrap百分位95%区间为[−0.609319,1.681125]和[−0.790945,1.575502]。两者都跨0。这些区间只描述给定checkpoint与eval seed下的对象组成敏感性，不是跨训练seed置信区间；枚举的947目录均值本身已确定。该单轨迹结果不足以确立稳定优势。

同一连续训练的2B−100M提升：control +2.686906，frontier +3.142027。control的1B→2B仍有小幅上升，frontier末段上升更明显；没有观察到足以判定渐近平台的证据。旧约36分钟的100M负结果确实受检验时间范围限制，但不会因本轮更长就自动变成正结论。

| 训练过程 | frontier | control |
|---|---:|---:|
| 实际Adam返回 / actor尝试 | 48,813 / 48,832 | 48,811 / 48,832 |
| GA merge实际返回 | 94 | 94 |
| 首次GA相对epoch | 404 | 416 |
| 原core FPS | 47,501.42 | 47,843.00 |
| 完整进程小时 | 11.9946 | 11.8879 |
| 采样显存峰MiB / 最低空闲MiB | 13,472 / 2,601 | 13,470 / 2,603 |
| 观察窗口整卡能量Wh | 802.945 | 799.603 |

能量和资源峰来自2秒采样，覆盖setup、训练及归档，不能称进程独占能耗或连续峰值。两组顺序执行，约0.72%的core FPS差不足以单独归因课程。实际Adam返回不等于每个参数都有非零变化；少量skip也不能排除梯度裁剪耦合。

[完整CUDA数值与尾部结果](/home/ubuntu22/EPO/artifacts/research/20260916-review/learning-curves/summary.json)、[学习曲线图](/home/ubuntu22/EPO/artifacts/research/20260916-review/learning-curves/learning_curves.png)、[可编辑SVG](/home/ubuntu22/EPO/artifacts/research/20260916-review/learning-curves/learning_curves.svg)、[全部审计与曲线W&B](https://wandb.ai/leo-ucbm-university-of-california-berkeley/EPO-reproduction/runs/23c5f91e304b496d)。该run实际wait0且fresh API核验finished，plan152文件30,837,868字节及result6文件284,012字节均COMMITTED并匹配，包含各专项审计、CUDA反事实、论文和作者源码材料。

## 距离50的缺口在哪里

`successes_max`是master320个环境上次结束episode缓存的最大值，不是block均值，更不是全947平均。success计数是同episode连续完成目标位置/姿态，通常成功后只换目标，不是每次重新抓取。日志max早到50与block均值约37并不冲突。

2B课程组达到cap50的比例为72.8617%，对照71.0665%。但课程组两个seed仍分别有47/49个S0对象，control为44/51；S<10课程组101/103，control103/93。课程没有显示一致的早期失败减少。课程组S<10的episode贡献总缺口58.71%/62.23%，说明平均值的提升尚未解决低尾。

对任意0≤S≤50的整数计数，不需要iid假设就有：

\[
E[S]=\sum_{k=1}^{50}P(S\ge k),\qquad
50-E[S]=\sum_{k=1}^{50}P(S<k).
\]

因此应同时看初次目标成功、生存曲线、cap与对象分布。训练池中后续goal约99%的成功率包含幸存者和频率加权，不能代入所有对象每一阶段，推出最终平均接近50。50仍是任务计分上限，不能由存在达cap样本证明期望最优为50。

## 逐结构检查

| 结构 | 当前已验证的问题或现象 | 对下一步的约束 |
|---|---|---|
| 目标课程 | 91次q平均TV仅initial0.00890、continuation0.02537；同刻q相对uniform的成功率proxy只降低约0.000119/0.0000924；轨迹非单调变难 | 不能把弱九格干预当成对所有group filtering的检验 |
| Score和条件 | 同历史计数换VADE型Beta分数，initial TV升到0.02343，continuation反降到0.00552；最难bin proxy界也小 | 不启动只换score的新长跑；先采集出生m×gene×bin闭合联合分母，现有边际不能补造联合表 |
| 对象覆盖 | 每gene直接320对象，master固定273–592，整体population覆盖947 | 其余627通过共享主干/follower/LF有信号，不能称完全未见；需把覆盖差异与失败尾部分开验证 |
| GA | 各94次真实merge，master未被替换；fitness来自不同固定对象面板的缓存均值及一代历史max；GA在rollout/LF/prepare后、更新前发生 | merge次数不证明每次gene非零变化，更不证明改善；不同对象面板不能保证parent排名可比 |
| LF/actor/critic | off行改gene而保留旧hidden，原support门；off critic系数0，原GAE和价值坐标路径保留 | 不能把LF副本当独立组；不能删除同success序列后声称无损或critic不受影响 |
| RNN/归一化 | H16/seq16，goal-only不重置hidden；LF后332800行分为3×81920+87040 | 长episode不直接证明H16不足；GAE迹在H16内截断，末端critic bootstrap承接更远回报，RNN hidden可跨rollout；行权重与8次顺序Adam不能简化成一次全batch梯度 |
| AMP/clip | 当前单卡每次backward前清空全model梯度；unscale仅处理Adam-owned参数，clip-all仍纳入本次未unscale的gene梯度；既有M2已证真实更新大小与方向变化 | 不归因为跨actor梯度累积；norm变大不等于效果更好，少量skip不能排除该耦合 |
| LR | 完整12208项调度链与原规则一致，终点LR同为3.42548739e−5；所用KL是原缓存参考 | 不把它称post-step或逐gene KL保证，也不据终点LR相同推断训练轨迹相同 |
| 工程/记录 | 旧队列因后置远端一次性断言中断；已有效仿真不需重跑 | 新监控只做有界只读核验，内容不符失败，保留原wait与失败身份 |

## 过去的负结果仍然有效，但只覆盖其预算

| 既有对照 | 预算 | 两个冻结评估的候选−对照 |
|---|---|---|
| 72/73：optimizer-owned clip | 100M | −1.636748 / +0.010560；GA打印3/0 |
| 76/77：on-cache KL调度 | 100M | −3.656811 / −2.865892 |
| burn-in16 | 100M | −0.701162 / +0.380148 |
| 84/85：去掉actor support门 | 100M | +0.510032 / −1.374868 |
| 原DAPO Clip-Higher pilot | 100M | −0.074974 / −0.916579 |
| 93/94：降低恢复AMP scale | 各1epoch | 只验证Adam4→8，没有完整学习收益结论 |

这些是不同已登记试验，不拼接成一个多seed估计。GA3/0是clip干预之后的实际下游差异，不是人为同时改了GA配置。具体旧原件和范围见[算法审查](/home/ubuntu22/EPO/artifacts/research/20260916-review/algorithm/REPORT.md)与[研究决策账本](/home/ubuntu22/EPO/docs/research/RESEARCH_DECISION_LEDGER_20260913.md)。

## 文献和下一轮

深查了DAPO正式论文与固定作者实现，并核对VADE、VIP、RL-ZVP、DA3PO，以及连续控制课程GoalGAN、ALP-GMM、PLR等一手材料。核心区分是“同条件多轨迹组”“未来任务提案”和“事后删原序列”：它们的优势、分母、critic影响和物理成本不同，不能混称同一种filtering。已保存33份论文/代码材料及各自版本SHA。[完整文献审查](/home/ubuntu22/EPO/artifacts/research/20260916-review/literature/REPORT.md)、[DAPO正式论文](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf)、[VADE](https://arxiv.org/html/2511.18902v1)、[VIP](https://arxiv.org/html/2602.01601v3)。

下一轮分别补足Clip-Higher2B和optimizer-owned clipping2B的长期范围，复用已经验证的同起点同seed连续control，并保留完整四节点、16次新评估、8个描述比较。两条新训练共用稀疏同次梯度探针，不同时改LR、AMP、GA、课程或LF。先通过新hook的CUDA实现合同与三项1epoch真实资格，再启动固定长预算。方案与实时运行身份分列于[固定协议](STRUCTURAL_LONG_PROTOCOL_20260916.md)和[当前状态](ACTIVE_RESEARCH_STATE_20260913.md)。

本轮结论仍限于一个E51祖先、一个continuation seed。无论下一轮哪个因素显示收益，论文层面的算法结论仍需多训练seed、匹配fresh普通PPO/EPO基线和泛化验证；不能凭当前+0.455或max=50宣称达到顶会标准。
