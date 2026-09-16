# 2026-09-16：补齐长预算对照与独立梯度裁剪因素

本轮回答两个此前没有完整长期结果的问题：DAPO 的 Clip-Higher 在完整 2B 预算下是否有效；已确认会改变真实 Adam 更新的 optimizer-owned clipping 在同一预算下是否有效。两项分别运行，不叠加、不以早期结果选择停止。此前 100M 的负向或不稳定结果保持原样；本实验不使旧试验变成“通过”。

## 为什么选择这两个问题

原 20260914 队列已完成 frontier/control 各 2,000,158,720 新控制帧，各有 94 次真实 GA merge 返回。其 controller 在有效评估之后的 monitor 远端状态核验处失败，原 Clip-Higher 尚未启动。独立恢复计划只补缺失的 5 次 control 评估和 4 个固定比较，不重训两条有效轨迹、不重跑已经验证的第 11 次评估。原 failed/wait1 记录与新恢复身份并存。

当前课程的 91 次 teacher 更新产生的平均 total variation 仅 initial 0.00890、continuation 0.02537。对同一历史统计作 CUDA 反事实，把 mixed-G16 分数换成 Beta 积分 E[p(1−p)^2]，continuation 的 TV 反降到 0.00552。固定九格和混合质量时，最难单格的 proxy 界也很小。这不能否定 group filtering；它削弱“仅换评分函数即可解决均值缺口”的依据。因此本轮不再追加同九格 score 长跑。

DAPO 的 Dynamic Sampling 使用同 prompt 多条 response 的组结果，并补充生成；Clip-Higher 是另一独立因素。本地连续 seq16、64gene 和 LF 复制不是条件 iid 组，且本地使用 dense reward/GAE/critic。保留未来任务采样研究，但不以“同 success”删除原训练序列。后续条件研究应首先收集出生阶段×gene×bin的完整联合统计，区分 pending、GA失效与自然失败；不能由现有两个边际恢复联合分布。[DAPO 正式论文](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf)、[VADE](https://arxiv.org/html/2511.18902v1)、[完整本轮文献审查](/home/ubuntu22/EPO/artifacts/research/20260916-review/literature/REPORT.md)。

E51 的 gene 参数退出 Adam 后仍可有梯度。原 unscale 只作用于 Adam 管理的参数，而 clip 使用完整 model 参数。已有 M2 在同梯度、同 Adam 历史上测到真实参数更新大小及方向改变；这不是仅看 gradient norm 的推断。72/73 的 owned-clipping 对照却只跑 100,270,080 新帧，两评估差 −1.636748/+0.010560，GA 打印次数 3/0。机制真实不代表收益有保证，本轮只扩大预先固定的检验时间范围。

## 固定配置与比较

| 项目 | 本轮固定值 |
|---|---|
| 起点 | 原 E51，SHA256 de220625c407b463d41f9242cd652a40f5e48c253b4b9225b7817e452cf0f6be |
| continuation seed | 36001；共同祖先 seed43，不声称 fresh 独立训练 |
| 已有主参考 | control run 4742038629e04897 的连续 2B 轨迹及全部 8 项验证评估 |
| 新训练顺序 | clip_higher → owned_clip，单 GPU 串行 |
| 每臂预算 | 6104 epoch，97,664 控制步，2,000,158,720 新环境控制帧 |
| 固定里程碑 | relative epoch 306/1526/3052/6104，即约 100M/500M/1B/2B |
| Clip-Higher | 仅 actor ratio 上界 1.1→1.2；下界、critic clip 和原 LF support 门不变 |
| Owned clipping | 仅改变全局梯度 norm 的参数集合为当前 Adam 管理的参数；保持 model 顺序与原 Torch clip 实现 |
| 其他学习结构 | N20480、64×320、H16/seq16、mini-epochs2、原 GAE/LF、GA、Adam moments、AMP/scale、LR、RMS、奖励与原目标分布 |
| 课程 | 保留所有臂共享的被动统计；不应用非均匀目标 proposal |
| 评估 | 每臂完成连续训练后依次冻结四里程碑；每项使用 17001/17002、master63、全947对象、首episode、tol0.01、随机动作、max_steps30000 |
| 统计 | 每里程碑 candidate−原control；完整对象配对与10,000次既有CUDA bootstrap，数值函数不改 |
| 队列 | 3 项独立1epoch技术资格、2条完整训练、16次评估、8项描述性比较 |

原 control 的全部端点由新恢复输出、原 training run、checkpoint SHA 和实际评估文件共同绑定。新评估继续使用原隔离目录内冻结的 evaluator 与数值项目源码，确保 paired protocol/source 一致。新增执行入口与 helper 放在 `artifacts` 的独立目录，并显式绑定、复制、上传；不改写原 535 文件源码快照。训练配置仍显示原 model clipping，`owned_clip` 的运行时覆盖由计划、入口、helper 和 W&B `study/gradient_clip_policy` 明确记录，复现必须包含这些执行源。

旧 control 没有新增 norm 探针；复用它依赖新被动 hook 在 control 路径上的实现不干扰验证，而非声称新旧进程是逐位可重复的物理轨迹。原先已有相同 native 数学与目标分布；新增稀疏观测有时间开销。吞吐对比因此报告分阶段实测，不能从顺序运行的墙时差独立归因算法。

## 资格与探针

先执行小 CUDA 合同 fixture：同合成梯度/Adam/scaler/RNG状态下，control 安装探针前后结果精确一致；owned 分支等于手工选定参数子集的原 clip；移除 optimizer 参数后集合随之更新；每次 unscale 恰好一次；非有限 norm 作为带标签数据记录。它验证新实现，不重复宣称发现 M2 已知机制，不以更新增大作为收益门。

随后 control/clip_higher/owned_clip 各 1 epoch，seed36191/36192/36193，里程碑仅[1]。检查真实 Adam 返回、完整配置、有限性、关闭的日志、实际 wait0、W&B artifact 字节以及 owned finish。资格不是效果筛选，不把1epoch结果作为最终性能。

两条正式新训练共用稀疏探针：前8次actor尝试；每64个相对epoch的最后batch；每次GA返回后的首batch。仅在同一次 native unscale 之后、native clip 之前记录 owned norm、excluded remaining-scaled norm、全参数 norm、两种 clip 系数、实际策略、scale、对象身份计数与采样原因。没有新增 forward/backward，没有额外清梯度或改 Adam state，不按探针值选择样本/学习率。`norms.jsonl` 关闭后进入最终不可变结果；失败也保留首因和实际进程结果。

## 推断边界与成本

主终点为2B，其他三个节点为事前固定学习曲线。所有947对象保留，删失按原上下界处理。附加 S0、S<10、cap比例、生存曲线及训练直接覆盖320/其余627分组均为描述性诊断。训练日志缓存均值、完成episode事件均值与冻结947均值不同，不拼接成一个指标；success50是计数上限，尚未证明最优期望可达50。

本轮仅一个共同祖先和一个 continuation seed。两个 eval seed、947对象和64gene都不是独立训练重复；object bootstrap不是跨训练seed置信区间。任何收益仍需要新的多seed、匹配fresh普通PPO/EPO基线及泛化实验，不能直接称顶会级改进或新颖方法。已有负结果和全部固定节点同时报告，不挑有利checkpoint。

按本轮已完成约47.5–47.8k core FPS，每条2B约12小时，两条约24小时，加资格、16次评估和上传，预计约25–28小时。固定新增训练帧4,000,317,440，资格983,040；评估按947×真实global steps另计，理论16项上限454,560,000。单4090 Laptop16GB串行，启动要求至少16GiB磁盘空闲，持续使用原资源监控与保留策略；不删旧实验来美化结果。

监控使用新的有界远端一致性读取：一次上传、一次finish后，仅重试读取状态和manifest；内容不符立即失败。它不自动重跑训练、评估或重复finish。训练/评估自身失败停止队列，保留原件，不把原失败改成成功；没有基于收益的提前停止门。
