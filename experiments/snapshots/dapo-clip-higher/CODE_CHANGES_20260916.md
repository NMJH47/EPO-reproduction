# 本轮代码前后变化与复用位置

新文件的完整可应用补丁：[new_experiment_files.patch](/home/ubuntu22/EPO/artifacts/research/20260916-review/new_experiment_files.patch)。从仓库根目录可用 `git apply --check` 检查；当前工作区已包含这些文件，不应重复应用。正式运行使用隔离目录中逐SHA绑定的同字节副本，源身份见[manifest](/home/ubuntu22/EPO/artifacts/research/20260916-review/new_experiment_source_manifest.json)。既有训练源与结果不回写。

## 梯度裁剪参数集合

原生调用位置：[a2c_common.py](/home/ubuntu22/EPO/rl_games/rl_games/common/a2c_common.py:374)。原行为：

```python
self.scaler.unscale_(self.optimizer)
nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
self.scaler.step(self.optimizer)
self.scaler.update()
```

实验使用的入口为[run_structural_training.py](/home/ubuntu22/EPO/scripts/research/run_structural_training.py)，在同次调用边界安装[clip_scope_hooks.py](/home/ubuntu22/EPO/scripts/research/clip_scope_hooks.py)。实际变化是：

```python
supplied = list(parameters)
owned, excluded = partition_parameters(self.agent, supplied)
selected = owned if self.policy == 'optimizer_owned' else supplied
result = self.native_clip(selected, max_norm, norm_type)
```

`partition_parameters` 按当前optimizer中参数的对象身份筛选，保持model原顺序。`owned_clip`选择该子集；`control`和`clip_higher`仍把完整原集合交给原Torch函数。原unscale、step、update各调用一次。gene是否requires_grad、是否在optimizer、原梯度清零和Adam历史均由原训练路径决定，不在这个实验中顺带修改。

稀疏norm探针在同次unscale后读owned/excluded/full范数与两种系数，不增加forward/backward。完整日志和关闭状态写入结果artifact。CUDA资格逐位核验control非干扰以及candidate与手工子集相等，实际通过范围见[fixture summary](/home/ubuntu22/EPO/artifacts/research/20260916-review/next/cuda-fixture-v2/summary.json)。首版资格的Torch1.8跨位宽view不兼容失败及修订均保留；它不属于模型训练失败。

## DAPO Clip-Higher

继续复用原已审 `dapo_actor.actor_loss_asymmetric`。`clip_higher`只将actor正向ratio上界从1.1改为1.2，下界0.9、原critic clip和LF support门保留。新入口增加长预算和norm观察，不同时启用owned clipping。

## 监控收尾

原[_research_support/monitor_job.py](/home/ubuntu22/EPO/_research_support/monitor_job.py:60)在一次finish后立即检查远端：

```python
remote = wandb.Api(timeout=30).run(run_path)
matches = [a for a in remote.logged_artifacts()
           if a.name == committed.name and a.state == 'COMMITTED']
assert remote.state == expected_state and len(matches) == 1
```

新[monitor_job_consistent.py](/home/ubuntu22/EPO/_research_support/monitor_job_consistent.py)改为：

```python
run.finish(exit_code=0 if status == 'completed' else 1)
return verify_remote_consistent(wandb, run, committed.name, output, files, status)
```

函数在120秒尝试准入预算内，每2秒做只读查询，记录每次状态、候选artifact与错误；当前请求有独立API超时，可能跨过准入截止。COMMITTED内容的集合、MD5或size不符立即失败，不因读取重试而再次上传、finish或重训。重试journal放在原四文件payload旁边，保持既有manifest合同。7个CPU故障模拟及5次真实恢复评估的monitor均通过。

## 固定队列与统计

[run_structural_study.py](/home/ubuntu22/EPO/scripts/research/run_structural_study.py)先验证实际CUDA资格、所有角色源码、已恢复control八项的确切来源和checkpoint，再串行运行三项1epoch资格、两条2B训练、16评估及8比较。原归档类型、实际subprocess.wait、source/config/checkpoint SHA及单GPU锁继续使用。

[analyze_structural_pair.py](/home/ubuntu22/EPO/scripts/research/analyze_structural_pair.py)仅扩展处理因素身份，数值函数、完整947对象配对、bootstrap和删失规则与原long-budget分析器AST一致。保留的legacy gate只作描述，队列不根据它分支。
