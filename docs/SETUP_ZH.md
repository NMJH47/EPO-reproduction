# EPO 本机环境与 GPU 配置

## 当前已完成状态

| 项目 | 当前状态 |
| --- | --- |
| 仓库 | `/home/ubuntu22/EPO`，`main@17c48ef`，与远端一致 |
| Conda | 环境 `epo`，目录 `/home/ubuntu22/anaconda3/envs/epo` |
| Python/PyTorch | Python 3.8.18，PyTorch 1.8.1 + CUDA 11.1 runtime |
| Isaac Gym | Preview 4，解压到 `/home/ubuntu22/isaacgym` 并 editable install |
| rl_games/EPO | 两者均已 editable install |
| 用户态依赖 | `pip check` 通过，28 个任务模块全部可导入 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU，16,376 MiB，PyTorch/PhysX 均可用 |
| 内核驱动 | NVIDIA 550.163.01 已加载；MOK 已登记，Secure Boot 保持启用 |

当前 `torch.cuda.is_available()` 为 `True`，设备数为 1；实测 CUDA 张量、GPU PhysX、
GPU pipeline、EPO rollout、反向传播、参数更新和 checkpoint 保存均成功。

## 日常使用

```bash
cd /home/ubuntu22/EPO
conda activate epo
```

Isaac Gym 要求先导入自身，再导入 PyTorch：

```python
import isaacgym
import torch
```

反过来导入会得到 `PyTorch was imported before isaacgym modules`，这是官方绑定的
加载约束，不是安装失败。

## Secure Boot 状态与恢复方式

本机已完成 MOK 登记，无需重复导入。可随时核验：

```bash
mokutil --test-key /var/lib/shim-signed/mok/MOK.der
modinfo -F signer nvidia
nvidia-smi
```

当前预期分别包含 `already enrolled`、
`ubuntu22 Secure Boot Module Signature key` 和驱动版本 `550.163.01`。

仅当以后重装驱动、重新生成签名密钥且模块再次被 Secure Boot 拒绝时，才需要重新执行：

```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
```

随后重启，在蓝色 MOK Manager 页面选择 `Enroll MOK`、`Continue`、`Yes`，输入导入时
设置的一次性密码。登记 MOK 比关闭 BIOS/UEFI Secure Boot 更少改变系统安全策略。

## 驱动恢复后的验证

```bash
conda run -n epo python -c \
  "import isaacgym; import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

预期最后一行是 `True`，设备名为 `NVIDIA GeForce RTX 4090 Laptop GPU`。

再执行：

```bash
conda run -n epo python -m pip check
conda run -n epo python -m isaacgymenvs.train --help
```

仓库还提供严格拒绝 CPU fallback 的全 GPU EPO 冒烟测试：

```bash
cd /home/ubuntu22/EPO
conda activate epo
./scripts/smoke_test_epo_gpu.sh
```

如需指定另一张卡，使用 `EPO_GPU_ID=1 ./scripts/smoke_test_epo_gpu.sh`。

## RTX 4090 与旧 CUDA 的兼容处理

Isaac Gym Preview 4 的基线 PyTorch 1.8.1 使用 CUDA 11.1。CUDA 11.1 的 NVRTC 无法为
RTX 4090 的 compute capability 8.9 编译 TorchScript 融合内核，原始入口会报：

```text
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

`isaacgymenvs/train.py` 现在只在 GPU capability `>= 8.9` 且 PyTorch CUDA runtime
`< 11.8` 时关闭 TorchScript 的 GPU kernel fusion。它不会切换到 CPU：普通 CUDA
算子、策略网络、PhysX 和仿真 pipeline 仍全部运行在 `cuda:0`。启动日志会显示：

```text
Disabled TorchScript GPU fusion because CUDA 11.1 cannot compile for compute capability 8.9
+++ Using GPU PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
current training device: cuda:0
```

## 从零重建 Conda 环境

仓库根目录的 `env.yaml` 已改为经过本机验证的显式版本，而不是无版本约束的完整
机器导出：

```bash
cd /home/ubuntu22/EPO
conda env create -f env.yaml
conda activate epo

pip install -e /home/ubuntu22/isaacgym/python
pip install -e ./rl_games
pip install -e .

conda env config vars set \
  CUDA_HOME=/usr/local/cuda-11.3 \
  LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/epo/lib:/usr/local/cuda-11.3/lib64 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONNOUSERSITE=1

conda deactivate
conda activate epo
```

选择 Python 3.8 是因为仓库 README 明确以 3.8 为目标，Isaac Gym Preview 4 同时只
提供 Python 3.6/3.7/3.8 二进制绑定。PyTorch 1.8.1、torchvision 0.9.1 和
CUDA runtime 11.1 与 NVIDIA 官方 Preview 4 环境保持一致。系统安装的 CUDA 11.3
只用于扩展编译，运行时仍使用 Conda 的 11.1 库。

## 已验证内容与可选依赖

已验证：

- `isaacgym.gymapi` 导入成功。
- `gymtorch` C++/CUDA 扩展编译和加载成功。
- `isaacgymenvs.tasks` 全量导入成功，共注册 28 个任务。
- Hydra 训练入口和 PBT launcher 的 `--help` 均成功。
- `pip check` 无 broken requirements。
- RTX 4090 上完成 AllegroKuka Regrasping 的 EPO 两轮全 GPU 训练，退出码为 0。
- 短测达到约 4,027 total FPS，并生成 23 MiB checkpoint。

导入任务时会看到 Autodesk `fbx` 模块缺失警告。该闭源 SDK 只用于 AMP 的 FBX
动作转换工具；EPO manipulation/Parkour 训练及仓库内已转换的 `.npy` motion 不依赖
它，因此没有把它作为必需依赖安装。

## 复现实验的资源注意事项

论文默认 EPO-64 使用 24,576 个环境，并报告约 24 GB GPU 显存。驱动恢复后应先用
较小的 `num_envs` 和较少 gene blocks 做冒烟测试，再逐步扩大；否则单张 Laptop GPU
很可能直接 OOM。完整论文结果还需要数十亿到百亿环境步和 5 个随机种子，环境安装
成功不等同于已完成论文数值复现。
