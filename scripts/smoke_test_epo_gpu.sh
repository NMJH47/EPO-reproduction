#!/usr/bin/env bash

set -euo pipefail

gpu_id="${EPO_GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

python - <<'PY'
import isaacgym
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing to run the EPO GPU smoke test")

print(f"GPU smoke test device: {torch.cuda.get_device_name(0)}")
PY

python -m isaacgymenvs.train \
  task=AllegroKuka task/env=regrasping \
  ++task.env.useSparseReward=False \
  task.env.numEnvs=64 task.env.episodeLength=32 \
  pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 \
  headless=True force_render=False graphics_device_id=0 \
  max_iterations=2 train.params.config.max_epochs=2 experiment=0_gpu_smoke \
  train.params.config.save_frequency=100 \
  train.params.config.minibatch_size=256 \
  train.params.config.mini_epochs=1 \
  train.params.config.good_reset_boundary=0 task.env.goodResetBoundary=0 \
  train.params.config.use_others_experience=lf \
  train.params.config.off_policy_ratio=1.0 \
  train.params.config.expl_type=mixed_expl_learn_param \
  train.params.config.expl_reward_type=none \
  train.params.config.expl_coef_block_size=8 \
  train.params.config.expl_reward_coef_scale=1.0 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  train.params.config.epo=True \
  wandb_activate=False
