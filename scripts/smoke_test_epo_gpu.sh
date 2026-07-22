#!/usr/bin/env bash

set -euo pipefail

gpu_id="${EPO_GPU_ID:-0}"
experiment="${EPO_SMOKE_EXPERIMENT:-0_gpu_smoke}"
wandb_activate="${EPO_WANDB_ACTIVATE:-False}"
intrinsic_reward_type="${EPO_SMOKE_IR_TYPE:-none}"
max_epochs="${EPO_SMOKE_MAX_EPOCHS:-2}"
checkpoint="${EPO_SMOKE_CHECKPOINT:-}"
wandb_resume="${WANDB_RESUME:-auto}"
wandb_run_id="${WANDB_RUN_ID:-}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

checkpoint_args=()
if [[ -n "${checkpoint}" ]]; then
  checkpoint_args+=("checkpoint=${checkpoint}")
fi

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
  "max_iterations=${max_epochs}" "train.params.config.max_epochs=${max_epochs}" "experiment=${experiment}" \
  train.params.config.save_frequency=100 \
  train.params.config.defer_summaries_sec=0 \
  train.params.config.summaries_interval_sec_min=0 \
  train.params.config.summaries_interval_sec_max=1 \
  train.params.config.minibatch_size=256 \
  train.params.config.mini_epochs=1 \
  train.params.config.good_reset_boundary=0 task.env.goodResetBoundary=0 \
  train.params.config.use_others_experience=lf \
  train.params.config.off_policy_ratio=1.0 \
  train.params.config.expl_type=mixed_expl_learn_param \
  "train.params.config.expl_reward_type=${intrinsic_reward_type}" \
  train.params.config.expl_coef_block_size=8 \
  train.params.config.expl_reward_coef_scale=1.0 \
  train.params.network.space.continuous.fixed_sigma=coef_cond \
  train.params.config.epo=True \
  "wandb_activate=${wandb_activate}" \
  "wandb_entity=${WANDB_ENTITY:-}" \
  "wandb_project=${WANDB_PROJECT:-EPO-reproduction}" \
  "wandb_group=${WANDB_GROUP:-gpu_smoke}" \
  wandb_mode=online \
  "wandb_resume=${wandb_resume}" \
  "wandb_run_id=${wandb_run_id}" \
  wandb_log_code=False \
  wandb_log_diff=False \
  "wandb_tags=[smoke,gpu_only,wandb_reliability]" \
  "${checkpoint_args[@]}"
