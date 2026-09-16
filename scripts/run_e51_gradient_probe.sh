#!/usr/bin/env bash
# One epoch of the preserved E51 trajectory, with a rolled-back paired Adam probe.
# This is a mechanism diagnostic, not a performance or speed benchmark.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
epo_env="${EPO_ENV_DIR:-/home/ubuntu22/anaconda3/envs/epo}"
export PATH="${epo_env}/bin:${PATH}"
export LD_LIBRARY_PATH="${epo_env}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${repo_dir}/rl_games:${repo_dir}:${PYTHONPATH:-}"
export EPO_GRAD_PROBE_OUTPUT="${EPO_GRAD_PROBE_OUTPUT:-${repo_dir}/artifacts/research/20260913-audit/actual-gradient-probe.json}"
export EPO_GRAD_PROBE_STEPS="${EPO_GRAD_PROBE_STEPS:-1}"
experiment="${EPO_PROBE_EXPERIMENT:-60_probe_E51_same_gradient_20260913}"
source_run="${repo_dir}/runs/51_full_epo_E8_legacy_equivalent_identity_seed43_N20480"
checkpoint="${source_run}/nn/last_51_full_epo_E8_legacy_equivalent_identity_seed43_N20480_frame_5000069120_rew__37925.93_.pth"

if [[ -e "${repo_dir}/runs/${experiment}" || -e "${EPO_GRAD_PROBE_OUTPUT}" ]]; then
    echo "Choose fresh EPO_PROBE_EXPERIMENT and EPO_GRAD_PROBE_OUTPUT; refusing overwrite." >&2
    exit 1
fi
cd "${repo_dir}"
exec "${epo_env}/bin/python" scripts/probe_training_gradients.py \
    --config-path "${source_run}" --config-name config \
    "hydra.run.dir=${repo_dir}" hydra.output_subdir=null hydra.job.chdir=False \
    "checkpoint=${checkpoint}" checkpoint_yaml= \
    "experiment=${experiment}" "train.params.config.name=${experiment}" \
    train.params.config.max_epochs=15260 train.params.config.max_frames=6000000000 \
    train.params.config.save_frequency=1000000 train.params.config.save_best_after=1000000 \
    headless=True force_render=False \
    wandb_activate=True wandb_mode=online wandb_resume=never wandb_run_id= \
    wandb_entity=leo-ucbm-university-of-california-berkeley wandb_project=EPO-reproduction \
    wandb_group=ceiling-audit-20260913 wandb_job_type=mechanism \
    'wandb_tags=[diagnostic,actual_gradient,paired_adam,not_learning_benefit]' \
    wandb_log_code=False wandb_log_diff=True \
    wandb_notes=paired_gradient_probe_only_legacy_trajectory_candidate_step_rolled_back
