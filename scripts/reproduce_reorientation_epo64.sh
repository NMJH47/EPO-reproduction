#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"

CONDA_ENV="${EPO_CONDA_ENV:-epo}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
NUM_ENVS="${EPO_NUM_ENVS:-20480}"
NUM_GENES="${EPO_NUM_GENES:-64}"
SEED="${EPO_SEED:-42}"
MAX_FRAMES="${EPO_MAX_FRAMES:-5000000000}"
EXPERIMENT="${EPO_EXPERIMENT:-00_repro_epo64_seed${SEED}_N${NUM_ENVS}}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-leo-ucbm-university-of-california-berkeley}"
WANDB_PROJECT_VALUE="${WANDB_PROJECT:-EPO-reproduction}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-reorientation_epo64_N${NUM_ENVS}}"
LOG_DIR="${EPO_LOG_DIR:-${ROOT_DIR}/logs/reproduction}"
LOG_FILE="${EPO_LOG_FILE:-${LOG_DIR}/${EXPERIMENT}.log}"
PID_FILE="${EPO_PID_FILE:-/tmp/${EXPERIMENT}.pid}"
RUN_DIR="${ROOT_DIR}/runs/${EXPERIMENT}"
WANDB_METADATA_FILE="${RUN_DIR}/wandb_run.json"
WANDB_RESUME_VALUE="${WANDB_RESUME:-auto}"
WANDB_RUN_ID_VALUE="${WANDB_RUN_ID:-}"

wandb_run_url() {
    if [[ ! -f "${WANDB_METADATA_FILE}" ]]; then
        echo "pending:${WANDB_METADATA_FILE}"
        return 0
    fi

    python3 - "${WANDB_METADATA_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as metadata_file:
    metadata = json.load(metadata_file)
print(metadata.get("url") or f"run_id:{metadata.get('id', 'unknown')}")
PY
}

is_running() {
    [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

show_log_progress() {
    [[ -f "${LOG_FILE}" ]] || return 0

    local start_line
    start_line="$(grep -n '^started_at=' "${LOG_FILE}" | tail -n 1 | cut -d: -f1)"
    start_line="${start_line:-1}"

    tail -n "+${start_line}" "${LOG_FILE}" | grep 'fps step:' | tail -n 1 || true
    tail -n "+${start_line}" "${LOG_FILE}" | \
        grep -E 'CUDA out of memory|Error executing job|MAX FRAMES NUM|MAX EPOCHS NUM' | tail -n 5 || true
}

show_status() {
    if is_running; then
        local pid
        pid="$(cat "${PID_FILE}")"
        echo "RUNNING pid=${pid}"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
        echo "wandb=$(wandb_run_url)"
        echo "log=${LOG_FILE}"
        show_log_progress
        return 0
    fi

    echo "STOPPED"
    echo "wandb=$(wandb_run_url)"
    echo "log=${LOG_FILE}"
    show_log_progress
    return 1
}

stop_run() {
    if ! is_running; then
        rm -f "${PID_FILE}"
        echo "Experiment is not running."
        return 0
    fi

    local pid
    pid="$(cat "${PID_FILE}")"
    echo "Stopping process group ${pid}..."
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true

    for _ in $(seq 1 30); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${PID_FILE}"
            echo "Stopped."
            return 0
        fi
        sleep 1
    done

    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 15); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${PID_FILE}"
            echo "Stopped after SIGTERM."
            return 0
        fi
        sleep 1
    done

    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "Stopped forcefully."
}

case "${ACTION}" in
    status)
        show_status
        exit $?
        ;;
    stop)
        stop_run
        exit 0
        ;;
    start|foreground|sync)
        ;;
    *)
        echo "Usage: $0 [start|status|stop|foreground|sync]" >&2
        exit 2
        ;;
esac

if [[ -z "${CONDA_EXE}" ]]; then
    echo "conda was not found in PATH." >&2
    exit 1
fi

if ! [[ "${NUM_ENVS}" =~ ^[0-9]+$ && "${NUM_GENES}" =~ ^[0-9]+$ && "${NUM_GENES}" -gt 0 ]]; then
    echo "EPO_NUM_ENVS and EPO_NUM_GENES must be positive integers." >&2
    exit 1
fi

if (( NUM_ENVS % NUM_GENES != 0 )); then
    echo "EPO_NUM_ENVS (${NUM_ENVS}) must be divisible by EPO_NUM_GENES (${NUM_GENES})." >&2
    exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU/driver is unavailable; CPU fallback is forbidden." >&2
    exit 1
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

if [[ "${ACTION}" == "sync" ]]; then
    if [[ ! -d "${RUN_DIR}/wandb" ]]; then
        echo "No local W&B data found under ${RUN_DIR}/wandb." >&2
        exit 1
    fi
    while IFS= read -r -d '' run_path; do
        "${CONDA_EXE}" run -n "${CONDA_ENV}" wandb sync "${run_path}"
    done < <(find "${RUN_DIR}/wandb" -mindepth 1 -maxdepth 1 -type d -name 'run-*' -print0)
    exit 0
fi

if [[ -z "${WANDB_ENTITY_VALUE}" ]]; then
    echo "WANDB_ENTITY must be set for online synchronization." >&2
    exit 1
fi

if is_running; then
    echo "Experiment is already running with pid $(cat "${PID_FILE}")." >&2
    exit 1
fi

rm -f "${PID_FILE}"
mkdir -p "${LOG_DIR}"

BLOCK_SIZE=$((NUM_ENVS / NUM_GENES))
MINIBATCH_SIZE=$((NUM_ENVS * 4))
CHECKPOINT="${EPO_CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" && -f "${ROOT_DIR}/runs/${EXPERIMENT}/last/model.pth" ]]; then
    CHECKPOINT="${ROOT_DIR}/runs/${EXPERIMENT}/last/model.pth"
    echo "Auto-resuming from ${CHECKPOINT}"
fi

if [[ -n "${CHECKPOINT}" && -z "${WANDB_RUN_ID_VALUE}" && -f "${WANDB_METADATA_FILE}" ]]; then
    WANDB_RUN_ID_VALUE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("id", ""))' "${WANDB_METADATA_FILE}")"
fi
if [[ -n "${WANDB_RUN_ID_VALUE}" ]]; then
    WANDB_RESUME_VALUE=allow
fi

TRAIN_COMMAND=(
    python -m isaacgymenvs.train
    task=AllegroKukaLSTM
    task/env=reorientation
    ++task.env.useSparseReward=False
    "task.env.numEnvs=${NUM_ENVS}"
    "train.params.config.minibatch_size=${MINIBATCH_SIZE}"
    "seed=${SEED}"
    pipeline=gpu
    sim_device=cuda:0
    rl_device=cuda:0
    headless=True
    force_render=False
    graphics_device_id=0
    "train.params.config.max_frames=${MAX_FRAMES}"
    train.params.config.max_epochs=1000000
    train.params.config.save_frequency=3000
    "experiment=${EXPERIMENT}"
    train.params.config.good_reset_boundary=0
    task.env.goodResetBoundary=0
    train.params.config.use_others_experience=lf
    train.params.config.off_policy_ratio=1.0
    train.params.config.expl_type=mixed_expl_learn_param
    train.params.config.expl_reward_type=entropy
    "train.params.config.expl_coef_block_size=${BLOCK_SIZE}"
    train.params.config.expl_reward_coef_scale=0.005
    ++train.params.config.expl_reward_coef=0.0
    train.params.network.space.continuous.fixed_sigma=coef_cond
    train.params.config.epo=True
    wandb_activate=True
    "wandb_entity=${WANDB_ENTITY_VALUE}"
    "wandb_project=${WANDB_PROJECT_VALUE}"
    "wandb_group=${WANDB_GROUP_VALUE}"
    wandb_mode=online
    "wandb_resume=${WANDB_RESUME_VALUE}"
    "wandb_tags=[reproduction,epo64,reorientation,gpu_only,scaled_N${NUM_ENVS}]"
    "++wandb_notes=gpu_only_epo64_reorientation_original_N24576_local_N${NUM_ENVS}"
)

if [[ -n "${CHECKPOINT}" ]]; then
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
        exit 1
    fi
    TRAIN_COMMAND+=("checkpoint=${CHECKPOINT}")
fi
if [[ -n "${WANDB_RUN_ID_VALUE}" ]]; then
    TRAIN_COMMAND+=("wandb_run_id=${WANDB_RUN_ID_VALUE}")
fi

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'num_envs=%s num_genes=%s block_size=%s minibatch_size=%s seed=%s max_frames=%s\n' \
        "${NUM_ENVS}" "${NUM_GENES}" "${BLOCK_SIZE}" "${MINIBATCH_SIZE}" "${SEED}" "${MAX_FRAMES}"
    printf 'command='
    printf '%q ' "${CONDA_EXE}" run -n "${CONDA_ENV}" --no-capture-output "${TRAIN_COMMAND[@]}"
    printf '\n'
} >>"${LOG_FILE}"

cd "${ROOT_DIR}"

if [[ "${ACTION}" == "foreground" ]]; then
    exec "${CONDA_EXE}" run -n "${CONDA_ENV}" --no-capture-output "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

nohup setsid "${CONDA_EXE}" run -n "${CONDA_ENV}" --no-capture-output "${TRAIN_COMMAND[@]}" \
    >>"${LOG_FILE}" 2>&1 </dev/null &
PID=$!
echo "${PID}" >"${PID_FILE}"

sleep 3
if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "Experiment failed during startup. Last log lines:" >&2
    tail -n 40 "${LOG_FILE}" >&2
    exit 1
fi

echo "Started ${EXPERIMENT} with pid ${PID}."
echo "W&B: $(wandb_run_url)"
echo "Log: ${LOG_FILE}"
echo "Status: $0 status"
