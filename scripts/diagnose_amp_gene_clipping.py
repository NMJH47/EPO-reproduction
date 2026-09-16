"""CUDA mechanism experiment: AMP clipping of genes outside Adam.

This is a controlled optimizer experiment, not an RL training result. Run with
the project's CUDA environment. W&B online logging is mandatory. The core
training code is neither imported nor modified.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch
import wandb


ROOT = Path(__file__).resolve().parents[1]
AUDITED_SOURCES = (
    "scripts/diagnose_amp_gene_clipping.py",
    "isaacgymenvs/utils/rlgames_utils.py",
    "rl_games/rl_games/common/a2c_common.py",
    "rl_games/rl_games/algos_torch/a2c_continuous.py",
    "rl_games/rl_games/algos_torch/network_builder.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 128, 65536, 16777216])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "EPO-reproduction"))
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--group", default="mechanism_amp_gene_clipping_20260913")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def command_output(command):
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def norm(parameters):
    gradients = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return torch.linalg.norm(torch.cat(gradients)).item()


class ToyNetwork(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([1.0, -0.5, 0.25], device=device))
        self.gene = torch.nn.Parameter(torch.tensor([2.0, -1.0], device=device))

    def loss(self):
        # Fixed deterministic FP32 inputs isolate GradScaler's ownership rules.
        # No half-precision forward overflow or changing batch is involved.
        return self.shared.square().sum() + self.gene.square().sum()


def new_optimizer(model, args):
    return torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_eps)


def state_after_warmup(args, warmup_steps):
    model = ToyNetwork(args.device)
    optimizer = new_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.0, growth_interval=1000000)
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        scaler.scale(model.loss()).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def run_arm(args, initial_state, optimizer_state, scale, owned_gene, clip_owned_only, history):
    model = ToyNetwork(args.device)
    model.load_state_dict(initial_state)
    optimizer = new_optimizer(model, args)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    if not owned_gene:
        for group in optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"] if parameter is not model.gene]
        optimizer.state.pop(model.gene, None)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    clipped = owned if clip_owned_only else list(model.parameters())
    scaler = torch.cuda.amp.GradScaler(init_scale=scale, growth_interval=1000000)
    arm = ("all_owned" if owned_gene else "frozen_gene") + ("_clip_owned" if clip_owned_only else "_clip_all")
    rows = []
    snapshots = []
    for step in range(args.steps):
        # model.zero_grad clears the unowned gene as in A2CAgent.calc_gradients.
        model.zero_grad()
        shared_before = model.shared.detach().clone()
        gene_before = model.gene.detach().clone()
        scaler.scale(model.loss()).backward()
        scaler.unscale_(optimizer)
        shared_norm = norm([model.shared])
        gene_norm = norm([model.gene])
        expected_shared = (2 * shared_before).norm().item()
        gene_multiplier = 1.0 if owned_gene else scale
        expected_gene = (2 * gene_before).norm().item() * gene_multiplier
        assert math.isclose(shared_norm, expected_shared, rel_tol=2e-6, abs_tol=1e-7)
        assert math.isclose(gene_norm, expected_gene, rel_tol=2e-6, abs_tol=1e-7)
        expected_clip_norm = math.hypot(shared_norm, gene_norm) if (owned_gene or not clip_owned_only) else shared_norm
        actual_clip_norm = torch.nn.utils.clip_grad_norm_(clipped, args.max_norm).item()
        assert math.isclose(actual_clip_norm, expected_clip_norm, rel_tol=2e-6, abs_tol=1e-7)
        clip_coefficient = min(1.0, args.max_norm / (actual_clip_norm + 1e-6))
        clipped_shared_norm = norm([model.shared])
        assert math.isclose(clipped_shared_norm, shared_norm * clip_coefficient, rel_tol=3e-6, abs_tol=1e-10)
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() == scale, "Unexpected AMP scale change or overflow"
        shared_delta = (model.shared.detach() - shared_before).norm().item()
        gene_delta = (model.gene.detach() - gene_before).norm().item()
        assert torch.isfinite(model.shared).all()
        if not owned_gene:
            assert torch.equal(model.gene.detach(), gene_before), "Frozen gene unexpectedly changed"
        row = {
            "history": history,
            "arm": arm,
            "step_in_arm": step,
            "amp_scale": scale,
            "gene_in_optimizer": int(owned_gene),
            "clip_owned_only": int(clip_owned_only),
            "shared_grad_norm_after_unscale": shared_norm,
            "gene_grad_norm_after_unscale": gene_norm,
            "clip_pre_norm": actual_clip_norm,
            "clip_coefficient": clip_coefficient,
            "shared_grad_norm_after_clip": clipped_shared_norm,
            "shared_parameter_delta_l2": shared_delta,
            "gene_parameter_delta_l2": gene_delta,
        }
        rows.append(row)
        snapshots.append(model.shared.detach().clone())
    return rows, snapshots


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("CUDA required; CPU fallback is forbidden")
    if args.steps < 1 or args.warmup_steps < 1:
        raise ValueError("steps and warmup-steps must be positive")
    if not args.scales or any(not math.isfinite(scale) or scale <= 0 for scale in args.scales):
        raise ValueError("Every AMP scale must be finite and positive")
    if args.learning_rate <= 0 or args.max_norm <= 0 or args.adam_eps <= 0:
        raise ValueError("learning-rate, max-norm and adam-eps must be positive")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or ROOT / "artifacts" / "mechanism" / ("amp_gene_clipping_" + timestamp)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "experiment_kind": "mechanism_not_learning_benefit",
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in AUDITED_SOURCES},
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(args.device),
        "gpu_capability": list(torch.cuda.get_device_capability(args.device)),
        "nvidia_driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "python": platform.python_version(),
        "utc_started_at": timestamp,
        "argv": sys.argv,
        "loss": "sum(shared ** 2) + sum(gene ** 2), fixed FP32 deterministic input",
        "interpretation_limit": "Gradient suppression need not imply equal Adam update suppression; actual deltas are measured. No RL effectiveness claim.",
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name or "amp_gene_clipping_" + timestamp,
        mode="online", config=config, dir=str(output_dir),
        tags=["mechanism", "cuda", "amp", "not_learning_result"],
    )
    if getattr(run.settings, "mode", "online") != "online":
        run.finish(exit_code=1)
        raise RuntimeError("An online W&B run is required")
    metadata = {"id": run.id, "entity": run.entity, "project": run.project, "url": run.url}
    (output_dir / "wandb_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    rows = []
    comparisons = []
    succeeded = False
    try:
        for history, warmup_steps in (("fresh_adam", 0), ("matched_moments", args.warmup_steps)):
            initial_state, optimizer_state = state_after_warmup(args, warmup_steps)
            for scale in args.scales:
                arms = {}
                for owned_gene, clip_owned_only in ((False, False), (False, True), (True, False), (True, True)):
                    arm_rows, snapshots = run_arm(args, initial_state, optimizer_state, scale, owned_gene, clip_owned_only, history)
                    arms[(owned_gene, clip_owned_only)] = (arm_rows, snapshots)
                    rows.extend(arm_rows)
                    for row in arm_rows:
                        run.log({"mechanism/" + key: value for key, value in row.items()})
                # Essential equivalence: changing the collection must do nothing
                # if every model parameter is optimizer-owned.
                for left, right in zip(arms[(True, False)][1], arms[(True, True)][1]):
                    assert torch.equal(left, right), "All-owned clipping collections are not exactly equivalent"
                for step, (legacy, candidate) in enumerate(zip(arms[(False, False)][0], arms[(False, True)][0])):
                    comparison = {
                        "history": history, "amp_scale": scale, "step_in_arm": step,
                        "all_owned_exact_equivalence": True,
                        "legacy_over_candidate_shared_gradient": legacy["shared_grad_norm_after_clip"] / candidate["shared_grad_norm_after_clip"],
                        "legacy_shared_parameter_delta_l2": legacy["shared_parameter_delta_l2"],
                        "candidate_shared_parameter_delta_l2": candidate["shared_parameter_delta_l2"],
                        "legacy_over_candidate_parameter_delta": legacy["shared_parameter_delta_l2"] / max(candidate["shared_parameter_delta_l2"], 1e-30),
                    }
                    comparisons.append(comparison)
                    run.log({"comparison/" + key: value for key, value in comparison.items()})
        with (output_dir / "per_scale.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = {"status": "passed", "wandb": metadata, "rows": rows, "comparisons": comparisons, "interpretation": config["interpretation_limit"]}
        (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        artifact = wandb.Artifact("amp-gene-clipping-" + run.id, type="mechanism-results")
        for filename in ("config.json", "wandb_run.json", "per_scale.csv", "results.json"):
            artifact.add_file(str(output_dir / filename))
        run.log_artifact(artifact)
        run.summary["mechanism/assertions_passed"] = True
        run.summary["mechanism/all_owned_exact_equivalence"] = True
        run.summary["mechanism/row_count"] = len(rows)
        run.summary["mechanism/learning_benefit_tested"] = False
        succeeded = True
        print(json.dumps({"status": "passed", "output_dir": str(output_dir), "wandb": metadata, "rows": len(rows)}, indent=2))
    finally:
        run.finish(exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
