#!/usr/bin/env python3
"""CUDA/W&B numerical KL audit using a strictly loaded EPO checkpoint.

The dummy forward verifies the actual coef_cond logstd-to-sigma transform.
Synthetic local distribution perturbations are diagnostics, not learning gains
or measurements of update KL on the checkpoint's state occupancy distribution.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/diagnose_checkpoint_kl.py",
    "rl_games/rl_games/algos_torch/models.py",
    "rl_games/rl_games/algos_torch/network_builder.py",
    "rl_games/rl_games/algos_torch/torch_ext.py",
    "rl_games/rl_games/algos_torch/a2c_continuous.py",
    "rl_games/rl_games/common/schedulers.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "EPO-reproduction"))
    parser.add_argument("--wandb-group", default="ceiling-audit-20260913")
    return parser.parse_args()


def scheduler_direction(kl, threshold):
    if kl > 2 * threshold:
        return -1
    if kl < 0.5 * threshold:
        return 1
    return 0


def main():
    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("Explicit CUDA device required; CPU fallback is forbidden")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite output directory: " + str(output))
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.train_config.resolve()
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Checkpoint and training config must exist")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "rl_games"))
    import isaacgym  # noqa: F401; required before torch
    import torch
    import wandb
    import isaacgymenvs  # noqa: F401; register OmegaConf resolvers
    from omegaconf import OmegaConf
    from isaacgymenvs.train import configure_torch_jit_for_modern_gpus
    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.algos_torch.torch_ext import policy_kl

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    torch.cuda.set_device(torch.device(args.device))
    configure_torch_jit_for_modern_gpus()
    cfg = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    params = copy.deepcopy(cfg["train"]["params"])
    if params["model"]["name"] != "continuous_a2c_logstd":
        raise ValueError("Only continuous_a2c_logstd checkpoints are supported")
    if params["network"]["space"]["continuous"]["fixed_sigma"] != "coef_cond":
        raise ValueError("This diagnostic requires per-gene coef_cond sigma")
    if params["config"]["expl_type"] != "mixed_expl_learn_param":
        raise ValueError("This diagnostic requires mixed_expl_learn_param")
    if not params["config"].get("normalize_input", False):
        raise ValueError("Observation dimension inference requires the checkpoint observation RMS")
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=args.device)
    state = checkpoint[0] if 0 in checkpoint else checkpoint
    model_state = state["model"]
    gene_count, latent_dim = model_state["a2c_network.extra_params"].shape
    raw_sigma_table = model_state["a2c_network.sigma"].detach().clone()
    if raw_sigma_table.ndim != 2 or raw_sigma_table.shape[0] != gene_count:
        raise ValueError("Sigma table shape does not match gene table")
    action_dim = raw_sigma_table.shape[1]
    obs_dim = model_state["running_mean_std.running_mean"].numel()
    train_envs = int(cfg["task"]["env"]["numEnvs"])
    train_block = int(params["config"]["expl_coef_block_size"])
    if train_envs != gene_count * train_block:
        raise ValueError("Checkpoint gene count and training config disagree")
    # Match the training implementation's CPU linspace before device transfer.
    coef_ids = torch.linspace(50.0, 0.0, gene_count).to(args.device)
    model = ModelBuilder().load(params).build({
        "actions_num": action_dim, "input_shape": (obs_dim + 1,),
        "num_seqs": gene_count, "value_size": 1,
        "normalize_value": params["config"].get("normalize_value", False),
        "normalize_input": True, "type": "extra_param", "param_size": latent_dim,
        "coef_ids": coef_ids, "coef_id_idx": obs_dim,
    }).to(args.device)
    model.load_state_dict(model_state, strict=True)
    if "running_mean_std" in state:
        model.running_mean_std.load_state_dict(state["running_mean_std"], strict=True)
    model.eval()
    model.requires_grad_(False)
    checkpoint_frame = int(state.get("frame", -1))
    checkpoint_epoch = int(state.get("epoch", -1))
    del checkpoint, state, model_state
    torch.cuda.empty_cache()
    with torch.no_grad():
        observations = torch.zeros(gene_count, obs_dim + 1, device=args.device)
        observations[:, -1] = coef_ids
        states = tuple(value.to(args.device) for value in model.get_default_rnn_state()) if model.is_rnn() else None
        decoded_ids = (observations[:, -1:] == model.a2c_network.sigma_ids).float().argmax(dim=1)
        if not torch.equal(decoded_ids, torch.arange(gene_count, device=args.device)):
            raise AssertionError("Dummy forward sigma lookup does not preserve gene identity")
        forward = model({"is_train": True, "obs": observations,
                         "prev_actions": torch.zeros(gene_count, action_dim, device=args.device),
                         "rnn_states": states, "seq_length": 1})
        sigmas = forward["sigmas"].detach().clone()
        activated_logstd = model.a2c_network.sigma_act(raw_sigma_table)
        if not torch.equal(sigmas, torch.exp(activated_logstd)):
            raise AssertionError("Real model sigma does not match exp(sigma_act(checkpoint table))")
        if not torch.isfinite(sigmas).all().item() or not (sigmas > 0).all().item():
            raise AssertionError("Checkpoint produces nonfinite or nonpositive standard deviations")
    output.mkdir(parents=True)
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    with tarfile.open(str(output / "source_snapshot.tar.gz"), "w:gz") as archive:
        for name in SOURCES:
            archive.add(str(ROOT / name), arcname=name)
    metadata = {
        "experiment_kind": "checkpoint_sigma_numerical_kl_diagnostic_not_learning_benefit",
        "checkpoint_sha256": checkpoint_hash, "checkpoint_path": str(checkpoint_path),
        "checkpoint_frame": checkpoint_frame, "checkpoint_epoch": checkpoint_epoch,
        "training_config_sha256": sha256_file(config_path), "training_seed": cfg.get("seed"),
        "source_sha256": source_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
        "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "sigma_transform_verified_by_real_model_forward": True,
        "source_sigma_activation": params["network"]["space"]["continuous"].get("sigma_activation"),
        "strict_model_load": True, "gene_count": gene_count, "action_dimensions": action_dim,
        "master_gene": gene_count - 1,
        "kl_threshold": float(params["config"]["kl_threshold"]),
        "state_occupancy_measured": False,
        "synthetic_mean_origin": "zero, exploiting translation invariance of both Gaussian KL formulas",
        "scheduler_decisions_scope": "hypothetical single-gene diagnostic only; actual minibatch scheduler decisions require real old/new parameters",
    }
    write_json(output / "protocol.json", metadata)
    write_json(output / "training_config.resolved.json", cfg)
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                     name=output.name, mode="online", resume="never", job_type="mechanism",
                     config=metadata, dir=str(output), tags=["diagnostic", "exact_kl", "not_learning_benefit"])
    run_metadata = {"id": run.id, "entity": run.entity, "project": run.project, "url": run.url}
    write_json(output / "wandb_run.json", run_metadata)
    succeeded = False
    try:
        gene_rows = []
        action_rows = []
        kl_rows = []
        with torch.no_grad():
            for gene in range(gene_count):
                sigma = sigmas[gene]
                gene_row = {"gene": gene, "is_master": gene == gene_count - 1,
                            "sigma_min": sigma.min().item(), "sigma_mean": sigma.mean().item(),
                            "sigma_max": sigma.max().item(),
                            "sigma_under_0_1_fraction": (sigma < 0.1).float().mean().item()}
                gene_rows.append(gene_row)
                for action in range(action_dim):
                    action_rows.append({"gene": gene, "action": action,
                                        "raw_sigma_parameter": raw_sigma_table[gene, action].item(),
                                        "activated_logstd": activated_logstd[gene, action].item(),
                                        "actual_sigma": sigma[action].item()})
                zero = torch.zeros_like(sigma)
                perturbations = (
                    ("self", zero, sigma),
                    ("mean_absolute_0_001", zero + 0.001, sigma),
                    ("mean_absolute_0_01", zero + 0.01, sigma),
                    ("mean_relative_0_01sigma", sigma * 0.01, sigma),
                    ("sigma_factor_1_01", zero, sigma * 1.01),
                    ("combined_relative_0_01", sigma * 0.01, sigma * 1.01),
                )
                for label, other_mean, other_sigma in perturbations:
                    legacy = policy_kl(zero.unsqueeze(0), sigma.unsqueeze(0), other_mean.unsqueeze(0), other_sigma.unsqueeze(0), reduce=False)[0]
                    exact_coordinate = torch.distributions.kl_divergence(
                        torch.distributions.Normal(zero.double(), sigma.double()),
                        torch.distributions.Normal(other_mean.double(), other_sigma.double()))
                    exact = exact_coordinate.sum()
                    if exact_coordinate.min().item() < -1e-12:
                        raise AssertionError("Exact Normal KL must be nonnegative")
                    if label == "self" and abs(exact.item()) > 1e-12:
                        raise AssertionError("Exact self KL must equal zero")
                    row = {"gene": gene, "is_master": gene == gene_count - 1, "perturbation": label,
                           "legacy_kl_float32": legacy.item(), "exact_kl_float64": exact.item(),
                           "legacy_minus_exact": legacy.item() - exact.item(),
                           "legacy_negative": legacy.item() < 0,
                           "legacy_hypothetical_lr_direction": scheduler_direction(legacy.item(), metadata["kl_threshold"]),
                           "exact_hypothetical_lr_direction": scheduler_direction(exact.item(), metadata["kl_threshold"])}
                    row["hypothetical_lr_direction_disagrees"] = row["legacy_hypothetical_lr_direction"] != row["exact_hypothetical_lr_direction"]
                    kl_rows.append(row)
                    run.log({"kl_diagnostic/" + key: value for key, value in row.items()})
        for filename, rows in (("sigma_by_gene.csv", gene_rows), ("sigma_by_action.csv", action_rows), ("kl_comparisons.csv", kl_rows)):
            with (output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        master_self = next(row for row in kl_rows if row["is_master"] and row["perturbation"] == "self")
        master_bias_ratio = abs(master_self["legacy_minus_exact"]) / metadata["kl_threshold"]
        priority = ("self-KL bias is small relative to configured threshold for this master; reduce current-priority absent real-update evidence"
                    if master_bias_ratio < 0.1 else
                    "self-KL bias is material relative to configured threshold; inspect real-update scheduler decisions before attributing performance")
        summary = {"status": "passed", "assertions_passed": True, "master_sigma": gene_rows[-1],
                   "master_self_kl": master_self, "master_self_bias_over_kl_threshold": master_bias_ratio,
                   "priority_interpretation": priority,
                   "synthetic_kl_rows": len(kl_rows),
                   "hypothetical_scheduler_disagreements": sum(row["hypothetical_lr_direction_disagrees"] for row in kl_rows),
                   "learning_benefit_tested": False, "executed_action_kl_measured": False}
        write_json(output / "summary.json", summary)
        for key, value in master_self.items():
            run.summary["master_self/" + key] = value
        run.summary["master_self_bias_over_kl_threshold"] = master_bias_ratio
        run.summary["priority_interpretation"] = priority
        run.summary["assertions_passed"] = True
        artifact = wandb.Artifact("checkpoint-kl-" + run.id, type="mechanism-results", metadata={"checkpoint_sha256": checkpoint_hash})
        for filename in ("protocol.json", "training_config.resolved.json", "wandb_run.json", "sigma_by_gene.csv", "sigma_by_action.csv", "kl_comparisons.csv", "summary.json", "source_snapshot.tar.gz"):
            artifact.add_file(str(output / filename))
        run.log_artifact(artifact)
        succeeded = True
        print(json.dumps({"output": str(output), "wandb": run_metadata, "summary": summary}, indent=2))
    finally:
        run.finish(exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
