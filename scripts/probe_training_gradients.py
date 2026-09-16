"""Run the unchanged training entrypoint with a paired Adam gradient probe.

Set EPO_GRAD_PROBE_OUTPUT to the output JSON path and optionally
EPO_GRAD_PROBE_STEPS (default: 1). All command-line arguments pass unchanged
to isaacgymenvs.train / Hydra. Each probed update first evaluates owned-only
clipping, restores its input state, and then executes the original update.
Only the original legacy update remains effective in the training trajectory.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile

# Isaac Gym must be loaded before torch in this environment.
import isaacgym  # noqa: F401
import torch

from rl_games.common.a2c_common import A2CBase
from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/probe_training_gradients.py",
    "rl_games/rl_games/common/a2c_common.py",
    "rl_games/rl_games/algos_torch/a2c_continuous.py",
    "isaacgymenvs/utils/rlgames_utils.py",
)


def cpu_copy(value):
    """Transfer snapshots only; optimization and vector metrics stay on CUDA."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def gradient_norm(parameters):
    square_sum = 0.0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all().item())
        value = torch.norm(gradient, p=2, dtype=torch.float64).item()
        square_sum += value * value
    return math.sqrt(square_sum), finite


def snapshot(agent, named_parameters):
    return {
        "parameters": {name: parameter.detach().cpu().clone() for name, parameter in named_parameters},
        "gradients": {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone() for name, parameter in named_parameters},
        "optimizer": cpu_copy(agent.optimizer.state_dict()),
        "scaler": copy.deepcopy(agent.scaler.state_dict()),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore(agent, named_parameters, saved):
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(saved["parameters"][name].to(parameter.device))
            gradient = saved["gradients"][name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    agent.optimizer.load_state_dict(saved["optimizer"])
    agent.scaler.load_state_dict(saved["scaler"])
    # A completed update clears this map. It must also be cleared if the
    # diagnostic branch failed between unscale_ and update().
    agent.scaler._per_optimizer_states.clear()
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])


def parameter_delta_metrics(named_parameters, before, candidate, owned_ids):
    candidate_square = 0.0
    legacy_square = 0.0
    inner_product = 0.0
    difference_square = 0.0
    unowned_unchanged = True
    legacy_finite = True
    candidate_finite = True
    with torch.no_grad():
        for name, parameter in named_parameters:
            previous = before[name].to(parameter.device)
            if id(parameter) not in owned_ids:
                unowned_unchanged = unowned_unchanged and bool(torch.equal(parameter, previous))
                continue
            proposed = candidate[name].to(parameter.device)
            # Compute norms and dot products on CUDA; release per-parameter
            # temporary tensors before moving to the next parameter.
            candidate_delta = (proposed - previous).double().reshape(-1)
            legacy_delta = (parameter.detach() - previous).double().reshape(-1)
            candidate_finite = candidate_finite and bool(torch.isfinite(candidate_delta).all().item())
            legacy_finite = legacy_finite and bool(torch.isfinite(legacy_delta).all().item())
            candidate_square += torch.dot(candidate_delta, candidate_delta).item()
            legacy_square += torch.dot(legacy_delta, legacy_delta).item()
            inner_product += torch.dot(candidate_delta, legacy_delta).item()
            difference = candidate_delta - legacy_delta
            difference_square += torch.dot(difference, difference).item()
    candidate_norm = math.sqrt(candidate_square)
    legacy_norm = math.sqrt(legacy_square)
    denominator = candidate_norm * legacy_norm
    return {
        "candidate_owned_delta_l2": candidate_norm,
        "legacy_owned_delta_l2": legacy_norm,
        "legacy_over_candidate_delta_l2": legacy_norm / candidate_norm if candidate_norm > 0 else None,
        "candidate_legacy_delta_cosine": inner_product / denominator if denominator > 0 else None,
        "candidate_legacy_delta_difference_l2": math.sqrt(difference_square),
        "unowned_parameters_unchanged": unowned_unchanged,
        "candidate_delta_finite": candidate_finite,
        "legacy_delta_finite": legacy_finite,
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def save_result(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)


def install_wandb_finalization(output, result):
    """Validate and archive the probe before the training observer ends its run.

    These hooks affect this diagnostic process only; the training implementation
    is unchanged. Count incompleteness is a failed run even if PPO returned
    normally. Initialization/rollout failures also reach the fallback in main.
    """
    original_before_init = WandbAlgoObserver.before_init
    original_finish = WandbAlgoObserver.finish
    source_bytes = {path: (ROOT / path).read_bytes() for path in SOURCE_PATHS}
    lifecycle = {"observers": [], "finalized": False}

    def before_init(observer, *args, **kwargs):
        lifecycle["observers"].append(observer)
        if str(observer.cfg.get("wandb_mode", "online")).lower() != "online":
            raise RuntimeError("The gradient probe requires online W&B logging")
        return original_before_init(observer, *args, **kwargs)

    def finish(observer, exit_code=0):
        if observer.finished:
            return
        active_error = sys.exc_info()[1]
        expected = int(result["max_probes"])
        observed = len(result["probes"])
        count_complete = observed == expected
        effective_exit_code = exit_code or (1 if active_error is not None or not count_complete else 0)
        status = "failed" if exit_code or active_error is not None else ("completed" if count_complete else "incomplete")
        result.update({
            "status": status,
            "expected_probes": expected,
            "observed_probes": observed,
            "probe_count_complete": count_complete,
            "utc_validated_at": datetime.now(timezone.utc).isoformat(),
            "training_exit": {
                "training_loop_exit_code": exit_code,
                "effective_wandb_exit_code": effective_exit_code,
                "status": "training_loop_returned" if active_error is None and exit_code == 0 else "training_loop_failed",
            },
        })
        if active_error is not None:
            result["training_exit"].update(exception=type(active_error).__name__, message=str(active_error))
        if not count_complete:
            result["completion_error"] = "Expected {} gradient probes, observed {}".format(expected, observed)

        archive_error = None
        try:
            if observer.run is None:
                raise RuntimeError("Probe finalization has no online W&B run")
            if observer.writer is not None:
                # Detect buffered metric failures before declaring completeness.
                observer.writer.flush()
            result["wandb"] = {"id": observer.run.id, "url": observer.run.url,
                               "artifact_name": "actual-gradient-probe-" + observer.run.id}
            save_result(output, result)
            observer.run.summary.update({
                "mechanism_probe/status": result["status"],
                "mechanism_probe/expected_probes": expected,
                "mechanism_probe/observed_probes": observed,
                "mechanism_probe/count_complete": count_complete,
                "mechanism_probe/effective_exit_code": effective_exit_code,
                "mechanism_probe/raw_json_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            })
            import wandb

            artifact = wandb.Artifact(result["wandb"]["artifact_name"], type="mechanism-probe",
                                      metadata={"status": status, "expected_probes": expected,
                                                "observed_probes": observed,
                                                "source_sha256": result["source_sha256"]})
            artifact.add_file(str(output), name="probe.json")
            # Freeze source bytes captured when this wrapper installed the hooks,
            # including the wrapper itself, instead of depending on git log_code.
            with tempfile.TemporaryDirectory(prefix="epo_gradient_probe_source_") as folder:
                for path, content in source_bytes.items():
                    if hashlib.sha256(content).hexdigest() != result["source_sha256"][path]:
                        raise RuntimeError("Probe source changed during installation: " + path)
                    snapshot_path = Path(folder) / path
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_path.write_bytes(content)
                    artifact.add_file(str(snapshot_path), name="source/" + path)
                observer.run.log_artifact(artifact).wait()
        except BaseException as error:
            archive_error = error
            effective_exit_code = effective_exit_code or 1
            result["status"] = "failed"
            result["finalization_error"] = {"exception": type(error).__name__, "message": str(error)}
            result["training_exit"]["effective_wandb_exit_code"] = effective_exit_code
            save_result(output, result)
            if observer.run is not None:
                try:
                    observer.run.summary.update({"mechanism_probe/status": "failed",
                                                 "mechanism_probe/effective_exit_code": effective_exit_code,
                                                 "mechanism_probe/finalization_error": str(error)})
                except Exception:
                    pass
        finally:
            # Crucially, the existing observer sees failure before run.finish;
            # incomplete probes can never first become a successful online run.
            try:
                original_finish(observer, exit_code=effective_exit_code)
            except BaseException as error:
                result["status"] = "failed"
                result["wandb_finish_error"] = {"exception": type(error).__name__, "message": str(error)}
                save_result(output, result)
                raise
            finally:
                lifecycle["finalized"] = True
        if archive_error is not None:
            raise RuntimeError("Could not archive the gradient probe before W&B finalization") from archive_error
        if not count_complete:
            raise RuntimeError(result["completion_error"])

    WandbAlgoObserver.before_init = before_init
    WandbAlgoObserver.finish = finish
    return lifecycle


def install_probe(output, max_probes):
    original_update = A2CBase.trancate_gradients_and_step
    result = {
        "experiment_kind": "paired_actual_gradient_mechanism_not_learning_benefit",
        "trajectory_branch": "legacy_original_update",
        "candidate_branch": "optimizer_owned_gradient_clipping_only",
        "interpretation": "Same actual gradient, parameters, Adam moments and AMP scale per branch; no claim of exact simulator or checkpoint RNG continuation.",
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS},
        "argv": sys.argv,
        "utc_started_at": datetime.now(timezone.utc).isoformat(),
        "max_probes": max_probes,
        "expected_probes": max_probes,
        "observed_probes": 0,
        "probes": [],
        "status": "installed",
    }
    save_result(output, result)

    def probed_update(agent):
        if len(result["probes"]) >= max_probes:
            return original_update(agent)
        if agent.multi_gpu:
            raise RuntimeError("The paired gradient probe currently supports one GPU only")
        if not agent.truncate_grads or not agent.scaler.is_enabled():
            raise RuntimeError("The probe requires gradient clipping and enabled GradScaler")
        if agent.scaler._per_optimizer_states:
            raise RuntimeError("Expected a fresh GradScaler update before the gradient probe")
        named_parameters = list(agent.model.named_parameters())
        optimizer_parameters = [parameter for group in agent.optimizer.param_groups for parameter in group["params"]]
        owned_ids = {id(parameter) for parameter in optimizer_parameters}
        if len(owned_ids) != len(optimizer_parameters):
            raise RuntimeError("Duplicate parameters in optimizer groups are not supported")
        model_ids = {id(parameter) for _, parameter in named_parameters}
        if not owned_ids.issubset(model_ids):
            raise RuntimeError("Optimizer contains parameters absent from the model; this probe expects the identity-preserving E51 baseline")
        owned = [parameter for _, parameter in named_parameters if id(parameter) in owned_ids]
        unowned = [parameter for _, parameter in named_parameters if id(parameter) not in owned_ids]
        scaled_owned_norm, scaled_owned_finite = gradient_norm(owned)
        scaled_unowned_norm, scaled_unowned_finite = gradient_norm(unowned)
        scale = agent.scaler.get_scale()
        saved = snapshot(agent, named_parameters)
        candidate_parameters = None
        try:
            agent.scaler.unscale_(agent.optimizer)
            owned_unscaled_norm, owned_unscaled_finite = gradient_norm(owned)
            unowned_unscaled_norm, unowned_unscaled_finite = gradient_norm(unowned)
            candidate_preclip_norm = torch.nn.utils.clip_grad_norm_(owned, agent.grad_norm).item()
            candidate_postclip_norm, candidate_postclip_finite = gradient_norm(owned)
            agent.scaler.step(agent.optimizer)
            agent.scaler.update()
            candidate_scale_after = agent.scaler.get_scale()
            candidate_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in named_parameters if id(parameter) in owned_ids
            }
        finally:
            # This rollback also runs if candidate computation raises.
            restore(agent, named_parameters, saved)
        # Preserve the original function, including its exact clipping and
        # scaler behavior, as the only step effective in the live agent.
        original_update(agent)
        legacy_postclip_norm, legacy_postclip_finite = gradient_norm(owned)
        deltas = parameter_delta_metrics(named_parameters, saved["parameters"], candidate_parameters, owned_ids)
        if not deltas["unowned_parameters_unchanged"]:
            raise AssertionError("The original update unexpectedly modified unowned parameters")
        row = {
            "probe_index": len(result["probes"]),
            "epoch": int(agent.epoch_num), "frame": int(agent.frame),
            "amp_scale_before": scale,
            "amp_scale_after_candidate": candidate_scale_after,
            "amp_scale_after_legacy": agent.scaler.get_scale(),
            "learning_rate": float(agent.optimizer.param_groups[0]["lr"]),
            "clip_threshold": float(agent.grad_norm),
            "owned_parameter_tensors": len(owned),
            "unowned_parameter_tensors": len(unowned),
            "unowned_parameter_names": [name for name, parameter in named_parameters if id(parameter) not in owned_ids],
            "owned_grad_norm_scaled": scaled_owned_norm,
            "unowned_grad_norm_scaled": scaled_unowned_norm,
            "owned_grad_norm_after_unscale": owned_unscaled_norm,
            "unowned_grad_norm_after_unscale": unowned_unscaled_norm,
            "legacy_preclip_norm_from_same_grads": math.hypot(owned_unscaled_norm, unowned_unscaled_norm),
            "candidate_preclip_norm": candidate_preclip_norm,
            "candidate_owned_grad_norm_after_clip": candidate_postclip_norm,
            "legacy_owned_grad_norm_after_clip": legacy_postclip_norm,
            "all_input_gradients_finite": scaled_owned_finite and scaled_unowned_finite,
            "all_unscaled_gradients_finite": owned_unscaled_finite and unowned_unscaled_finite,
            "candidate_clipped_gradients_finite": candidate_postclip_finite,
            "legacy_clipped_gradients_finite": legacy_postclip_finite,
            **deltas,
        }
        if not unowned and row["candidate_legacy_delta_difference_l2"] != 0.0:
            raise AssertionError("All-owned control must produce identical parameter updates")
        result["probes"].append(row)
        result["observed_probes"] = len(result["probes"])
        result["status"] = "probes_completed" if len(result["probes"]) == max_probes else "collecting"
        save_result(output, result)
        for key, value in row.items():
            if isinstance(value, (int, float, bool)) and math.isfinite(value):
                agent.writer.add_scalar("mechanism_probe/" + key, value, int(agent.frame))
        agent.writer.flush()
        print("GRADIENT_PROBE " + json.dumps(json_safe(row), sort_keys=True, allow_nan=False), flush=True)

    A2CBase.trancate_gradients_and_step = probed_update
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is forbidden")
    max_probes = int(os.getenv("EPO_GRAD_PROBE_STEPS", "1"))
    if max_probes < 1:
        raise ValueError("EPO_GRAD_PROBE_STEPS must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(os.getenv("EPO_GRAD_PROBE_OUTPUT", str(ROOT / "artifacts" / "mechanism" / ("training_gradient_probe_" + timestamp + ".json")))).resolve()
    result = install_probe(output, max_probes)
    lifecycle = install_wandb_finalization(output, result)
    try:
        runpy.run_module("isaacgymenvs.train", run_name="__main__")
    except BaseException as error:
        # Some startup failures occur before the training entrypoint's finally.
        # Finalize any initialized observer while this exception is still active.
        for observer in lifecycle["observers"]:
            if observer.run is not None and not observer.finished:
                try:
                    observer.finish(exit_code=1)
                except BaseException as finalization_error:
                    print("Probe failure finalization: " + str(finalization_error), flush=True)
        if not lifecycle["finalized"]:
            result["status"] = "failed"
            result["observed_probes"] = len(result["probes"])
            result["training_exit"] = {"exception": type(error).__name__, "message": str(error)}
            save_result(output, result)
        raise
    if not lifecycle["finalized"]:
        result["status"] = "failed"
        result["completion_error"] = "Training returned without the required online W&B probe finalization"
        save_result(output, result)
        raise RuntimeError(result["completion_error"])


if __name__ == "__main__":
    main()
