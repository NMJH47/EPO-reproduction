"""Explicit precision validator; --cpu-reference-only never exercises CUDA.

Without that flag, all production metrics run on the requested CUDA device and
the command fails on a missing GPU; it never silently falls back to CPU.
This script does not train, alter a checkpoint, or claim learning benefit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mpmath as mp

from clipped_gaussian_reference import CASES, reference_ess, reference_logprob, reference_metrics


ATOL, RTOL = 1e-10, 1e-8


def source_hashes():
    base = Path(__file__).resolve().parent
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in ("clipped_gaussian_metrics.py", "clipped_gaussian_reference.py",
                         "validate_clipped_gaussian_metrics.py")}


def run_cpu_reference_validation(cases=None):
    """Validate independent quadrature convergence; does not validate GPU code."""
    cases = list(CASES if cases is None else cases)
    rows = []
    for case in cases:
        p, sp = case["p"]
        q, sq = case["q"]
        lo, hi = case.get("low", -1.0), case.get("high", 1.0)
        lower = reference_metrics(p, sp, q, sq, lo, hi, dps=80)
        higher = reference_metrics(p, sp, q, sq, lo, hi, dps=120)
        differences = {}
        valid = True
        with mp.workdps(120):
            for direction in ("p_to_q", "q_to_p"):
                for quantity in ("latent", "command"):
                    a = mp.mpf(lower[direction][quantity]["mp"])
                    b = mp.mpf(higher[direction][quantity]["mp"])
                    difference = abs(a - b)
                    differences[direction + "/" + quantity] = mp.nstr(difference, 15)
                    valid = valid and bool(difference <= mp.mpf(str(ATOL)) + mp.mpf(str(RTOL)) * abs(b))
                latent = mp.mpf(higher[direction]["latent"]["mp"])
                command = mp.mpf(higher[direction]["command"]["mp"])
                tol = mp.mpf(str(ATOL)) + mp.mpf(str(RTOL)) * abs(latent)
                valid = valid and bool(latent >= -tol and command >= -tol and command <= latent + tol)
        rows.append({"case": case, "reference_80": lower, "reference_120": higher,
                     "precision_differences": differences, "passed": valid})
    return {"status": "cpu_reference_passed_gpu_not_run" if all(r["passed"] for r in rows)
            else "cpu_reference_failed_gpu_not_run",
            "cpu_reference_passed": all(r["passed"] for r in rows),
            "gpu_validated": False, "training_benefit_tested": False,
            "cases": rows, "source_sha256": source_hashes()}


def run_cuda_validation(device="cuda:0", actual_cases=None):
    """Validate CUDA API against mpmath, optionally including real parameter rows.

    actual_cases: list of dict(name=..., p=[mu,sigma], q=[mu,sigma], low=-1,high=1).
    Returns a JSON-serializable report. Any assertion/numerical/domain failure is
    recorded and causes status=failed; callers must not claim validated on failure.
    """
    import torch
    import clipped_gaussian_metrics as metrics

    if not str(device).startswith("cuda"):
        raise ValueError("CUDA validation requires a CUDA device, no CPU fallback")
    device = torch.device(device)
    torch.cuda.set_device(device)  # Deliberately fails if CUDA is unavailable.
    cases = list(CASES) + list(actual_cases or [])
    rows, errors, api_checks = [], [], []

    def tensor(values, dtype=torch.float64):
        return torch.tensor(values, device=device, dtype=dtype)

    def check_close(name, got, expected, atol=ATOL, rtol=RTOL):
        error = abs(float(got) - float(expected))
        allowed = atol + rtol * abs(float(expected))
        passed = math.isfinite(float(got)) and math.isfinite(float(expected)) and error <= allowed
        entry = {"name": name, "value": float(got), "reference": float(expected),
                 "absolute_error": error, "allowed_error": allowed, "passed": passed}
        if not passed:
            errors.append(entry)
        return entry

    for case in cases:
        try:
            p, sp = case["p"]
            q, sq = case["q"]
            lo, hi = case.get("low", -1.0), case.get("high", 1.0)
            reference = reference_metrics(p, sp, q, sq, lo, hi, dps=120)
            result = metrics.clipped_gaussian_kl(tensor([p]), tensor([sp]), tensor([q]), tensor([sq]), lo, hi)
            comparisons = []
            for direction in ("p_to_q", "q_to_p"):
                for quantity in ("latent", "command"):
                    comparisons.append(check_close(direction + "/" + quantity,
                                                   result[direction][quantity + "joint"].item()
                                                   if quantity + "joint" in result[direction]
                                                   else result[direction][quantity + "_joint"].item(),
                                                   reference[direction][quantity]["float"]))
            for distribution in ("p", "q"):
                for side in ("left", "right"):
                    key = distribution + "_log_mass_" + side
                    comparisons.append(check_close(key,
                        result[distribution + "_components"]["log_mass_" + side].item(),
                        reference[key]["float"]))
            violation = bool(result["any_violation"].item())
            if violation:
                errors.append({"name": case["name"], "error": "CUDA numerical inequality/finite violation"})
            underflows = {d + "/" + region: bool(result[d + "_components"]["mass_underflow_" + region].item())
                          for d in ("p", "q") for region in ("left", "right", "interior")}
            negative = {d: bool(result[d]["negative_command_coordinate"].item()) for d in ("p_to_q", "q_to_p")}
            rows.append({"case": case, "comparisons": comparisons,
                         "reference": reference, "mass_underflows": underflows,
                         "negative_command_raw": negative, "any_violation": violation})
        except Exception as exc:
            entry = {"name": case.get("name", "unknown"), "error": repr(exc)}
            errors.append(entry)
            rows.append(entry)

    # GPU API properties and invalid-input refusals are separate from scalar math.
    def expect_rejection(name, fn):
        try:
            fn()
        except (TypeError, ValueError) as exc:
            api_checks.append({"name": name, "passed": True, "exception": repr(exc)})
        else:
            failure = {"name": name, "passed": False, "error": "input unexpectedly accepted"}
            api_checks.append(failure)
            errors.append(failure)

    zero, one = tensor([0.0]), tensor([1.0])
    expect_rejection("CPU tensors rejected", lambda: metrics.clipped_gaussian_kl(
        torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)))
    expect_rejection("float32 rejected", lambda: metrics.clipped_gaussian_kl(zero.float(), one.float(), zero.float(), one.float()))
    expect_rejection("requires_grad rejected", lambda: metrics.clipped_gaussian_kl(zero.clone().requires_grad_(True), one, zero, one))
    expect_rejection("sigma below supported range rejected", lambda: metrics.clipped_gaussian_kl(zero, tensor([1e-4]), zero, one))
    expect_rejection("zero sigma rejected", lambda: metrics.clipped_gaussian_kl(zero, zero, zero, one))
    expect_rejection("large mu rejected", lambda: metrics.clipped_gaussian_kl(tensor([101.0]), one, zero, one))
    expect_rejection("invalid bounds rejected", lambda: metrics.clipped_gaussian_kl(zero, one, zero, one, 1.0, -1.0))
    expect_rejection("outside command rejected", lambda: metrics.clipped_gaussian_log_prob(tensor([1.1]), zero, one))
    expect_rejection("nonfinite log weights rejected", lambda: metrics.logspace_ess(tensor([0.0, -float("inf")])) )

    try:
        # Broadcasting and coordinate additivity: three different coordinates.
        means_p = tensor([[0.0, 3.0, 0.8]])
        sigmas_p = tensor([[0.1, 1.0, 0.7]])
        means_q = tensor([[0.01, 4.0, 0.8000001]])
        sigmas_q = tensor([[0.11, 1.0, 0.7000001]])
        joint = metrics.clipped_gaussian_kl(means_p, sigmas_p, means_q, sigmas_q)
        for direction in ("p_to_q", "q_to_p"):
            for kind in ("latent", "command"):
                refs = [reference_metrics(*case["p"], *case["q"], dps=80)[direction][kind]["float"]
                        for case in (CASES[0], CASES[1], CASES[4])]
                api_checks.append(check_close("joint_sum/" + direction + "/" + kind,
                                               joint[direction][kind + "_joint"].item(), sum(refs), atol=3 * ATOL))
        broadcast = metrics.clipped_gaussian_kl(tensor([[0.0, 0.1], [0.2, 0.3]]), one,
                                                tensor([0.0, 0.1]), one)
        if tuple(broadcast["p_to_q"]["command_coordinate"].shape) != (2, 2):
            raise AssertionError("broadcast shape mismatch")
        self_kl = metrics.clipped_gaussian_kl(means_p, sigmas_p, means_p, sigmas_p)
        api_checks.append(check_close("self_command_joint", self_kl["p_to_q"]["command_joint"].item(), 0.0, atol=3 * ATOL))

        for mu, sigma in ((0.7, 0.8), (50.0, 1.0), (-21.0, 1.0)):
            values = [-1.0, 0.0, 1.0]
            result = metrics.clipped_gaussian_log_prob(tensor(values), tensor([mu]), tensor([sigma]))
            for index, action in enumerate(values):
                ref = reference_logprob(action, mu, sigma, dps=120)
                api_checks.append(check_close("endpoint_logprob/%s/%s" % (mu, action),
                                               result["coordinate"][index].item(), ref["logprob"]["float"]))

        raw = [-2.0, -0.2, 4.0]
        mus_b, sig_b = [0.3, -0.2, 1.0], [0.6, 0.8, 0.7]
        mus_t, sig_t = [0.4, 0.0, 1.4], [0.7, 0.9, 0.8]
        lr = metrics.joint_logratio_and_support(tensor(raw), tensor(mus_b), tensor(sig_b), tensor(mus_t), tensor(sig_t))
        raw_ref = sum((-0.5 * ((u-mt)/st)**2 - math.log(st))
                      - (-0.5 * ((u-mb)/sb)**2 - math.log(sb))
                      for u, mb, sb, mt, st in zip(raw, mus_b, sig_b, mus_t, sig_t))
        cmd_ref = sum(reference_logprob(max(-1.0, min(1.0, u)), mt, st)["logprob"]["float"]
                      - reference_logprob(max(-1.0, min(1.0, u)), mb, sb)["logprob"]["float"]
                      for u, mb, sb, mt, st in zip(raw, mus_b, sig_b, mus_t, sig_t))
        api_checks.append(check_close("raw_joint_logratio", lr["raw_logratio"].item(), raw_ref))
        api_checks.append(check_close("command_joint_logratio", lr["command_logratio"].item(), cmd_ref))
        for key, expected in (("raw_support", math.log(0.9) <= raw_ref <= math.log(1.1)),
                              ("command_support", math.log(0.9) <= cmd_ref <= math.log(1.1))):
            if bool(lr[key].item()) != expected:
                raise AssertionError("support mismatch: " + key)

        for index, logs in enumerate(([1000.0, 1000.0, 1000.0],
                                       [1000.0, -1000.0, 999.0], [-10000.0, -10000.1])):
            result = metrics.logspace_ess(tensor(logs))
            ref = reference_ess(logs)
            api_checks.append(check_close("ESS/%d" % index, result["effective_sample_size"].item(),
                                           ref["effective_sample_size"]["float"]))
            shifted = metrics.logspace_ess(tensor(logs) + 1e6)
            api_checks.append(check_close("ESS_shift_invariance/%d" % index,
                                           shifted["effective_sample_size"].item(), result["effective_sample_size"].item()))
    except Exception as exc:
        errors.append({"name": "API property checks", "error": repr(exc)})

    return {"status": "passed" if not errors else "failed", "gpu_validated": not errors,
            "training_benefit_tested": False, "device": str(device),
            "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda, "precision": "CUDA float64 distributions; CPU mpmath references",
            "cases": rows, "api_checks": api_checks, "errors": errors,
            "source_sha256": source_hashes(), "module_version": metrics.VERSION,
            "actual_cases_count": len(actual_cases or []),
            "scope": "Passed finite test cases only; not exhaustive domain or autograd validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--actual-cases-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    actual = json.loads(args.actual_cases_json.read_text()) if args.actual_cases_json else []
    if args.cpu_reference_only:
        report = run_cpu_reference_validation(list(CASES) + actual)
        passed = report["cpu_reference_passed"]
    else:
        report = run_cuda_validation(args.device, actual)
        passed = report["gpu_validated"]
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        print(json.dumps({"output": str(args.output), "status": report["status"],
                          "gpu_validated": report["gpu_validated"]}))
    else:
        print(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
