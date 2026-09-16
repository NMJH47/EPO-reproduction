"""Diagnostic-only clipped Gaussian metrics; every tensor calculation is CUDA FP64.

No CPU fallback and no autograd contract. Inputs must be detached CUDA float64
tensors. All outputs containing numbers computed from inputs remain on CUDA.
The last dimension is the action coordinate; preceding dimensions broadcast.

Supported diagnostic domain: |mu|, |bounds| <= 100, sigma in [1e-3, 100],
interval width >= 1e-3. Inputs outside this domain raise NumericalDomainError;
they are not silently clipped or sent to a CPU reference. This domain is an
explicit validation target, not a statement that GPU validation already passed.
"""
import math

import torch


VERSION = "clipped_gaussian_metrics_v1_diagnostic"
ATOL = 1e-10
RTOL = 1e-8
MIN_SIGMA = 1e-3
MAX_SIGMA = 100.0
MAX_ABS_PARAMETER = 100.0
MIN_INTERVAL_WIDTH = 1e-3
_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_LOG_2 = math.log(2.0)


class NumericalDomainError(ValueError):
    """Input is outside the declared diagnostic domain; no fallback occurred."""


def _tensor(name, value, finite=True):
    if not isinstance(value, torch.Tensor):
        raise TypeError("%s must be a CUDA float64 Tensor" % name)
    if value.device.type != "cuda" or value.dtype != torch.float64:
        raise TypeError("%s must be CUDA float64; CPU fallback/conversion is forbidden" % name)
    if value.requires_grad:
        raise ValueError("%s must be detached; this module is diagnostic-only" % name)
    if value.ndim < 1 or value.numel() == 0:
        raise ValueError("%s must have a nonempty action/sample dimension" % name)
    if finite and not bool(torch.isfinite(value).all().item()):
        raise NumericalDomainError("%s contains nonfinite values" % name)
    return value


def _bound(name, value, anchor):
    if isinstance(value, torch.Tensor):
        if value.device != anchor.device or value.dtype != torch.float64:
            raise TypeError("%s must have the same CUDA device and float64 dtype" % name)
        if value.requires_grad:
            raise ValueError("bounds must be detached")
        return value
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise TypeError("%s must be a finite scalar or CUDA float64 tensor" % name)
    return torch.tensor(float(value), dtype=torch.float64, device=anchor.device)


def _parameters(mu_p, sigma_p, mu_q, sigma_q, low, high):
    values = [_tensor(n, v) for n, v in zip(
        ("mu_p", "sigma_p", "mu_q", "sigma_q"),
        (mu_p, sigma_p, mu_q, sigma_q))]
    if any(v.device != values[0].device for v in values):
        raise TypeError("all parameters must be on the same CUDA device")
    lo = _bound("low", low, values[0])
    hi = _bound("high", high, values[0])
    p, sp, q, sq, lo, hi = torch.broadcast_tensors(*(values + [lo, hi]))
    checks = {
        "nonfinite bound": ~torch.isfinite(lo) | ~torch.isfinite(hi),
        "mean/bound outside absolute limit":
            (p.abs() > MAX_ABS_PARAMETER) | (q.abs() > MAX_ABS_PARAMETER)
            | (lo.abs() > MAX_ABS_PARAMETER) | (hi.abs() > MAX_ABS_PARAMETER),
        "sigma outside [1e-3,100]":
            (sp < MIN_SIGMA) | (sq < MIN_SIGMA) | (sp > MAX_SIGMA) | (sq > MAX_SIGMA),
        "interval width below 1e-3": (hi - lo) < MIN_INTERVAL_WIDTH,
    }
    for name, mask in checks.items():
        if bool(mask.any().item()):
            raise NumericalDomainError(name + "; no fallback occurred")
    return p, sp, q, sq, lo, hi


def _log_ndtr(x):
    """FP64 log Phi using erfc centrally and a Mills expansion below -20.

    Twelve alternating correction terms at |x|>=20 have truncation error well
    below FP64 roundoff. Domain validation bounds |x| and prevents x*x overflow.
    No torch.special.log_ndtr dependency (compatible with Torch 1.8.1).
    """
    negative_central = x.clamp(min=-20.0, max=0.0)
    central = torch.log(0.5 * torch.erfc(-negative_central / _SQRT_2))
    positive_x = x.clamp(min=0.0)
    positive = torch.log1p(-0.5 * torch.erfc(positive_x / _SQRT_2))
    tail_x = torch.where(x < -20.0, -x, torch.full_like(x, 20.0))
    inv_square = tail_x.reciprocal().square()
    term = torch.ones_like(x)
    series = torch.ones_like(x)
    for n in range(1, 13):
        term = term * (-(2 * n - 1)) * inv_square
        series = series + term
    tail = -0.5 * tail_x.square() - torch.log(tail_x) - 0.5 * _LOG_2PI + torch.log(series)
    return torch.where(x < -20.0, tail, torch.where(x > 0.0, positive, central))


def _log_one_minus_exp(x):
    # x<=0. At x=0 the mathematically correct result is -inf.
    return torch.where(x < -_LOG_2, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def _exp_difference(log_a, log_b):
    a_larger = log_a >= log_b
    bigger = torch.maximum(log_a, log_b)
    smaller = torch.minimum(log_a, log_b)
    magnitude = torch.exp(bigger) * (-torch.expm1(smaller - bigger))
    return torch.where(a_larger, magnitude, -magnitude)


def _components(mu, sigma, low, high):
    alpha, beta = (low - mu) / sigma, (high - mu) / sigma
    log_left, log_right = _log_ndtr(alpha), _log_ndtr(-beta)
    log_cdf_beta, log_sf_alpha = _log_ndtr(beta), _log_ndtr(-alpha)
    log_z_left = log_cdf_beta + _log_one_minus_exp(log_left - log_cdf_beta)
    log_z_right = log_sf_alpha + _log_one_minus_exp(log_right - log_sf_alpha)
    # Crossing zero avoids cancellation between two near-one CDFs.
    z_cross = 1.0 - torch.exp(log_left) - torch.exp(log_right)
    log_z_cross = torch.log(z_cross)
    log_interior = torch.where(beta <= 0.0, log_z_left,
                              torch.where(alpha >= 0.0, log_z_right, log_z_cross))
    log_phi_alpha = -0.5 * alpha.square() - 0.5 * _LOG_2PI
    log_phi_beta = -0.5 * beta.square() - 0.5 * _LOG_2PI
    phi_alpha, phi_beta = torch.exp(log_phi_alpha), torch.exp(log_phi_beta)
    interior = torch.exp(log_interior)
    t1 = _exp_difference(log_phi_alpha, log_phi_beta)
    t2 = interior + alpha * phi_alpha - beta * phi_beta
    left, right = torch.exp(log_left), torch.exp(log_right)
    return {
        "log_mass_left": log_left, "log_mass_right": log_right,
        "log_mass_interior": log_interior,
        "mass_left": left, "mass_right": right, "mass_interior": interior,
        "first_standardized_interior_moment": t1,
        "second_standardized_interior_moment": t2,
        # Underflow is reported, not treated as an exact mathematical zero mass.
        "mass_underflow_left": (left == 0.0) & torch.isfinite(log_left),
        "mass_underflow_right": (right == 0.0) & torch.isfinite(log_right),
        "mass_underflow_interior": (interior == 0.0) & torch.isfinite(log_interior),
    }


def _one_direction(mu_p, sigma_p, mu_q, sigma_q, cp, cq):
    delta = mu_p - mu_q
    log_sigma_ratio = torch.log(sigma_q / sigma_p)
    latent = log_sigma_ratio + (sigma_p.square() + delta.square()) / (2.0 * sigma_q.square()) - 0.5
    atoms = cp["mass_left"] * (cp["log_mass_left"] - cq["log_mass_left"])
    atoms = atoms + cp["mass_right"] * (cp["log_mass_right"] - cq["log_mass_right"])
    z = cp["mass_interior"]
    t1 = cp["first_standardized_interior_moment"]
    t2 = cp["second_standardized_interior_moment"]
    command = atoms + z * log_sigma_ratio
    command = command + (delta.square() * z + 2.0 * delta * sigma_p * t1
                         + sigma_p.square() * t2) / (2.0 * sigma_q.square()) - 0.5 * t2
    tolerance = ATOL + RTOL * latent.abs()
    finite = torch.isfinite(latent) & torch.isfinite(command)
    violation = (~finite) | (latent < -tolerance) | (command < -tolerance) | (command > latent + tolerance)
    return {
        "latent_coordinate": latent, "command_coordinate": command,
        "latent_joint": latent.sum(dim=-1), "command_joint": command.sum(dim=-1),
        "gap_coordinate": latent - command, "gap_joint": (latent - command).sum(dim=-1),
        "negative_latent_coordinate": latent < 0.0,
        "negative_command_coordinate": command < 0.0,
        "violation_coordinate": violation,
        "any_violation": violation.any(),
        "coordinate_tolerance": tolerance,
    }


@torch.no_grad()
def clipped_gaussian_kl(mu_p, sigma_p, mu_q, sigma_q, low=-1.0, high=1.0):
    """Return p_to_q/q_to_p dicts plus endpoint/interior component diagnostics.

    Direction dictionaries contain latent_coordinate, command_coordinate,
    latent_joint, command_joint, gap_coordinate/joint and numerical flags.
    Negative KL is NEVER clamped to zero. Check any_violation before inference.
    """
    p, sp, q, sq, lo, hi = _parameters(mu_p, sigma_p, mu_q, sigma_q, low, high)
    cp, cq = _components(p, sp, lo, hi), _components(q, sq, lo, hi)
    forward = _one_direction(p, sp, q, sq, cp, cq)
    reverse = _one_direction(q, sq, p, sp, cq, cp)
    return {"p_to_q": forward, "q_to_p": reverse, "p_components": cp,
            "q_components": cq, "any_violation": forward["any_violation"] | reverse["any_violation"]}


def _action_tensor(actions, anchor):
    actions = _tensor("actions", actions)
    if actions.device != anchor.device:
        raise TypeError("actions and parameters must share a CUDA device")
    return actions


def _command_log_prob(actions, mu, sigma, low, high):
    a, mu, sigma, low, high = torch.broadcast_tensors(actions, mu, sigma, low, high)
    if bool(((a < low) | (a > high)).any().item()):
        raise NumericalDomainError("clipped_gaussian_log_prob expects commands within bounds")
    left, right = a == low, a == high
    interior_logp = -0.5 * ((a - mu) / sigma).square() - torch.log(sigma) - 0.5 * _LOG_2PI
    logp = torch.where(left, _log_ndtr((low - mu) / sigma),
                       torch.where(right, _log_ndtr((mu - high) / sigma), interior_logp))
    return {"coordinate": logp, "joint": logp.sum(dim=-1),
            "at_lower": left, "at_upper": right, "in_interior": ~(left | right)}


@torch.no_grad()
def clipped_gaussian_log_prob(actions, mu, sigma, low=-1.0, high=1.0):
    """Mixed-measure log probability of executed commands, including endpoint mass."""
    p, sp, _, _, lo, hi = _parameters(mu, sigma, mu, sigma, low, high)
    return _command_log_prob(_action_tensor(actions, p), p, sp, lo, hi)


@torch.no_grad()
def joint_logratio_and_support(raw_actions, mu_behavior, sigma_behavior,
                               mu_target, sigma_target, e_clip=0.1,
                               low=-1.0, high=1.0):
    """Return target/behavior raw and command joint log-ratios and log-space support.

    raw_actions must really be behavior samples for empirical support/ESS claims.
    This function does not reconstruct RNN histories or validate that provenance.
    """
    if not isinstance(e_clip, (int, float)) or not 0.0 < float(e_clip) < 1.0:
        raise ValueError("e_clip must be strictly between 0 and 1")
    b, sb, t, st, lo, hi = _parameters(mu_behavior, sigma_behavior, mu_target, sigma_target, low, high)
    a = _action_tensor(raw_actions, b)
    a, b, sb, t, st, lo, hi = torch.broadcast_tensors(a, b, sb, t, st, lo, hi)
    zb, zt = (a - b) / sb, (a - t) / st
    if bool(((zb.abs() > 1e6) | (zt.abs() > 1e6)).any().item()):
        raise NumericalDomainError("raw action standardized magnitude exceeds 1e6")
    raw_behavior = (-0.5 * zb.square() - torch.log(sb) - 0.5 * _LOG_2PI).sum(dim=-1)
    raw_target = (-0.5 * zt.square() - torch.log(st) - 0.5 * _LOG_2PI).sum(dim=-1)
    command = torch.maximum(torch.minimum(a, hi), lo)
    command_behavior = _command_log_prob(command, b, sb, lo, hi)["joint"]
    command_target = _command_log_prob(command, t, st, lo, hi)["joint"]
    raw_lr, command_lr = raw_target - raw_behavior, command_target - command_behavior
    lower, upper = math.log1p(-float(e_clip)), math.log1p(float(e_clip))
    raw_support = torch.isfinite(raw_lr) & (raw_lr >= lower) & (raw_lr <= upper)
    command_support = torch.isfinite(command_lr) & (command_lr >= lower) & (command_lr <= upper)
    return {
        "raw_logratio": raw_lr, "command_logratio": command_lr,
        "raw_behavior_logprob": raw_behavior, "raw_target_logprob": raw_target,
        "command_behavior_logprob": command_behavior, "command_target_logprob": command_target,
        "raw_support": raw_support, "command_support": command_support,
        "both_supported": raw_support & command_support,
        "raw_only_supported": raw_support & ~command_support,
        "command_only_supported": ~raw_support & command_support,
        "neither_supported": ~raw_support & ~command_support,
        "command": command, "raw_at_exact_boundary": (a == lo) | (a == hi),
        "coordinate_clipped": (a < lo) | (a > hi),
        "all_logratios_finite": torch.isfinite(raw_lr).all() & torch.isfinite(command_lr).all(),
    }


@torch.no_grad()
def logspace_ess(log_weights, dim=-1):
    """Finite-log-weight ESS along dim without exponentiating uncentered weights.

    Returns effective_sample_size, normalized_ess, log_weight_mean and
    log_weight_second_moment. Nonfinite log weights are rejected, never dropped.
    """
    lw = _tensor("log_weights", log_weights)
    dim = int(dim)
    if not -lw.ndim <= dim < lw.ndim:
        raise ValueError("invalid ESS dimension")
    count = lw.shape[dim]
    if count <= 0:
        raise ValueError("ESS requires at least one weight")
    maximum = lw.max(dim=dim, keepdim=True)[0]
    centered = lw - maximum
    first = torch.logsumexp(centered, dim=dim)
    second = torch.logsumexp(2.0 * centered, dim=dim)
    ess = torch.exp(2.0 * first - second)
    shift = maximum.squeeze(dim)
    return {"effective_sample_size": ess, "normalized_ess": ess / float(count),
            "log_weight_mean": shift + first - math.log(count),
            "log_weight_second_moment": 2.0 * shift + second - math.log(count),
            "count": count,
            "range_violation": (ess < 1.0 - ATOL) | (ess > float(count) + ATOL + RTOL * count)}
