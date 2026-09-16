"""Independent CPU mpmath reference, never a fallback for the CUDA API.

Command KL is integrated directly in standardized coordinates, without the
analytic truncated moments used by clipped_gaussian_metrics.py.
"""
import math

import mpmath as mp


CASES = [
    {"name": "interior", "p": (0.0, 0.1), "q": (0.01, 0.11)},
    {"name": "same_upper_tail", "p": (3.0, 1.0), "q": (4.0, 1.0)},
    {"name": "deep_same_upper_tail", "p": (5.0, 1.0), "q": (6.0, 1.0)},
    {"name": "opposite_tails", "p": (5.0, 0.5), "q": (-5.0, 0.5)},
    {"name": "near_self", "p": (0.8, 0.7), "q": (0.8000001, 0.7000001)},
    {"name": "narrow_boundary", "p": (1.0, 0.001), "q": (1.0001, 0.0012)},
    {"name": "broad", "p": (0.0, 100.0), "q": (1.0, 80.0)},
    {"name": "self", "p": (0.7, 0.8), "q": (0.7, 0.8)},
    {"name": "checkpoint_sigma_bounds", "p": (0.0, 0.599849), "q": (0.1, 1.061083)},
    {"name": "lower_boundary", "p": (-1.0, 0.01), "q": (-1.002, 0.02)},
    {"name": "tail_logcdf_branch", "p": (21.0, 1.0), "q": (21.1, 1.0)},
    {"name": "tail_logcdf_far", "p": (50.0, 1.0), "q": (49.0, 1.0)},
    {"name": "very_narrow_interior", "p": (0.0, 0.001), "q": (0.0001, 0.002)},
    {"name": "wide_vs_narrow", "p": (0.0, 100.0), "q": (0.0, 0.001)},
    {"name": "large_mean_opposite", "p": (-100.0, 100.0), "q": (100.0, 100.0)},
]


def _number(value):
    # A float is interpreted as its exact binary value, matching GPU tensor
    # construction, rather than silently comparing a nearby decimal parameter.
    result = mp.mpf(value)
    if not mp.isfinite(result):
        raise ValueError("reference parameters must be finite")
    return result


def _cdf(x):
    return mp.erfc(-x / mp.sqrt(2)) / 2


def _sf(x):
    return mp.erfc(x / mp.sqrt(2)) / 2


def _logcdf(x):
    # For positive x, preserve a tiny negative log-CDF instead of rounding CDF
    # to one before taking log, even when arbitrary-precision exp underflows
    # during a later conversion to ordinary double.
    if x > 0:
        return mp.log1p(-_sf(x))
    return mp.log(_cdf(x))


def _logsf(x):
    return _logcdf(-x)


def _phi(x):
    return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)


def _direction(mu_p, sigma_p, mu_q, sigma_q, low, high):
    alpha, beta = (low - mu_p) / sigma_p, (high - mu_p) / sigma_p
    alpha_q, beta_q = (low - mu_q) / sigma_q, (high - mu_q) / sigma_q
    log_p_l, log_p_r = _logcdf(alpha), _logsf(beta)
    log_q_l, log_q_r = _logcdf(alpha_q), _logsf(beta_q)
    atom = mp.exp(log_p_l) * (log_p_l - log_q_l)
    atom += mp.exp(log_p_r) * (log_p_r - log_q_r)
    delta = mu_p - mu_q
    log_ratio_sigma = mp.log(sigma_q / sigma_p)

    def integrand(x):
        return _phi(x) * (log_ratio_sigma + (delta + sigma_p * x) ** 2
                          / (2 * sigma_q * sigma_q) - x * x / 2)

    cuts = [alpha] + [mp.mpf(x) for x in (-12, -8, -4, 0, 4, 8, 12)
                      if alpha < x < beta] + [beta]
    interior = mp.quad(integrand, cuts)
    latent = log_ratio_sigma + (sigma_p * sigma_p + delta * delta) / (2 * sigma_q * sigma_q) - mp.mpf("0.5")
    return latent, atom + interior


def _encoded(value, digits):
    return {"float": float(value), "mp": mp.nstr(value, digits)}


def reference_metrics(mu_p, sigma_p, mu_q, sigma_q, low=-1.0, high=1.0, dps=80):
    """Return CPU scalar KL reference in both directions, float and mp strings.

    Does not enforce the CUDA module's conservative parameter domain; invalid
    sigma/bounds are rejected. Temporarily sets precision with mp.workdps.
    """
    if int(dps) < 50:
        raise ValueError("reference requires at least 50 decimal digits")
    with mp.workdps(int(dps)):
        p, sp, q, sq, lo, hi = map(_number, (mu_p, sigma_p, mu_q, sigma_q, low, high))
        if sp <= 0 or sq <= 0 or hi <= lo:
            raise ValueError("sigma must be positive and low < high")
        forward = _direction(p, sp, q, sq, lo, hi)
        reverse = _direction(q, sq, p, sp, lo, hi)
        return {
            "p_to_q": {"latent": _encoded(forward[0], int(dps)), "command": _encoded(forward[1], int(dps))},
            "q_to_p": {"latent": _encoded(reverse[0], int(dps)), "command": _encoded(reverse[1], int(dps))},
            "p_log_mass_left": _encoded(_logcdf((lo - p) / sp), int(dps)),
            "p_log_mass_right": _encoded(_logsf((hi - p) / sp), int(dps)),
            "q_log_mass_left": _encoded(_logcdf((lo - q) / sq), int(dps)),
            "q_log_mass_right": _encoded(_logsf((hi - q) / sq), int(dps)),
            "decimal_digits": int(dps),
            "method": "independent_standardized_coordinate_quadrature_plus_erfc_atoms",
        }


def reference_logprob(action, mu, sigma, low=-1.0, high=1.0, dps=80):
    """CPU scalar mixed log probability of a command; outside bounds is invalid."""
    with mp.workdps(int(dps)):
        a, m, s, lo, hi = map(_number, (action, mu, sigma, low, high))
        if s <= 0 or hi <= lo or a < lo or a > hi:
            raise ValueError("invalid sigma, bounds or command outside bounds")
        if a == lo:
            logp, category = _logcdf((lo - m) / s), "lower_atom"
        elif a == hi:
            logp, category = _logsf((hi - m) / s), "upper_atom"
        else:
            logp = -mp.mpf("0.5") * ((a - m) / s) ** 2 - mp.log(s) - mp.log(2 * mp.pi) / 2
            category = "interior_density"
        return {"logprob": _encoded(logp, int(dps)), "category": category}


def reference_ess(log_weights, dps=80):
    """CPU high precision ESS reference using unnormalized log weights."""
    if not log_weights:
        raise ValueError("empty log weights")
    with mp.workdps(int(dps)):
        logs = [_number(v) for v in log_weights]
        maximum = max(logs)
        weights = [mp.exp(v - maximum) for v in logs]
        first, second = mp.fsum(weights), mp.fsum(w * w for w in weights)
        ess = first * first / second
        return {"effective_sample_size": _encoded(ess, int(dps)),
                "normalized_ess": _encoded(ess / len(logs), int(dps))}
