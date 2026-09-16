"""Actor-only asymmetric PPO and detached scalar-leaf shadow algebra.

Import is stdlib-only. No RNG, model forward, autograd, optimizer or support
changes occur here. The shadow differentiates an independent logratio leaf;
its coefficients are NOT model gradients or mixed-dtype neglogp gradients.
"""
import math


def _bounds(clip_low, clip_high):
    if not (math.isfinite(clip_low) and math.isfinite(clip_high)
            and 0 <= clip_low < 1 and clip_high >= 0):
        raise ValueError('Finite clip_low in [0,1) and clip_high >= 0 required')
    return 1.0 - clip_low, 1.0 + clip_high


def actor_loss_asymmetric(torch_runtime, old_neglogp, new_neglogp, A,
                          is_ppo, clip_low, clip_high):
    """Native operation sequence, with separate actor lower/upper distances."""
    t = torch_runtime
    low, high = _bounds(clip_low, clip_high)
    if is_ppo:
        ratio = t.exp(old_neglogp - new_neglogp)
        surr1 = A * ratio
        surr2 = A * t.clamp(ratio, low, high)
        loss = t.max(-surr1, -surr2)
    else:
        loss = new_neglogp * A
    return loss


def _branch_arrays(t, ratio, A, clip_low, clip_high):
    """Finite-row dL/dlogratio = ratio * dL/dratio, native rounding order.

    For exact max ties the local Torch1.8.1 contract selects its SECOND input.
    Gradients through clamp include endpoints. Reproduce the cast back to
    ratio.dtype on both multiply branches BEFORE gradient accumulation, then
    the exp backward multiplication. Nonfinite rows are retained but derivative
    interpretation/equality is only qualified on returned finite rows.
    Caller supplies a no_grad context. Shapes must match; no broadcasting.
    """
    low, high = _bounds(clip_low, clip_high)
    if ratio.shape != A.shape:
        raise ValueError('Shadow requires identical ratio and advantage shapes')
    surr1 = A * ratio
    bounded = t.clamp(ratio, low, high)
    surr2 = A * bounded
    raw_loss, clipped_loss = -surr1, -surr2
    loss = t.max(raw_loss, clipped_loss)
    raw_selected = raw_loss > clipped_loss
    clipped_selected = ~raw_selected
    inside = (ratio >= low) & (ratio <= high)
    # Native max -> neg -> multiply -> cast-to-input -> clamp -> add -> exp.
    grad_raw = (-raw_selected.to(surr1.dtype) * A).to(ratio.dtype)
    grad_clip = (-clipped_selected.to(surr2.dtype) * A).to(ratio.dtype)
    grad_clip = t.where(inside, grad_clip, t.zeros_like(grad_clip))
    dlogratio = (grad_raw + grad_clip) * ratio
    finite = (t.isfinite(ratio) & t.isfinite(A) & t.isfinite(surr1)
              & t.isfinite(surr2) & t.isfinite(loss) & t.isfinite(dlogratio))
    return {'loss': loss, 'dlogratio': dlogratio, 'finite': finite,
            'raw_loss': raw_loss, 'clipped_loss': clipped_loss,
            'raw_selected': raw_selected, 'tie': raw_loss == clipped_loss,
            'ratio_inside_bounds': inside}


def actor_shadow(torch_runtime, old_neglogp, new_neglogp, A,
                 native_clip=.1, clip_low=.1, clip_high=.2):
    """PPO-only same-forward counterfactual; all results detached, no backward.

    Native/candidate dlogratio is unweighted and unreduced. Apply the ACTUAL
    detached off/RNN weights and original full-batch denominator externally;
    finite mask is not a license to delete rows from the real objective.
    Native exp remains unsanitized. No clamp of logratio or new support gate.
    """
    t = torch_runtime
    if not (old_neglogp.shape == new_neglogp.shape == A.shape):
        raise ValueError('Shadow requires identical old/new/A shapes')
    if not (old_neglogp.is_cuda and new_neglogp.is_cuda and A.is_cuda):
        raise ValueError('CUDA-only shadow; no CPU algorithm fallback')
    with t.no_grad():
        old, new, advantage = old_neglogp.detach(), new_neglogp.detach(), A.detach()
        logratio = old - new
        ratio = t.exp(logratio)
        native = _branch_arrays(t, ratio, advantage, native_clip, native_clip)
        candidate = _branch_arrays(t, ratio, advantage, clip_low, clip_high)
        finite = (t.isfinite(old) & t.isfinite(new) & t.isfinite(logratio)
                  & native['finite'] & candidate['finite'])
        return {'logratio': logratio, 'ratio': ratio, 'native': native,
                'candidate': candidate, 'finite': finite,
                'coefficient_changed': finite & (native['dlogratio'] != candidate['dlogratio'])}
