"""Process-local clipping scope and sparse read-only gradient diagnostics.

Torch is injected only after IsaacGym runtime setup. No top-level torch import.
"""
import hashlib
import json
import math
from pathlib import Path
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_scalar': repr(value)}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def sample_reasons(ordinal, relative_epoch, previous_ga_count, current_ga_count):
    reasons = []
    if ordinal <= 8:
        reasons.append('first_eight_attempts')
    if relative_epoch % 64 == 0 and ordinal % 8 == 0:
        reasons.append('every_64th_epoch_last_batch')
    if current_ga_count > previous_ga_count:
        reasons.append('first_batch_after_ga')
    return reasons


def partition_parameters(agent, supplied):
    """Filter model order by current optimizer identity; no tensor comparison."""
    model = list(agent.model.parameters())
    require([id(p) for p in supplied] == [id(p) for p in model], 'Native clipping input is not the complete ordered model parameters')
    flat = [p for group in agent.optimizer.param_groups for p in group['params']]
    owned_ids = {id(p) for p in flat}
    require(len(owned_ids) == len(flat), 'Duplicate optimizer-owned parameter identity')
    require(owned_ids <= {id(p) for p in model}, 'Optimizer owns a parameter outside this model')
    return [p for p in model if id(p) in owned_ids], [p for p in model if id(p) not in owned_ids]


class ClipScopeProbe:
    def __init__(self, agent, plan, output, observations, patches, torch_runtime):
        self.agent, self.plan, self.obs, self.patches, self.t = agent, plan, observations, patches, torch_runtime
        self.stream = (Path(output) / 'norms.jsonl').open('x')
        self.policy = 'optimizer_owned' if plan['arm'] == 'owned_clip' else 'full_model'
        self.unscale_calls = self.clip_calls = self.sample_count = 0
        self.last_unscale_ordinal = self.last_clip_ordinal = self.last_ga_seen = 0
        self.reason_counts = {}
        self.native_clip = torch_runtime.nn.utils.clip_grad_norm_
        self.native_unscale = agent.scaler.unscale_
        self.closed = False

    def install(self):
        def unscale(optimizer, *args, **kwargs):
            require(self.obs.in_actor and optimizer is self.agent.optimizer, 'Unexpected scaler unscale owner/boundary')
            ordinal = self.obs.counts['actor_attempts']
            require(ordinal > self.last_unscale_ordinal, 'Repeated unscale in one actor attempt')
            result = self.native_unscale(optimizer, *args, **kwargs)
            self.unscale_calls += 1
            self.last_unscale_ordinal = ordinal
            return result
        self.patches.set(self.agent.scaler, 'unscale_', unscale)
        self.patches.set(self.t.nn.utils, 'clip_grad_norm_', self.clip)
        return self

    def _norm(self, parameters, fallback_grad):
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return self.t.zeros((), device=fallback_grad.device, dtype=fallback_grad.dtype)
        device = grads[0].device
        # Exact Torch1.8 L2 reduction formula/order, without its gradient mutation.
        return self.t.norm(self.t.stack([self.t.norm(g.detach(), 2.).to(device) for g in grads]), 2.)

    def _coefficient(self, norm, max_norm):
        raw = float((float(max_norm) / (norm + 1e-6)).item())
        # Torch1.8's conditional is false for NaN; distinguish raw from applied.
        return {'raw': raw, 'effective': raw if raw < 1. else 1., 'native_branch_would_multiply': raw < 1.}

    def clip(self, parameters, max_norm, norm_type=2.):
        if not self.obs.in_actor:
            return self.native_clip(parameters, max_norm, norm_type)
        ordinal = self.obs.counts['actor_attempts']
        require(self.last_unscale_ordinal == ordinal and self.last_clip_ordinal < ordinal, 'Clip did not immediately follow this attempt successful native unscale')
        require(float(norm_type) == 2. and float(max_norm) == float(self.agent.grad_norm), 'Native clipping L2/threshold differs')
        supplied = list(parameters)
        owned, excluded = partition_parameters(self.agent, supplied)
        selected = owned if self.policy == 'optimizer_owned' else supplied
        relative = int(self.agent.epoch_num) - (15259 if self.plan['mode'] == 'e51' else 0)
        ga_count = self.obs.counts['ga_merge_returns']
        reasons = sample_reasons(ordinal, relative, self.last_ga_seen, ga_count)
        row = None
        if reasons:
            grads = [p.grad for p in supplied if p.grad is not None]
            require(grads, 'No model gradients at sampled post-unscale boundary')
            require(all(g.is_cuda for g in grads), 'Diagnostic norms require actual CUDA gradients')
            with self.t.no_grad():
                owned_norm = self._norm(owned, grads[0])
                excluded_norm = self._norm(excluded, grads[0])
                all_norm = self._norm(supplied, grads[0])
                original = self._coefficient(all_norm, max_norm)
                counterfactual = self._coefficient(owned_norm, max_norm)
                row = {'boundary': 'after_this_native_unscale_before_this_native_clip',
                       'actor_ordinal': ordinal, 'relative_epoch': relative,
                       'native_epoch': int(self.agent.epoch_num), 'native_frame_before_outer_increment': int(self.agent.frame),
                       'ga_merge_returns': ga_count, 'sampling_reasons': reasons,
                       'scale': float(self.agent.scaler.get_scale()), 'clip_threshold': float(max_norm),
                       'actual_clip_policy': self.policy,
                       'owned_norm': float(owned_norm.item()),
                       'excluded_remaining_scaled_norm': float(excluded_norm.item()),
                       'all_parameter_norm': float(all_norm.item()),
                       'original_full_model_coefficient': original,
                       'counterfactual_optimizer_owned_coefficient': counterfactual,
                       'owned_parameter_count': len(owned), 'excluded_parameter_count': len(excluded),
                       'owned_with_grad_count': sum(p.grad is not None for p in owned),
                       'excluded_with_grad_count': sum(p.grad is not None for p in excluded),
                       'excluded_gradient_scope': 'Current remaining buffer; unscale did not visit excluded parameters. May accumulate prior scales/clipping; not assumed to equal this attempt gradient times current scale.',
                       'forward_backward_added': 0, 'probe_mutates_gradients': False}
        # BEFORE: native_clip(all model parameters, max_norm, norm_type).
        # AFTER owned_clip only: same native function, model-order optimizer-owned subset.
        result = self.native_clip(selected, max_norm, norm_type)
        self.clip_calls += 1
        self.last_clip_ordinal = ordinal
        self.last_ga_seen = ga_count
        if row is not None:
            row['native_clip_returned_norm'] = float(result.detach().item())
            row['monotonic'] = time.monotonic()
            self.stream.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + '\n')
            self.stream.flush()
            self.sample_count += 1
            for reason in reasons:
                self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        return result

    def summary(self):
        return {'policy': self.policy, 'unscale_returned': self.unscale_calls,
                'clip_returned': self.clip_calls, 'sample_count': self.sample_count,
                'sampling': 'first8 attempts OR each64th relative epoch batch8 OR first batch after GA return; union, no duplicate row',
                'sampling_reason_counts': dict(self.reason_counts), 'journal_closed': self.stream.closed,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'Same-pass sparse CUDA norm telemetry; no extra forward/backward, no additional zeroing, no RMS/scaler/GA/LF change'}

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
        self.closed = True


def install(agent, plan, output, observations, patches, torch_runtime):
    require(plan['arm'] in ('control', 'clip_higher', 'owned_clip'), 'Unsupported structural arm')
    handle = ClipScopeProbe(agent, plan, output, observations, patches, torch_runtime)
    try:
        return handle.install()
    except BaseException:
        try:
            handle.close()
        except BaseException:
            pass  # Preserve the installation's original exception.
        raise
