"""One native epoch of scalar CUDA observations; import is stdlib-only.

No model forward/backward, RNG-state access, CPU algorithm tensor copy or data
mutation is added. CUDA diagnostic allocations/reductions and scalar transfers
have cost; this is not an established bitwise non-interference proof.
"""
import json
import math
from pathlib import Path
import sys
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_scalar': str(value)}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def critic_arrays(t, old, value, target, clip, clip_value=True):
    """Unweighted per-row critic loss and piecewise value-leaf derivative.

    Version-bound max tie rule is SECOND branch, pending the separate actual
    CUDA/autograd fixture. Finite inputs/intermediates only receive derivative
    comparisons. Derivative is with respect to the original value dtype, not
    the network parameters or scaled/weighted total loss.
    """
    raw = (value - target).square()
    if not clip_value:
        finite = t.isfinite(old) & t.isfinite(value) & t.isfinite(target) & t.isfinite(raw)
        derivative = (2.0 * (value - target)).to(value.dtype)
        return {'raw': raw, 'clipped': raw, 'loss': raw,
                'raw_selected': t.ones_like(raw, dtype=t.bool),
                'tie': t.zeros_like(raw, dtype=t.bool), 'finite': finite,
                'derivative': derivative}
    delta = value - old
    bounded = old + t.clamp(delta, -clip, clip)
    clipped = (bounded - target).square()
    raw_selected = raw > clipped
    # In PyTorch 1.8.1 the derivative of clamp includes both endpoints.
    interior = (delta >= -clip) & (delta <= clip)
    raw_derivative = (2.0 * (value - target)).to(value.dtype)
    clipped_derivative = t.where(interior, 2.0 * (bounded - target),
                                 t.zeros_like(bounded)).to(value.dtype)
    derivative = t.where(raw_selected, raw_derivative, clipped_derivative)
    finite = (t.isfinite(old) & t.isfinite(value) & t.isfinite(target)
              & t.isfinite(raw) & t.isfinite(clipped) & t.isfinite(derivative))
    return {'raw': raw, 'clipped': clipped, 'loss': t.max(raw, clipped),
            'raw_selected': raw_selected, 'tie': raw == clipped,
            'finite': finite, 'derivative': derivative}


class ChainProbe:
    def __init__(self, agent, output_dir, expected_epochs=1, torch_runtime=None):
        require(expected_epochs == 1, 'Probe scope is exactly one native epoch')
        self.t = torch_runtime if torch_runtime is not None else sys.modules.get('torch')
        require(self.t is not None, 'Install only after the native CUDA runtime exists')
        self.agent = agent
        require(agent.is_rnn and not agent.multi_gpu and not agent.has_central_value,
                'Only the fixed single-GPU recurrent shared-critic path is supported')
        require((agent.num_actors, agent.horizon_length, agent.seq_length,
                 agent.mini_epochs_num, agent.minibatch_size, agent.num_agents,
                 agent.value_size) == (20480, 16, 16, 2, 81920, 1, 1),
                'Unexpected native layout')
        require(next(agent.model.parameters()).is_cuda, 'CUDA model required')
        require(agent.normalize_value and agent.has_value_loss and agent.schedule_type == 'standard',
                'Normalized critic and standard scheduler required')
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / 'stages.jsonl').open('x', encoding='utf-8')
        self.patches = []
        self.active = False
        self.in_play = False
        self.in_actor = False
        self.in_prepare = False
        self.finalized = None
        self.failure = None
        self.counts = {k: 0 for k in (
            'epoch_entries', 'epoch_returns', 'play_returns', 'env_step_returns',
            'actor_attempts', 'actor_returns', 'critic_calls', 'model_forwards',
            'actor_model_forwards', 'truncate_calls', 'adam_entered', 'adam_returned',
            'adam_exceptions', 'scheduler_calls', 'scheduler_returns',
            'lr_applications', 'prepare_calls', 'value_rms_calls', 'gae_calls',
            'cache_writes', 'after_steps_calls', 'after_steps_returns', 'gene_merge_calls', 'gene_merge_returns')}
        self.success_steps, self.done_steps = [], []
        self.rollout_raw_a = None
        self.group_cache = None
        self.cf_old = None
        self.current_batch = None
        self.current_cf_old = None
        self.current_model = None
        self.stage = 'install'
        try:
            self._install()
        except BaseException as exc:
            self._fail(exc, 'partial_install')
            try:
                self.uninstall()
            except BaseException:
                pass
            try:
                self.stream.close()
            except BaseException:
                pass
            raise

    def _patch(self, owner, name, value):
        missing = object()
        prior = owner.__dict__.get(name, missing)
        original = getattr(owner, name)
        self.patches.append((owner, name, prior, missing))
        setattr(owner, name, value)
        return original

    def _clock(self):
        return {'epoch': int(self.agent.epoch_num), 'frame': int(self.agent.frame),
                'actor_ordinal': self.counts['actor_attempts']}

    def _emit(self, kind, **fields):
        row = dict(kind=kind, host_monotonic=time.monotonic(), **self._clock(), **fields)
        self.stream.write(json.dumps(clean(row), allow_nan=False, separators=(',', ':')) + '\n')
        self.stream.flush()

    def _fail(self, exc, where):
        if self.failure is None:
            self.failure = {'where': where, 'error': repr(exc), **self._clock(),
                            'counts': dict(self.counts)}
        try:
            self._emit('failure', first_failure=self.failure)
        except BaseException:
            pass

    def _stats(self, value):
        t = self.t
        require(isinstance(value, t.Tensor) and value.is_cuda, 'Algorithm tensor must be CUDA')
        x = value.detach().reshape(-1)
        finite = t.isfinite(x)
        good = x[finite].float()
        result = {'n': x.numel(), 'dtype': str(x.dtype), 'finite': int(finite.sum().item()),
                  'zero': int((x == 0).sum().item())}
        if good.numel():
            result.update(mean=float(good.mean().item()), std=float(good.std(unbiased=False).item()),
                          minimum=float(good.min().item()), maximum=float(good.max().item()),
                          max_abs=float(good.abs().max().item()))
        return result

    def _rms(self):
        rms = self.agent.value_mean_std
        return {'count': float(rms.count.item()), 'mean': float(rms.running_mean.item()),
                'variance': float(rms.running_var.item()), 'training': bool(rms.training),
                'epsilon': float(rms.epsilon), 'norm_only': bool(rms.norm_only)}

    def _encode_final_rms(self, raw):
        rms = self.agent.value_mean_std
        if rms.norm_only:
            return raw / self.t.sqrt(rms.running_var.float() + rms.epsilon)
        return ((raw - rms.running_mean.float()) /
                self.t.sqrt(rms.running_var.float() + rms.epsilon)).clamp(-5.0, 5.0)

    def _norms(self):
        t = self.t
        owned = {id(p) for group in self.agent.optimizer.param_groups for p in group['params']}
        result = {}
        for label, want_owned in [('owned', True), ('unowned', False)]:
            params = [p for p in self.agent.model.parameters()
                      if (id(p) in owned) == want_owned and p.grad is not None]
            result[label] = {'grad_tensors': len(params), 'grad_elements': sum(p.grad.numel() for p in params)}
            if params:
                norms = t.stack([t.norm(p.grad.detach().float(), p=2) for p in params])
                result[label].update(norm=float(t.norm(norms, p=2).item()),
                                     nonfinite_tensor_norms=int((~t.isfinite(norms)).sum().item()))
        return result

    def _group_counts(self, normalized_on=None):
        t = self.t
        if self.rollout_raw_a is None or len(self.success_steps) != 16:
            return {'coverage': 'unavailable'}
        # These are time-window and slot groups, never GRPO prompt replicates.
        success = t.stack(self.success_steps, dim=1)  # N,H
        dones = t.stack(self.done_steps, dim=1)
        raw = self.rollout_raw_a.reshape(20480, 16)
        same_seq = success.max(dim=1).values == success.min(dim=1).values
        no_done = ~dones.any(dim=1)
        result = {'coverage': 'all_native_rollout_rows', 'sequence_groups': 20480,
                  'success_homogeneous_seq': int(same_seq.sum().item()),
                  'success_homogeneous_seq_without_done': int((same_seq & no_done).sum().item()),
                  'homogeneous_seq_with_nonzero_raw_A': int((same_seq & (raw != 0).any(dim=1)).sum().item()),
                  'homogeneous_seq_without_done_with_nonzero_raw_A': int((same_seq & no_done & (raw != 0).any(dim=1)).sum().item())}
        if self.agent.expl_type.startswith('mixed_expl'):
            same_slot = (success[:, -1].reshape(64, 320).max(dim=1).values ==
                         success[:, -1].reshape(64, 320).min(dim=1).values)
            result.update(slot_groups=64, end_success_homogeneous_slots=int(same_slot.sum().item()),
                          homogeneous_slots_with_nonzero_raw_A=int((same_slot & (raw.reshape(64, -1) != 0).any(dim=1)).sum().item()))
        if normalized_on is not None:
            normalized = normalized_on.reshape(20480, 16)
            result['homogeneous_seq_with_nonzero_reencoded_normalized_A'] = int((same_seq & (normalized != 0).any(dim=1)).sum().item())
            result['normalized_scope'] = 'native on raw A reencoded using actual full augmented raw-A mean/std; no group-relative normalization'
        return result

    def _install(self):
        t, agent = self.t, self.agent
        common_losses = sys.modules.get('rl_games.common.common_losses')
        require(common_losses is not None, 'Native common_losses must already be imported')
        old_epoch = agent.train_epoch
        old_play = agent.play_steps
        old_env = agent.env_step
        old_gae = agent.discount_values
        old_prepare = agent.prepare_dataset
        old_model = agent.model.forward
        old_actor = agent.train_actor_critic
        old_critic = common_losses.critic_loss
        old_truncate = agent.trancate_gradients_and_step
        old_scheduler = agent.scheduler.update
        old_lr = agent.update_lr
        old_cache = agent.dataset.update_mu_sigma
        old_after_steps = agent.algo_observer.after_steps
        network = agent.model.a2c_network
        old_merge = getattr(network, 'merge_extra_params', None)

        def epoch(*args, **kwargs):
            require(self.counts['epoch_entries'] == 0, 'Probe may observe exactly one epoch')
            self.active = True
            self.counts['epoch_entries'] += 1
            try:
                self._emit('epoch_start', clip_value=bool(agent.clip_value),
                           normalize_value=bool(agent.normalize_value),
                           normalize_advantage=bool(agent.normalize_advantage),
                           value_rms=self._rms(), scaler_scale=float(agent.scaler.get_scale()),
                           off_actor_coef=float(agent.off_policy_actor_coef),
                           off_critic_coef=float(agent.off_policy_critic_coef),
                           support_mode=agent.off_policy_support_mode,
                           method_expl_type=agent.expl_type, lr=float(agent.last_lr))
                result = old_epoch(*args, **kwargs)
                self.counts['epoch_returns'] += 1
                self._emit('epoch_return', observed=dict(self.counts))
                return result
            except BaseException as exc:
                self._fail(exc, 'train_epoch')
                raise
            finally:
                self.active = False
                self.rollout_raw_a = self.cf_old = self.current_cf_old = None
                self.success_steps.clear()
                self.done_steps.clear()

        def env_step(*args, **kwargs):
            result = old_env(*args, **kwargs)
            if self.active and self.in_play:
                self.counts['env_step_returns'] += 1
                obs, reward, intrinsic, done, infos = result
                with t.no_grad():
                    successes = infos.get('current_episode_successes')
                    if successes is not None:
                        self.success_steps.append(successes.detach().clone())
                        self.done_steps.append(done.detach().bool().clone())
                    fields = {'step': self.counts['env_step_returns'], 'reward': self._stats(reward),
                              'done_count': int(done.bool().sum().item()),
                              'current_success_field_available': successes is not None}
                    if 'time_outs' in infos:
                        fields['timeout_count'] = int(infos['time_outs'].bool().sum().item())
                    if successes is not None:
                        fields['current_successes'] = self._stats(successes)
                    self._emit('env_return', **fields)
            return result

        def gae(fdones, last_values, mb_dones, mb_values, mb_rewards):
            result = old_gae(fdones, last_values, mb_dones, mb_values, mb_rewards)
            if self.active and self.in_play:
                self.counts['gae_calls'] += 1
                with t.no_grad():
                    next_values = t.cat((mb_values[1:], last_values.unsqueeze(0)), dim=0)
                    next_dones = t.cat((mb_dones[1:], fdones.unsqueeze(0)), dim=0)
                    delta = mb_rewards + agent.gamma * next_values * (1.0 - next_dones).unsqueeze(-1) - mb_values
                    self._emit('gae', shaped_rewards=self._stats(mb_rewards),
                               sampled_values=self._stats(mb_values), bootstrap_values=self._stats(last_values),
                               td_delta=self._stats(delta), gae=self._stats(result),
                               next_done_count=int(next_dones.bool().sum().item()))
            return result

        def play(*args, **kwargs):
            self.in_play = self.active
            try:
                result = old_play(*args, **kwargs)
                if self.active:
                    self.counts['play_returns'] += 1
                    batch, extras = result
                    with t.no_grad():
                        self.rollout_raw_a = (batch['returns'] - batch['values']).sum(dim=1)
                        self._emit('rollout_return', rows=len(batch['returns']),
                                   raw_advantage=self._stats(self.rollout_raw_a),
                                   rewards=self._stats(extras['rewards']), groups=self._group_counts())
                return result
            finally:
                self.in_play = False

        def prepare(batch, *args, **kwargs):
            if not self.active:
                return old_prepare(batch, *args, **kwargs)
            self.counts['prepare_calls'] += 1
            self.in_prepare = True
            rms = agent.value_mean_std
            old_rms_forward = rms.forward
            missing = object()
            prior = rms.__dict__.get('forward', missing)
            self._emit('prepare_before', value_rms=self._rms(), rows=len(batch['returns']))
            def rms_forward(*rargs, **rkwargs):
                output = old_rms_forward(*rargs, **rkwargs)
                self.counts['value_rms_calls'] += 1
                self._emit('value_rms_return', call=self.counts['value_rms_calls'],
                           state=self._rms(), output=self._stats(output))
                return output
            rms.forward = rms_forward
            try:
                result = old_prepare(batch, *args, **kwargs)
            finally:
                if prior is missing:
                    del rms.forward
                else:
                    rms.forward = prior
                self.in_prepare = False
            with t.no_grad():
                data = agent.dataset.values_dict
                raw = (batch['returns'] - batch['values']).sum(dim=1)
                self.cf_old = self._encode_final_rms(batch['values']) if agent.normalize_value else batch['values'].detach()
                fields = {'value_rms': self._rms(), 'raw_A': self._stats(raw),
                          'normalized_A': self._stats(data['advantages']),
                          'native_old_values': self._stats(data['old_values']),
                          'final_rms_old_values': self._stats(self.cf_old),
                          'old_anchor_delta': self._stats(self.cf_old - data['old_values']),
                          'returns': self._stats(data['returns'])}
                if agent.normalize_advantage and not agent.normalize_rms_advantage and data.get('rnn_masks') is None:
                    mean, std = raw.mean(), raw.std()
                    encoded = (raw - mean) / (std + 1e-8)
                    normalized_on = (self.rollout_raw_a - mean) / (std + 1e-8)
                    fields['normalization_reencoding_delta'] = self._stats(encoded - data['advantages'])
                    fields['groups'] = self._group_counts(normalized_on)
                else:
                    fields['groups'] = {'normalized_scope': 'not supported by this narrow reencoding diagnostic'}
                self._emit('prepare_return', **fields)
            return result

        def model(*args, **kwargs):
            result = old_model(*args, **kwargs)
            if self.active:
                self.counts['model_forwards'] += 1
                if self.in_actor:
                    self.counts['actor_model_forwards'] += 1
                    self.current_model = {key: result[key].detach() for key in ('prev_neglogp', 'values', 'entropy', 'mus', 'sigmas')}
            return result

        def critic(model_arg, old_values, values, clip, returns, clip_value):
            result = old_critic(model_arg, old_values, values, clip, returns, clip_value)
            if self.active and self.in_actor:
                self.counts['critic_calls'] += 1
                with t.no_grad():
                    native = critic_arrays(t, old_values.detach(), values.detach(), returns.detach(), clip, clip_value)
                    candidate = critic_arrays(t, self.current_cf_old, values.detach(), returns.detach(), clip, clip_value)
                    finite = native['finite'] & candidate['finite']
                    off = self.current_batch.get('off_policy_mask')
                    masks = {'all': t.ones_like(finite, dtype=t.bool)}
                    if off is not None:
                        masks.update(on=~off.reshape_as(finite).bool(), off=off.reshape_as(finite).bool())
                    fields = {'clip_value': bool(clip_value), 'clip': float(clip),
                              'native_loss_rebuild_delta': self._stats(native['loss'] - result.detach()),
                              'native_loss': self._stats(result), 'candidate_loss': self._stats(candidate['loss']),
                              'values': self._stats(values), 'anchor_delta': self._stats(self.current_cf_old - old_values),
                              'derivative_qualification': 'version-specific analytic values leaf; requires separate actual CUDA/autograd fixture; not network gradient'}
                    for label, mask in masks.items():
                        valid = finite & mask
                        fields[label] = {'rows': int(mask.sum().item()), 'finite_comparable': int(valid.sum().item()),
                            'native_ties': int((native['tie'] & mask).sum().item()),
                            'native_raw_branch': int((native['raw_selected'] & mask).sum().item()),
                            'candidate_raw_branch': int((candidate['raw_selected'] & mask).sum().item()),
                            'branch_changed': int(((native['raw_selected'] != candidate['raw_selected']) & valid).sum().item()),
                            'native_leaf_nonzero': int(((native['derivative'] != 0) & valid).sum().item()),
                            'candidate_leaf_nonzero': int(((candidate['derivative'] != 0) & valid).sum().item()),
                            'leaf_changed': int(((native['derivative'] != candidate['derivative']) & valid).sum().item()),
                            'leaf_delta': self._stats((candidate['derivative'] - native['derivative'])[valid])}
                    self._emit('critic', **fields)
            return result

        def actor(batch, *args, **kwargs):
            if not self.active:
                return old_actor(batch, *args, **kwargs)
            require(self.counts['actor_attempts'] < 8, 'Unexpected ninth actor attempt')
            self.counts['actor_attempts'] += 1
            self.in_actor = True
            self.current_batch = batch
            start, end = agent.dataset.last_range
            self.current_cf_old = self.cf_old[start:end]
            require(len(self.current_cf_old) == len(batch['old_values']), 'Counterfactual anchor range mismatch')
            self.current_model = None
            try:
                result = old_actor(batch, *args, **kwargs)
                self.counts['actor_returns'] += 1
                with t.no_grad():
                    require(self.current_model is not None, 'No native training model output observed')
                    model_output = self.current_model
                    logratio = batch['old_logp_actions'] - model_output['prev_neglogp']
                    ratio = t.exp(logratio)
                    safe_ratio = t.exp(t.nan_to_num(logratio, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0))
                    advantage = batch['advantages']
                    finite = t.isfinite(ratio) & t.isfinite(advantage)
                    ordinary_zero = ((advantage == 0) | ((advantage > 0) & (ratio > 1 + agent.e_clip)) |
                                     ((advantage < 0) & (ratio < 1 - agent.e_clip)))
                    # Ratio categories are descriptive; FP32 surrogate ties can
                    # require the exact native max derivative, so not a leaf proof.
                    off = batch.get('off_policy_mask')
                    masks = {'all': t.ones_like(ratio, dtype=t.bool)}
                    if off is not None:
                        masks.update(on=~off.reshape(-1).bool(), off=off.reshape(-1).bool())
                    categories = {}
                    support = (safe_ratio >= 1-agent.e_clip) & (safe_ratio <= 1+agent.e_clip)
                    for label, mask in masks.items():
                        categories[label] = {'rows': int(mask.sum().item()),
                            'finite_ratio_and_A': int((mask & finite).sum().item()),
                            'A_zero': int((mask & (advantage == 0)).sum().item()),
                            'ratio_based_zero_region': int((mask & finite & ordinary_zero).sum().item()),
                            'safe_support': int((mask & support).sum().item()),
                            'outside_support_nonzero_ratio_region': int((mask & finite & ~support & ~ordinary_zero).sum().item())}
                    self._emit('actor_return', rows=len(ratio), range=[int(start), int(end)],
                               mini=(self.counts['actor_attempts']-1)//4,
                               batch=(self.counts['actor_attempts']-1)%4,
                               logratio=self._stats(logratio), raw_ratio=self._stats(ratio),
                               A=self._stats(advantage), entropy=self._stats(model_output['entropy']),
                               mu=self._stats(model_output['mus']), sigma=self._stats(model_output['sigmas']),
                               actual_support_mode=agent.off_policy_support_mode,
                               regions=categories, original_actor_loss=self._stats(result[0]),
                               original_critic_loss=self._stats(result[1]), original_bounds_loss=self._stats(result[8]))
                return result
            except BaseException as exc:
                self._fail(exc, 'actor')
                raise
            finally:
                self.in_actor = False
                self.current_batch = self.current_cf_old = self.current_model = None

        def truncate(*args, **kwargs):
            if not (self.active and self.in_actor):
                return old_truncate(*args, **kwargs)
            self.counts['truncate_calls'] += 1
            optimizer = agent.optimizer
            original_step = optimizer.step
            missing = object()
            prior = optimizer.__dict__.get('step', missing)
            def actual_step(*sargs, **skwargs):
                self.counts['adam_entered'] += 1
                try:
                    result = original_step(*sargs, **skwargs)
                except BaseException:
                    self.counts['adam_exceptions'] += 1
                    raise
                self.counts['adam_returned'] += 1
                return result
            with t.no_grad():
                self._emit('pre_native_truncate', scaled_grad_norms=self._norms(),
                           scaler_scale=float(agent.scaler.get_scale()))
            optimizer.step = actual_step
            error = None
            try:
                result = old_truncate(*args, **kwargs)
            except BaseException as exc:
                error = exc
                self._fail(exc, 'native_truncate')
                raise
            finally:
                if prior is missing:
                    del optimizer.step
                else:
                    optimizer.step = prior
                try:
                    with t.no_grad():
                        self._emit('post_native_truncate', grad_norms=self._norms(),
                                   scaler_scale=float(agent.scaler.get_scale()),
                                   adam_entered=self.counts['adam_entered'], adam_returned=self.counts['adam_returned'],
                                   original_error=None if error is None else repr(error))
                except BaseException as secondary:
                    self._fail(secondary, 'post_truncate_observation')
                    if error is None:
                        raise
            return result

        def scheduler(*args, **kwargs):
            if self.active:
                self.counts['scheduler_calls'] += 1
            result = old_scheduler(*args, **kwargs)
            if self.active:
                self.counts['scheduler_returns'] += 1
                self._emit('scheduler', original_args=args, original_kwargs=kwargs, original_result=result)
            return result

        def lr(*args, **kwargs):
            result = old_lr(*args, **kwargs)
            if self.active:
                self.counts['lr_applications'] += 1
                self._emit('lr_application', original_args=args, original_kwargs=kwargs)
            return result

        def cache(*args, **kwargs):
            result = old_cache(*args, **kwargs)
            if self.active:
                self.counts['cache_writes'] += 1
                self._emit('cache_write', range=list(agent.dataset.last_range))
            return result

        def after_steps(*args, **kwargs):
            if self.active:
                self.counts['after_steps_calls'] += 1
                self._emit('after_steps_enter', timing='after prepare before actor; eligibility is not actual GA')
            result = old_after_steps(*args, **kwargs)
            if self.active:
                self.counts['after_steps_returns'] += 1
                self._emit('after_steps_return', gene_merge_returns=self.counts['gene_merge_returns'])
            return result

        def merge(*args, **kwargs):
            if self.active:
                self.counts['gene_merge_calls'] += 1
            result = old_merge(*args, **kwargs)
            if self.active:
                self.counts['gene_merge_returns'] += 1
                self._emit('gene_merge_return', scope='native merge call returned; no gene tensor delta inferred')
            return result

        for owner, name, fn in [(agent, 'train_epoch', epoch), (agent, 'play_steps', play),
                (agent, 'env_step', env_step), (agent, 'discount_values', gae),
                (agent, 'prepare_dataset', prepare), (agent.model, 'forward', model),
                (common_losses, 'critic_loss', critic), (agent, 'train_actor_critic', actor),
                (agent, 'trancate_gradients_and_step', truncate),
                (agent.scheduler, 'update', scheduler), (agent, 'update_lr', lr),
                (agent.dataset, 'update_mu_sigma', cache), (agent.algo_observer, 'after_steps', after_steps)]:
            self._patch(owner, name, fn)
        if old_merge is not None:
            self._patch(network, 'merge_extra_params', merge)

    def finalize(self, require_complete=True):
        if self.finalized is not None:
            return self.finalized
        expected = {'epoch_entries': 1, 'epoch_returns': 1, 'play_returns': 1,
                    'env_step_returns': 16, 'actor_attempts': 8, 'actor_returns': 8,
                    'actor_model_forwards': 8, 'critic_calls': 8, 'truncate_calls': 8,
                    'scheduler_calls': 2, 'scheduler_returns': 2, 'lr_applications': 2,
                    'prepare_calls': 1, 'value_rms_calls': 2, 'gae_calls': 1,
                    'cache_writes': 8, 'after_steps_calls': 1, 'after_steps_returns': 1}
        failures = {k: {'actual': self.counts[k], 'expected': v}
                    for k, v in expected.items() if self.counts[k] != v}
        completed = not failures and self.failure is None
        summary = {'status': 'completed' if completed else 'incomplete', 'observed': dict(self.counts),
                   'expected': expected, 'quota_failures': failures, 'first_failure': self.failure,
                   'require_complete_requested': bool(require_complete), **self._clock(),
                   'physics_frames_from_successful_env_returns': self.counts['env_step_returns'] * 20480,
                   'extra_model_forwards': 0, 'extra_model_backwards': 0,
                   'rng_state_read_or_write_calls_added': 0,
                   'counterfactual_scope': 'S2 old-anchor scalar critic leaf only; no changed training',
                   'group_scope': 'correlated seq16/heterogeneous slot groups; not DAPO prompt replicas',
                   'derivative_fixture_required': True,
                   'coverage_limits': ['No shared parameter gradient contributions or Adam delta measured',
                       'No new post-step policy KL or target-history forward',
                       'GA and episode/timeout events may not occur in this fixed horizon',
                       'Read-only numerical diagnostics have allocation/synchronization cost; no bitwise equivalence claim']}
        self.stream.flush()
        self.stream.close()
        (self.output / 'summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+'\n')
        self.finalized = summary
        if require_complete and not completed:
            raise RuntimeError('Incomplete native chain probe: '+repr(failures or self.failure))
        return summary

    def uninstall(self):
        first = None
        while self.patches:
            owner, name, prior, missing = self.patches.pop()
            try:
                if prior is missing:
                    delattr(owner, name)
                else:
                    setattr(owner, name, prior)
            except BaseException as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first


def install_chain_probe(agent, output_dir, expected_epochs=1, torch_runtime=None):
    return ChainProbe(agent, output_dir, expected_epochs, torch_runtime)
