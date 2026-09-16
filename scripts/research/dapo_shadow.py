"""Read-only actor-loss counterfactuals on one original EPO training epoch.

CUDA provenance tags travel through the native LF augmentation and recurrent
shuffle. They are ignored by the model/loss. Only scalar counts and sums are
written; no new model forward, backward, sampling, or parameter update is added.
"""
import json
from pathlib import Path

from scripts.research.run_training_probe import require, write, secondary_report
from scripts.research.dapo_actor import actor_shadow


class DapoShadow:
    def __init__(self, agent, output):
        import torch
        from rl_games.algos_torch.a2c_continuous import off_policy_training_weights
        self.t, self.agent = torch, agent
        self.weights = off_policy_training_weights
        require(agent.is_rnn and agent.seq_length == agent.horizon_length == 16,
                'Fixed recurrent horizon required')
        require(agent.num_actors == 20480 and agent.intr_coef_block_size == 320,
                'Fixed condition-slot assignment required')
        require(len(agent.vec_env.env.object_asset_files) == 947, 'Object catalogue differs')
        require(agent.e_clip == .1 and agent.off_policy_support_mode == 'actor_clip',
                'Original lower/upper/support must stay .1')
        require(agent.off_policy_actor_coef == 1 and agent.off_policy_critic_coef == 0,
                'Original LF coefficients required')
        self.output = Path(output)
        self.output.mkdir()
        self.stream = (self.output / 'actor.jsonl').open('x')
        self.patches, self.rows = [], []
        self.batch = None
        self.attempts = self.returns = 0
        self.finalized = None
        try:
            self._install()
        except BaseException as primary:
            errors = []
            try:
                self.uninstall()
            except BaseException as exc:
                errors.append(repr(exc))
            try:
                self.stream.close()
            except BaseException as exc:
                errors.append(repr(exc))
            if errors:
                secondary_report(self.output / 'install-cleanup-errors.json',
                                 {'primary': repr(primary), 'secondary': errors})
            raise

    def patch(self, owner, name, fn):
        missing = object()
        prior = owner.__dict__.get(name, missing)
        original = getattr(owner, name)
        self.patches.append((owner, name, prior, missing))
        setattr(owner, name, fn)
        return original

    def histogram(self, ids, mask, count):
        result = self.t.bincount(ids[mask], minlength=count)
        return [int(value.item()) for value in result.unbind()]

    def _install(self):
        t, agent = self.t, self.agent

        def augment(batch, extras, *args, **kwargs):
            require('research_env_id' not in batch, 'Provenance already present')
            batch['research_env_id'] = t.arange(agent.num_actors, device=agent.ppo_device,
                                                dtype=t.long).repeat_interleave(agent.horizon_length)
            return old_augment(batch, extras, *args, **kwargs)

        def prepare(batch, *args, **kwargs):
            result = old_prepare(batch, *args, **kwargs)
            tag = batch['research_env_id']
            require(tag.is_cuda and tag.dtype == t.long and tag.shape == batch['neglogpacs'].shape,
                    'Provenance layout differs')
            seq = tag.reshape(-1, agent.seq_length)
            require(bool((seq == seq[:, :1]).all().item()), 'Native shuffle broke sequence identity')
            agent.dataset.values_dict['research_env_id'] = tag
            return result

        def actor(batch, *args, **kwargs):
            require(self.batch is None and self.attempts < 8, 'Unexpected actor boundary')
            require(batch['rnn_masks'] is None, 'This probe requires native unmasked rows')
            self.attempts += 1
            self.batch = batch
            try:
                result = old_actor(batch, *args, **kwargs)
                self.returns += 1
                return result
            finally:
                self.batch = None

        def loss(old, new, advantage, is_ppo, clip):
            result = old_loss(old, new, advantage, is_ppo, clip)
            require(self.batch is not None and is_ppo and clip == .1, 'Wrong native actor loss')
            with t.no_grad():
                a = advantage.detach()
                terms = actor_shadow(t, old.detach(), new.detach(), a,
                                     native_clip=.1, clip_low=.1, clip_high=.2)
                native, candidate = terms['native'], terms['candidate']
                require(t.equal(native['loss'], result.detach()), 'Native loss rebuild is not exact')
                off = self.batch['off_policy_mask'].reshape(-1).bool()
                weight = self.weights(old.detach(), new.detach(), off, clip,
                    actor_coef=agent.off_policy_actor_coef, critic_coef=agent.off_policy_critic_coef,
                    support_mode=agent.off_policy_support_mode)['actor']
                finite = terms['finite'] & t.isfinite(weight)
                native_leaf = native['dlogratio'] * weight
                candidate_leaf = candidate['dlogratio'] * weight
                changed = finite & (native_leaf != candidate_leaf)
                recovered = finite & (native_leaf == 0) & (candidate_leaf != 0)
                lost = finite & (native_leaf != 0) & (candidate_leaf == 0)
                source_env = self.batch['research_env_id']
                source_gene = source_env // 320
                target_gene = t.where(off, t.full_like(source_gene, 63), source_gene)
                d = agent.intr_reward_coef_embd.shape[1]
                expected = agent.intr_reward_coef_embd[target_gene * 320]
                require(t.equal(self.batch['obs'][:, -d:], expected), 'Target condition tag is wrong')
                ratio = terms['ratio']
                row = {'attempt': self.attempts, 'frame': int(agent.frame), 'epoch': int(agent.epoch_num),
                       'rows': len(ratio), 'native_loss_rebuild_exact': True,
                       'finite_comparable': int(finite.sum().item()),
                       'native_lower': .1, 'native_upper': .1, 'candidate_upper': .2,
                       'scope': 'Weighted d(actor row loss)/d logratio; not shared parameter gradient',
                       'native_full_batch_denominator': len(ratio),
                       'target_condition_tag_verified': True}
                for label, mask in [('on', ~off), ('off', off)]:
                    valid = finite & mask
                    rec = recovered & mask
                    row[label] = {
                        'rows': int(mask.sum().item()), 'finite': int(valid.sum().item()),
                        'positive_A': int((valid & (a > 0)).sum().item()),
                        'negative_A': int((valid & (a < 0)).sum().item()),
                        'positive_A_ratio_1p1_to_1p2': int((valid & (a > 0) & (ratio > 1.1) & (ratio <= 1.2)).sum().item()),
                        'changed_leaf': int((changed & mask).sum().item()),
                        'recovered_leaf': int(rec.sum().item()),
                        'positive_A_recovered_leaf': int((rec & (a > 0)).sum().item()),
                        'lost_leaf': int((lost & mask).sum().item()),
                        'native_leaf_abs_sum': float(native_leaf[valid].abs().sum().item()),
                        'candidate_leaf_abs_sum': float(candidate_leaf[valid].abs().sum().item()),
                        'recovered_leaf_abs_sum': float(candidate_leaf[rec].abs().sum().item()),
                        'recovered_leaf_abs_mean_over_full_batch': float(candidate_leaf[rec].abs().sum().item()) / len(ratio),
                        'native_loss_sum': float((native['loss'][valid] * weight[valid]).sum().item()),
                        'candidate_loss_sum': float((candidate['loss'][valid] * weight[valid]).sum().item()),
                        'recovered_by_target_gene': self.histogram(target_gene, rec, 64),
                        'recovered_by_source_gene': self.histogram(source_gene, rec, 64),
                        'recovered_by_object': self.histogram(source_env % 947, rec, 947)}
                    intervals = [('below_lower', ratio < .9), ('at_lower', ratio == .9),
                        ('inside_old', (ratio > .9) & (ratio < 1.1)), ('at_old_upper', ratio == 1.1),
                        ('new_band', (ratio > 1.1) & (ratio < 1.2)), ('at_new_upper', ratio == 1.2),
                        ('above_new_upper', ratio > 1.2)]
                    row[label]['joint'] = {}
                    for sign, selected_a in [('negative', a < 0), ('zero', a == 0), ('positive', a > 0)]:
                        for region, selected_r in intervals:
                            selected = valid & selected_a & selected_r
                            delta = candidate_leaf[selected] - native_leaf[selected]
                            row[label]['joint'][sign + '/' + region] = {
                                'n': int(selected.sum().item()), 'A_sum': float(a[selected].sum().item()),
                                'delta_leaf_sum': float(delta.sum().item()),
                                'delta_leaf_abs_sum': float(delta.abs().sum().item())}
                    require(sum(item['n'] for item in row[label]['joint'].values()) == row[label]['finite'],
                            'Joint partition does not cover finite rows')
                require(row['off']['changed_leaf'] == 0 and row['on']['lost_leaf'] == 0,
                        'Counterfactual changed an unintended effective gradient region')
                self.stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                self.stream.flush()
                self.rows.append(row)
                require(row['finite_comparable'] == row['rows'], 'Nonfinite shadow input or loss')
            return result

        old_augment = self.patch(agent, 'augment_batch_for_mixed_expl', augment)
        old_prepare = self.patch(agent, 'prepare_dataset', prepare)
        old_actor = self.patch(agent, 'train_actor_critic', actor)
        old_loss = self.patch(agent, 'actor_loss_func', loss)

    def finalize(self, require_complete=True):
        if self.finalized is not None:
            return self.finalized
        complete = self.attempts == self.returns == len(self.rows) == 8
        recovered_attempts = sum(row['on']['positive_A_recovered_leaf'] > 0 for row in self.rows)
        finite = all(row['finite_comparable'] == row['rows'] for row in self.rows)
        result = {'status': 'completed' if complete else 'incomplete', 'attempts': self.attempts,
                  'actor_returns': self.returns, 'observed_actor_losses': len(self.rows),
                  'all_rows_finite': finite, 'recovered_on_attempts': recovered_attempts,
                  'recovered_on_row_exposures': sum(r['on']['recovered_leaf'] for r in self.rows),
                  'total_on_row_exposures': sum(r['on']['rows'] for r in self.rows),
                  'screening_gate': bool(complete and finite and recovered_attempts >= 2),
                  'screening_gate_meaning': 'Nontrivial actor effect warrants a fixed learning pilot; not a gain prediction',
                  'extra_model_forwards': 0, 'extra_model_backwards': 0,
                  'effective_training_actor_upper': .1, 'candidate_not_applied': True,
                  'provenance': 'Native LF/filter/shuffle of CUDA original env IDs; object ID = env ID modulo audited 947 catalogue'}
        self.stream.close()
        write(self.output / 'summary.json', result)
        self.finalized = result
        if require_complete:
            require(complete and finite, 'Incomplete or nonfinite shadow')
        return result

    def uninstall(self):
        errors = []
        for owner, name, prior, missing in reversed(self.patches):
            try:
                if prior is missing:
                    delattr(owner, name)
                else:
                    setattr(owner, name, prior)
            except BaseException as exc:
                errors.append(exc)
        self.patches.clear()
        if errors:
            raise errors[0]
