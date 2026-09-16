"""CUDA-only, goal-cohort feedback and shared target-position curriculum.

No top-level Torch/Isaac imports. Native loss/GAE/reset order is untouched.
install() owns its method patches; callers must close it before their outer patches.
New teacher checkpoints are archival only: automatic resume is unsupported.
"""
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cuda_tensor(t, x):
    require(t.is_tensor(x) and x.is_cuda and x.device.index == 0, 'Algorithm tensors must be CUDA:0')


def beta_mixed_score(t, successes, failures, group_size=16):
    """E[1-p**G-(1-p)**G] for Beta(1+S,1+F), not an iid-posterior claim."""
    cuda_tensor(t, successes); cuda_tensor(t, failures)
    require(successes.dtype == t.float64 and failures.dtype == t.float64, 'Counts must be CUDA FP64')
    require(successes.shape == failures.shape and group_size == 16, 'Invalid Beta-score contract')
    require(bool((t.isfinite(successes) & t.isfinite(failures) & (successes >= 0) & (failures >= 0)).all().item()), 'Invalid decayed counts')
    alpha, beta = successes + 1, failures + 1
    pos, neg = t.ones_like(alpha), t.ones_like(beta)
    for j in range(group_size):
        pos = pos * ((alpha + j) / (alpha + beta + j))
        neg = neg * ((beta + j) / (alpha + beta + j))
    score = 1 - pos - neg
    require(bool((t.isfinite(score) & (score >= 0) & (score <= 1)).all().item()), 'Invalid mixed-score arithmetic')
    return score


def shared_probabilities(t, successes, failures):
    require(tuple(successes.shape) == (2, 64, 9), 'Expected phase/gene/bin counts')
    score = beta_mixed_score(t, successes, failures)
    macro = score.mean(dim=1)
    require(bool((macro.sum(dim=1) > 0).all().item()), 'Empty teacher score mass')
    q = 0.5 / 9 + 0.5 * macro / macro.sum(dim=1, keepdim=True)
    require(bool((t.isfinite(q) & (q >= 0.5 / 9 - 1e-15)).all().item()), 'Invalid teacher support')
    require(bool(((q.sum(dim=1) - 1).abs() <= 1e-12).all().item()), 'Teacher does not normalize')
    return q, score


def position_bins(t, xyz, lo, hi):
    cuda_tensor(t, xyz); cuda_tensor(t, lo); cuda_tensor(t, hi)
    unit = (xyz.double() - lo.double()) / (hi.double() - lo.double())
    require(bool(t.isfinite(unit).all().item()), 'Nonfinite native target position')
    # Native FP32 coordinate rounding can put reconstructed u just outside [0,1].
    require(bool(((unit >= -1e-6) & (unit <= 1 + 1e-6)).all().item()), 'Target outside native volume')
    unit = unit.clamp(0, 1)
    ix = (unit[:, 0] * 3).long().clamp(0, 2)
    iz = (unit[:, 2] * 3).long().clamp(0, 2)
    return ix * 3 + iz, unit


def inverse_cdf_positions(t, native_xyz, lo, hi, q, phases):
    """Joint x/z inverse-CDF, using quantized uniforms reconstructed from native xyz.

    x marginal then z conditional; no fresh RNG. Native y is copied unchanged.
    Returned bin describes the requested stratum; realized FP32 bins are checked
    separately by the live hook and used as labels (rounding can cross a boundary).
    """
    cuda_tensor(t, q); cuda_tensor(t, phases)
    _, unit = position_bins(t, native_xyz, lo, hi)
    probs = q.index_select(0, phases).reshape(-1, 3, 3)
    px = probs.sum(dim=2)
    cdfx = px.cumsum(dim=1)
    ix = (unit[:, 0:1] >= cdfx[:, :2]).sum(dim=1).long()
    prevx = t.cat((t.zeros_like(cdfx[:, :1]), cdfx[:, :2]), dim=1)
    ux = (unit[:, 0] - prevx.gather(1, ix[:, None]).squeeze(1)) / px.gather(1, ix[:, None]).squeeze(1)
    pz = probs.gather(1, ix[:, None, None].expand(-1, 1, 3)).squeeze(1)
    pz = pz / pz.sum(dim=1, keepdim=True)
    cdfz = pz.cumsum(dim=1)
    iz = (unit[:, 2:3] >= cdfz[:, :2]).sum(dim=1).long()
    prevz = t.cat((t.zeros_like(cdfz[:, :1]), cdfz[:, :2]), dim=1)
    uz = (unit[:, 2] - prevz.gather(1, iz[:, None]).squeeze(1)) / pz.gather(1, iz[:, None]).squeeze(1)
    mapped = native_xyz.clone()
    mapped[:, 0] = (lo[0].double() + (hi[0] - lo[0]).double() * ((ix.double() + ux.clamp(0, 1)) / 3)).to(native_xyz.dtype)
    mapped[:, 2] = (lo[2].double() + (hi[2] - lo[2]).double() * ((iz.double() + uz.clamp(0, 1)) / 3)).to(native_xyz.dtype)
    return mapped, ix * 3 + iz


class CohortFeedback:
    """Four birth cohorts; never update from a partially resolved eligible cohort."""
    def __init__(self, t, cohort_steps=1024, maturity_steps=600):
        self.t = t
        self.cohort_steps, self.maturity_steps = cohort_steps, maturity_steps
        self.ids = [None] * 4
        self.born = t.zeros((4, 2, 64, 9), dtype=t.int64, device='cuda:0')
        self.success = t.zeros_like(self.born)
        self.failure = t.zeros_like(self.born)
        self.invalid = t.zeros_like(self.born)
        self.s = t.zeros((2, 64, 9), dtype=t.float64, device='cuda:0')
        self.f = t.zeros_like(self.s)
        self.generation = t.zeros(64, dtype=t.int64, device='cuda:0')
        self.merged = []
        self.last_merged = -1
        self.invalidated = 0
        self.deferred = []

    def ensure(self, cohort):
        slot = cohort % 4
        if self.ids[slot] != cohort:
            require(self.ids[slot] is None, 'Unresolved cohort ring would be overwritten')
            for field in (self.born, self.success, self.failure, self.invalid):
                field[slot].zero_()
            self.ids[slot] = cohort
        return slot

    def add_births(self, cohort, cells):
        slot = self.ensure(cohort)
        self.born[slot].view(-1).index_add_(0, cells, self.t.ones_like(cells))

    def close(self, cohorts, cells, successes):
        flat = (cohorts % 4) * (2 * 64 * 9) + cells
        ones = self.t.ones_like(cells)
        self.success.view(-1).index_add_(0, flat, ones * successes.long())
        self.failure.view(-1).index_add_(0, flat, ones * (~successes).long())

    def invalidate_gene(self, gene):
        require(0 <= gene < 64, 'Invalid GA gene')
        extra = self.born[:, :, gene] - self.invalid[:, :, gene]
        self.invalidated += int(extra.sum().item())
        self.invalid[:, :, gene].copy_(self.born[:, :, gene])
        self.success[:, :, gene].zero_(); self.failure[:, :, gene].zero_()
        self.s[:, gene].zero_(); self.f[:, gene].zero_()
        self.generation[gene] += 1

    def mature(self, control_step):
        """Merge chronologically, including empty cohorts, only after end+600."""
        completed = []
        candidate = self.last_merged + 1
        while control_step >= (candidate + 1) * self.cohort_steps + self.maturity_steps:
            slot = self.ensure(candidate)
            pending = self.born[slot] - self.success[slot] - self.failure[slot] - self.invalid[slot]
            require(bool((pending >= 0).all().item()), 'Cohort closure exceeds birth denominator')
            n = int(pending.sum().item())
            if n:
                self.deferred.append({'cohort': candidate, 'at_control_step': control_step, 'eligible_pending': n})
                break
            row = {'cohort': candidate, 'at_control_step': control_step,
                   'born': int(self.born[slot].sum().item()),
                   'success': int(self.success[slot].sum().item()),
                   'failure': int(self.failure[slot].sum().item()),
                   'invalid': int(self.invalid[slot].sum().item()), 'eligible_pending': 0}
            self.s.mul_(0.9).add_(self.success[slot].double())
            self.f.mul_(0.9).add_(self.failure[slot].double())
            self.ids[slot] = None
            for field in (self.born, self.success, self.failure, self.invalid):
                field[slot].zero_()
            self.last_merged = candidate
            self.merged.append(row); completed.append(row)
            candidate += 1
        return completed

    def state_dict(self):
        return {'ids': list(self.ids), 'last_merged': self.last_merged,
                'merged': list(self.merged), 'deferred': list(self.deferred),
                'invalidated': self.invalidated,
                **{k: getattr(self, k).detach().clone() for k in ('born', 'success', 'failure', 'invalid', 's', 'f', 'generation')}}


class FrontierCurriculum:
    def __init__(self, agent, config, output, _synthetic=False):
        import torch
        self.t, self.agent, self.env = torch, agent, agent.vec_env.env
        self.output = Path(output) / 'frontier'
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = {'enabled': bool(config['enabled']), 'warmup_epochs': int(config['warmup_epochs']),
                       'cohort_epochs': int(config.get('cohort_epochs', 64)),
                       'maturity_steps': int(config.get('maturity_steps', 600)),
                       'decay': float(config.get('decay', 0.9)),
                       'adaptive_mass': float(config.get('adaptive_mass', 0.5)),
                       'group_proxy_size': int(config.get('group_proxy_size', 16)),
                       'grid_size': int(config.get('grid_size', 3))}
        require(self.config['warmup_epochs'] >= 0, 'Invalid warmup')
        for key, value in {'cohort_epochs': 64, 'maturity_steps': 600, 'decay': .9, 'adaptive_mass': .5, 'group_proxy_size': 16, 'grid_size': 3}.items():
            require(self.config[key] == value, 'Frozen teacher constant differs: ' + key)
        self.n = int(agent.num_actors)
        self.block = int(agent.intr_coef_block_size)
        require(self.n == self.block * 64 and (_synthetic or (self.n == 20480 and self.block == 320)), 'Unsupported population layout')
        require(int(agent.horizon_length) == 16 and int(agent.seq_length) == 16, 'Expected native H16/seq16')
        require(not agent.multi_gpu, 'Single GPU only')
        require(int(self.env.max_episode_length) == 600 and int(self.env.success_steps) == 1 and int(self.env.max_consecutive_successes) == 50, 'Goal lifetime/cap assumptions differ')
        require(int(self.env.good_reset_boundary) == 0, 'Saved-state reset path unsupported')
        self.synthetic = bool(_synthetic)
        require(len(self.env.object_asset_files) == 947 and len(self.env.object_asset_scales) == 947, 'Fixed947 catalog differs')
        self.catalog = [{'asset_name': str(name), 'scale': [float(v) for v in scale]}
                        for name, scale in zip(self.env.object_asset_files, self.env.object_asset_scales)]
        self.catalog_sha256 = hashlib.sha256(json.dumps(self.catalog, sort_keys=True).encode()).hexdigest()
        self.lo = (self.env.target_volume_origin + self.env.target_volume_extent[:, 0]).detach().clone()
        self.hi = (self.env.target_volume_origin + self.env.target_volume_extent[:, 1]).detach().clone()
        cuda_tensor(torch, self.lo); cuda_tensor(torch, self.hi)
        require(bool((self.hi > self.lo).all().item()), 'Invalid target volume')
        self.gene = torch.arange(self.n, device='cuda:0', dtype=torch.int64) // self.block
        # Ordinals refer to the fixed catalog order; root binds source/catalog metadata.
        self.object = torch.arange(self.n, device='cuda:0', dtype=torch.int64) % 947
        self.feedback = CohortFeedback(torch)
        self.q = torch.full((2, 9), 1 / 9, dtype=torch.float64, device='cuda:0')
        self.q_generation = torch.zeros(64, dtype=torch.int64, device='cuda:0')
        self.q_teacher_version = 0
        self.q_success_proxy = torch.full((2, 9), .5, dtype=torch.float64, device='cuda:0')
        self.teacher_proxy = {}
        self.q_nonuniform = False
        self.active_proposals = self.teacher_updates = self.teacher_computations = 0
        self.boundary_rounding_mismatches = 0
        self.control_steps = self.relative_epoch = self.completed_epochs = 0
        self._inside_epoch = False
        self._reset_depth = 0
        self.closed = False
        self.first_error = None
        self._patches = []
        self.ga_events = []
        z = lambda: torch.zeros(self.n, dtype=torch.int64, device='cuda:0')
        b = lambda: torch.zeros(self.n, dtype=torch.bool, device='cuda:0')
        self.episode_id, self.goal_id = z(), z()
        self.episode_start, self.goal_start = z(), z()
        self.goal_cell, self.goal_cohort = z(), z()
        self.goal_m, self.goal_teacher, self.goal_policy, self.goal_generation = z(), z(), z(), z()
        self.episode_policy, self.episode_ga_changes = z(), z()
        self.episode_open = torch.ones(self.n, dtype=torch.bool, device='cuda:0')
        self.goal_open = torch.ones_like(self.episode_open)
        self.episode_complete_origin, self.goal_complete_origin, self.goal_eligible = b(), b(), b()
        self.last_done_step = torch.full((self.n,), -1, dtype=torch.int64, device='cuda:0')
        self.new_finished_envs = b()
        self.goal_counts = torch.zeros((4, 2, 64, 9), dtype=torch.int64, device='cuda:0')
        self.goal_duration_sum = torch.zeros((2, 2, 64, 9), dtype=torch.int64, device='cuda:0')
        self.goal_reason_counts = torch.zeros((5, 2, 64, 9), dtype=torch.int64, device='cuda:0')
        self.goal_phase_m_counts = torch.zeros((4, 2, 51), dtype=torch.int64, device='cuda:0')
        self.prescribed_bin_mass = torch.zeros((2, 9), dtype=torch.float64, device='cuda:0')
        self.eligible_at_close_cumulative = 0
        self.episode_hist = torch.zeros((64, 51), dtype=torch.int64, device='cuda:0')
        self.object_gene = torch.zeros((4, 947, 64), dtype=torch.int64, device='cuda:0')
        self.inherited_goal_closures = self.inherited_episode_closures = self.forced_goal_resets = 0
        self.raw_episode_done_count = self.episode_births = self.full_episode_done_count = self.episode_cancelled = 0
        try:
            self._install()
            self._write('catalog.json', {'ordered_catalog': self.catalog, 'sha256': self.catalog_sha256, 'mapping': 'object=env_id%947; gene=env_id//block_size', 'mapping_scope': 'native source construction plus actual asset catalog, not a separately queried PhysX actor asset table'})
            self._write('resolved.json', {'config': self.config, 'schema': 1,
                        'scope': 'Birth-cohort population heuristic; no iid posterior or full DAPO claim',
                        'teacher_shape': [2, 9], 'stat_shape': [2, 64, 9],
                        'bin_order': '3*x_bin+z_bin', 'phases': ['initial', 'continuation'],
                        'uniform_source': 'FP64 inversion of native FP32 coordinates, not original random-draw bits',
                        'resume_supported': False, 'synthetic_fixture': self.synthetic})
        except BaseException as exc:
            self._remember(exc, 'installation')
            try:
                self._uninstall()
            except BaseException:
                pass  # Preserve the installation's first exception.
            raise

    def _remember(self, exc, where):
        if self.first_error is None:
            self.first_error = {'type': type(exc).__name__, 'message': str(exc), 'where': where}

    def _guard(self, where, func):
        def call(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                self._remember(exc, where)
                raise
        return call

    def _write(self, name, value):
        with (self.output / name).open('w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    def _bind(self, owner, name, value):
        had = name in owner.__dict__
        old = owner.__dict__.get(name)
        self._patches.append((owner, name, had, old))
        setattr(owner, name, value)

    def _uninstall(self):
        errors = []
        for owner, name, had, old in reversed(self._patches):
            try:
                if had: setattr(owner, name, old)
                elif name in owner.__dict__: delattr(owner, name)
            except BaseException as exc:
                errors.append(exc)
        self._patches.clear()
        if errors: raise errors[0]

    def _ids(self, ids):
        cuda_tensor(self.t, ids)
        return ids.reshape(-1).long()

    def _increment(self, array, ids, weights=None):
        array.view(-1).index_add_(0, ids, self.t.ones_like(ids) if weights is None else weights)

    def _cancel_open_goals(self, ids):
        known = ids[self.goal_open[ids] & self.goal_complete_origin[ids]]
        if known.numel():
            self.forced_goal_resets += known.numel()
            self._increment(self.goal_counts[3], self.goal_cell[known])
            self._increment(self.goal_phase_m_counts[3], (self.goal_cell[known] // (64 * 9)) * 51 + self.goal_m[known])
            eligible = known[self.goal_eligible[known]]
            if eligible.numel():
                flat = (self.goal_cohort[eligible] % 4) * (2 * 64 * 9) + self.goal_cell[eligible]
                self._increment(self.feedback.invalid, flat)
        self.goal_open[ids] = False
        self.goal_eligible[ids] = False

    def _install(self):
        native_target, native_reset, native_reward = self.env._reset_target, self.env.reset_idx, self.env.compute_kuka_reward
        def target(env_ids, *args, **kwargs):
            result = native_target(env_ids, *args, **kwargs)
            tensor_reset = kwargs.get('tensor_reset', args[1] if len(args) > 1 else True)
            if self._inside_epoch and tensor_reset:
                reset_buf_idxs = kwargs.get('reset_buf_idxs', args[0] if args else None)
                require(reset_buf_idxs is None, 'Non-native target reset source unsupported')
                self._target_birth(self._ids(env_ids), initial=self._reset_depth > 0)
            return result
        def reset(env_ids, *args, **kwargs):
            episode_reset = kwargs.get('episode_reset', args[1] if len(args) > 1 else True)
            tensor_reset = kwargs.get('tensor_reset', args[2] if len(args) > 2 else True)
            tracked = self._inside_epoch and episode_reset and tensor_reset
            ids = self._ids(env_ids) if tracked else None
            if tracked:
                self._cancel_open_goals(ids)
                self.episode_cancelled += int((self.episode_open[ids] & self.episode_complete_origin[ids]).sum().item())
                self.episode_id[ids] += 1
                self.episode_start[ids] = self.control_steps
                self.episode_policy[ids] = self.relative_epoch
                self.episode_open[ids] = True
                self.episode_complete_origin[ids] = True
                self.episode_ga_changes[ids] = 0
                self.episode_births += ids.numel()
                self._reset_depth += 1
            try:
                return native_reset(env_ids, *args, **kwargs)
            finally:
                if tracked: self._reset_depth -= 1
        def reward(*args, **kwargs):
            result = native_reward(*args, **kwargs)
            if self._inside_epoch:
                require(isinstance(result, tuple) and len(result) == 2, 'Native reward return contract differs')
                self._after_reward(result[1])
            return result
        self._bind(self.env, '_reset_target', self._guard('target_reset', target))
        self._bind(self.env, 'reset_idx', self._guard('episode_reset', reset))
        self._bind(self.env, 'compute_kuka_reward', self._guard('reward_event', reward))

    def _target_birth(self, ids, initial):
        if not ids.numel(): return
        self._cancel_open_goals(ids)
        phases = self.t.full_like(ids, 0 if initial else 1)
        native_xyz = self.env.goal_states[ids, :3]
        active = self.config['enabled'] and self.teacher_updates > 0 and self.q_nonuniform
        if active:
            mapped, requested_bins = inverse_cdf_positions(self.t, native_xyz, self.lo, self.hi, self.q, phases)
            self.env.goal_states[ids, 0] = mapped[:, 0]
            self.env.goal_states[ids, 2] = mapped[:, 2]
            root_ids = self.env.goal_object_indices[ids]
            self.env.root_state_tensor[root_ids, 0] = mapped[:, 0]
            self.env.root_state_tensor[root_ids, 2] = mapped[:, 2]
            bins, _ = position_bins(self.t, mapped, self.lo, self.hi)
            self.boundary_rounding_mismatches += int((bins != requested_bins).sum().item())
            self.active_proposals += ids.numel()
        else:
            bins, _ = position_bins(self.t, native_xyz, self.lo, self.hi)
        cells = (phases * 64 + self.gene[ids]) * 9 + bins
        cohort = self.control_steps // 1024
        self.feedback.add_births(cohort, cells)
        self._increment(self.goal_counts[0], cells)
        self.goal_id[ids] += 1
        self.goal_start[ids] = self.control_steps
        self.goal_m[ids] = 0 if initial else self.env.successes[ids].long()
        require(bool(((self.goal_m[ids] >= 0) & (self.goal_m[ids] < 50)).all().item()), 'Native new-goal success level outside 0..49')
        self._increment(self.goal_phase_m_counts[0], phases * 51 + self.goal_m[ids])
        phase = 0 if initial else 1
        self.prescribed_bin_mass[phase].add_(self.q[phase] * ids.numel() if active else ids.numel() / 9)
        self.goal_cell[ids] = cells
        self.goal_cohort[ids] = cohort
        self.goal_teacher[ids] = self.q_teacher_version
        self.goal_policy[ids] = self.relative_epoch
        self.goal_generation[ids] = self.feedback.generation[self.gene[ids]]
        self.goal_open[ids] = True
        self.goal_complete_origin[ids] = True
        self.goal_eligible[ids] = True

    def _after_reward(self, is_success):
        t = self.t
        cuda_tensor(t, is_success)
        self.control_steps += 1
        success = is_success.reshape(-1).bool()
        done = self.env.reset_buf.reshape(-1).bool()
        require(success.numel() == self.n and done.numel() == self.n, 'Native event shape differs')
        closing = (success | done) & self.goal_open
        ids = closing.nonzero(as_tuple=False).reshape(-1)
        known = ids[self.goal_complete_origin[ids]]
        self.inherited_goal_closures += ids.numel() - known.numel()
        if known.numel():
            cells = self.goal_cell[known]
            good = success[known]
            self._increment(self.goal_counts[1], cells, good.long())
            self._increment(self.goal_counts[2], cells, (~good).long())
            pm = (cells // (64 * 9)) * 51 + self.goal_m[known]
            self._increment(self.goal_phase_m_counts[1], pm, good.long())
            self._increment(self.goal_phase_m_counts[2], pm, (~good).long())
            duration = self.control_steps - self.goal_start[known]
            self._increment(self.goal_duration_sum[0], cells, duration * good.long())
            self._increment(self.goal_duration_sum[1], cells, duration * (~good).long())
            flags = (success, self.env.object_pos[:, 2] < .1,
                     self.env.curr_fingertip_distances.max(dim=-1).values > 1.5,
                     self.env.progress_buf >= self.env.max_episode_length - 1,
                     self.env.successes >= 50)
            for index, flag in enumerate(flags):
                self._increment(self.goal_reason_counts[index], cells, flag[known].long())
            eligible = known[self.goal_eligible[known]]
            if eligible.numel():
                self.feedback.close(self.goal_cohort[eligible], self.goal_cell[eligible], success[eligible])
                self.eligible_at_close_cumulative += eligible.numel()
        self.goal_open[ids] = False
        self.goal_eligible[ids] = False
        end = done.nonzero(as_tuple=False).reshape(-1)
        if end.numel():
            self.raw_episode_done_count += end.numel()
            complete = end[self.episode_open[end] & self.episode_complete_origin[end]]
            self.inherited_episode_closures += end.numel() - complete.numel()
            self.full_episode_done_count += complete.numel()
            if complete.numel():
                s = self.env.successes[complete]
                require(bool((self.t.isfinite(s) & (s == s.long()) & (s >= 0) & (s <= 50)).all().item()), 'Episode success not a capped integer')
                g, o = self.gene[complete], self.object[complete]
                self._increment(self.episode_hist, g * 51 + s.long())
                cell = o * 64 + g
                for index, values in enumerate((self.t.ones_like(g), s.long(), (s == 0).long(), (s == 50).long())):
                    self._increment(self.object_gene[index], cell, values)
            self.episode_open[end] = False
            self.last_done_step[end] = self.control_steps
            self.new_finished_envs[end] = True

    def epoch_begin(self, relative_epoch):
        require(not self.closed and not self._inside_epoch, 'Invalid epoch begin')
        require(int(relative_epoch) == self.completed_epochs + 1, 'Nonsequential epoch')
        require(self.control_steps == self.completed_epochs * 16, 'Control-step clock drift')
        self.relative_epoch = int(relative_epoch)
        self._inside_epoch = True

    def epoch_end(self, relative_epoch, new_frames):
        require(self._inside_epoch and int(relative_epoch) == self.relative_epoch, 'Invalid epoch end')
        require(self.control_steps == int(relative_epoch) * 16, 'Native reward/control-step quota differs')
        require(int(new_frames) == self.control_steps * self.n, 'Physical frame clock differs')
        self._inside_epoch = False
        self.completed_epochs = int(relative_epoch)
        if self.completed_epochs % 64 == 0:
            self.feedback.mature(self.control_steps)
            if self.completed_epochs >= self.config['warmup_epochs']:
                self.q, _ = shared_probabilities(self.t, self.feedback.s, self.feedback.f)
                self.q_success_proxy = ((1 + self.feedback.s) / (2 + self.feedback.s + self.feedback.f)).mean(dim=1)
                self.q_teacher_version += 1
                self.q_generation.copy_(self.feedback.generation)
                self.q_nonuniform = bool((self.q != self.q[:, :1]).any().item())
                self.teacher_computations += 1
                if self.config['enabled']: self.teacher_updates += 1
                self._update_teacher_proxy()
            self._write('cohort_%05d.json' % self.completed_epochs, self._snapshot())
        self._assert_counts()
        # A separate JSONL is never passed through TensorBoard's time-based throttle.
        summary = self.summary()
        with (self.output / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(summary, sort_keys=True, allow_nan=False) + '\n')
        self._write_scalars(summary)

    def _update_teacher_proxy(self):
        # These estimates and q remain jointly frozen between teacher boundaries.
        # They describe a population smoothing heuristic, not measured performance.
        for phase, label in enumerate(('initial', 'continuation')):
            q, p = self.q[phase], self.q_success_proxy[phase]
            self.teacher_proxy[label] = {
                'q_weighted_success_proxy': float((q * p).sum().item()),
                'uniform_weighted_success_proxy': float(p.mean().item()),
                'kl_q_to_uniform': float((q * (q * 9).log()).sum().item()),
                'min_bin_probability': float(q.min().item()),
                'max_bin_probability': float(q.max().item())}

    def _write_scalars(self, summary):
        writer = self.agent.writer
        step = int(self.agent.frame)  # Same pre-increment clock as native write_stats.
        for key in ('new_physical_frames', 'teacher_updates', 'active_proposals',
                    'goal_births', 'goal_success', 'goal_failure', 'goal_pending',
                    'teacher_eligible_pending', 'eligible_at_close_cumulative',
                    'complete_new_episode_endings', 'new_finished_env_count'):
            writer.add_scalar('frontier/' + key, summary[key], step)
        for label, hist in (('population', self.episode_hist.sum(dim=0)), ('master63', self.episode_hist[63])):
            count = hist.sum()
            if int(count.item()):
                values = self.t.arange(51, dtype=self.t.float64, device='cuda:0')
                for key, value in (('mean_success', (hist.double() * values).sum() / count),
                                   ('zero_fraction', hist[0].double() / count),
                                   ('cap_fraction', hist[50].double() / count)):
                    writer.add_scalar('frontier/new_complete_episode_' + label + '/' + key, float(value.item()), step)
            writer.add_scalar('frontier/new_complete_episode_' + label + '/count', int(count.item()), step)
        for phase, row in self.teacher_proxy.items():
            for key, value in row.items():
                writer.add_scalar('frontier/teacher_smoothed_proxy/' + phase + '/' + key, value, step)

    def on_ga(self, replaced_gene):
        gene = int(replaced_gene)
        self.feedback.invalidate_gene(gene)
        mask = self.gene == gene
        self.goal_eligible[mask] = False
        self.episode_ga_changes[mask & self.episode_open] += 1
        self.ga_events.append({'gene': gene, 'relative_epoch': self.relative_epoch,
                               'control_step': self.control_steps,
                               'generation': int(self.feedback.generation[gene].item()),
                               'q_teacher_version_still_frozen': self.q_teacher_version})

    def _assert_counts(self):
        born = self.goal_counts[0].sum()
        ended = self.goal_counts[1:].sum()
        pending = (self.goal_open & self.goal_complete_origin).sum()
        require(bool((born == ended + pending).item()), 'Physical goal birth/closed/pending mismatch')
        require(int(self.episode_hist.sum().item()) == self.full_episode_done_count, 'Episode histogram count mismatch')
        ep_pending = int((self.episode_open & self.episode_complete_origin).sum().item())
        require(self.episode_births == self.full_episode_done_count + self.episode_cancelled + ep_pending, 'Episode birth/closed/pending mismatch')

    def summary(self):
        def scalar(x): return int(x.item())
        return {'schema': 1, 'status': 'failed' if self.first_error is not None else ('closed' if self.closed else 'observing'),
                'enabled': self.config['enabled'], 'completed_epochs': self.completed_epochs,
                'control_steps': self.control_steps, 'new_physical_frames': self.control_steps * self.n,
                'active_proposals': self.active_proposals, 'teacher_updates': self.teacher_updates,
                'teacher_computations': self.teacher_computations, 'q_teacher_version': self.q_teacher_version,
                'q_nonuniform': self.q_nonuniform,
                'native_coordinate_rounding_bin_changes': self.boundary_rounding_mismatches,
                'goal_births': scalar(self.goal_counts[0].sum()),
                'goal_success': scalar(self.goal_counts[1].sum()),
                'goal_failure': scalar(self.goal_counts[2].sum()),
                'goal_cancelled': scalar(self.goal_counts[3].sum()),
                'goal_pending': scalar((self.goal_open & self.goal_complete_origin).sum()),
                'teacher_eligible_pending': scalar((self.goal_open & self.goal_eligible).sum()),
                'eligible_at_close_cumulative': self.eligible_at_close_cumulative,
                'teacher_invalidated_births': self.feedback.invalidated,
                'cohorts_merged': len(self.feedback.merged), 'cohorts_deferred': len(self.feedback.deferred),
                'new_episode_births': self.episode_births,
                'new_episode_cancelled': self.episode_cancelled,
                'new_episode_pending': scalar((self.episode_open & self.episode_complete_origin).sum()),
                'catalog_sha256': self.catalog_sha256,
                'complete_new_episode_endings': self.full_episode_done_count,
                'raw_done_events': self.raw_episode_done_count,
                'inherited_goal_closures': self.inherited_goal_closures,
                'inherited_episode_closures': self.inherited_episode_closures,
                'new_finished_env_count': scalar(self.new_finished_envs.sum()),
                'oldest_unfinished_env_age_steps': self.control_steps if bool((~self.new_finished_envs).any().item()) else 0,
                'ga_notifications': len(self.ga_events), 'resume_supported': False,
                'teacher_smoothed_proxy_at_q_version': self.teacher_proxy,
                'first_error': self.first_error}

    def _snapshot(self, include_object_counts=False):
        cpu = lambda x: x.detach().cpu().tolist()
        pending = self.feedback.born - self.feedback.success - self.feedback.failure - self.feedback.invalid
        snapshot = {'summary': self.summary(), 'resolved': self.config,
                'q': cpu(self.q), 'q_generation': cpu(self.q_generation),
                'current_gene_generation': cpu(self.feedback.generation),
                'ema_success': cpu(self.feedback.s), 'ema_failure': cpu(self.feedback.f),
                'cohort_ids': list(self.feedback.ids), 'cohort_born': cpu(self.feedback.born),
                'cohort_success': cpu(self.feedback.success), 'cohort_failure': cpu(self.feedback.failure),
                'cohort_invalid': cpu(self.feedback.invalid), 'cohort_eligible_pending': cpu(pending),
                'merged': list(self.feedback.merged), 'deferred': list(self.feedback.deferred),
                'physical_goal_counts_born_success_failure_cancelled': cpu(self.goal_counts),
                'goal_reason_counts_success_fall_handfar_timeout_cap': cpu(self.goal_reason_counts),
                'goal_duration_sum_success_failure': cpu(self.goal_duration_sum),
                'goal_phase_start_m_born_success_failure_cancelled': cpu(self.goal_phase_m_counts),
                'sum_prescribed_bin_probabilities': cpu(self.prescribed_bin_mass),
                'prescribed_mass_scope': 'Cumulative conditional probabilities at each actual proposal; native bypass uses 1/9. Finite-precision coordinate inversion may shift realized bins.',
                'full_new_episode_histogram_per_gene': cpu(self.episode_hist),
                'ga_events': list(self.ga_events),
                'full_event_trajectories_saved': False}
        if include_object_counts:
            snapshot['object_gene_complete_episode_count_sumS_zero_cap'] = cpu(self.object_gene)
        snapshot['object_gene_full_counts_included'] = bool(include_object_counts)
        return snapshot

    def state_dict(self):
        names = ('q', 'q_generation', 'q_success_proxy', 'gene', 'object', 'episode_id', 'goal_id', 'episode_start',
                 'goal_start', 'goal_cell', 'goal_cohort', 'goal_m', 'goal_teacher', 'goal_policy',
                 'goal_generation', 'episode_policy', 'episode_ga_changes', 'episode_open',
                 'goal_open', 'episode_complete_origin', 'goal_complete_origin', 'goal_eligible',
                 'last_done_step', 'new_finished_envs', 'goal_counts', 'goal_duration_sum',
                 'goal_reason_counts', 'goal_phase_m_counts', 'prescribed_bin_mass', 'episode_hist', 'object_gene')
        return {'schema': 1, 'resume_supported': False, 'config': dict(self.config),
                'summary': self.summary(), 'feedback': self.feedback.state_dict(),
                'ga_events': list(self.ga_events),
                'tensors': {k: getattr(self, k).detach().clone() for k in names}}

    def close(self):
        if self.closed: return
        first = None
        try:
            self._assert_counts()
            self._write('final_snapshot.json', self._snapshot(include_object_counts=True))
        except BaseException as exc:
            first = exc
            self._remember(exc, 'close_snapshot')
        try:
            self._uninstall()
        except BaseException as exc:
            self._remember(exc, 'uninstall')
            if first is None: first = exc
        self.closed = True
        try:
            self._write('summary.json', self.summary())
        except BaseException as exc:
            self._remember(exc, 'close_summary')
            if first is None: first = exc
        if first is not None: raise first


def install(agent, config, output, patches=None):
    require(patches is None, 'Frontier owns its hooks; call close before restoring outer patches')
    return FrontierCurriculum(agent, config, output)
