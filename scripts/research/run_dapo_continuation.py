#!/usr/bin/env python3
"""Source-bound native EPO training; actor upper-clip is the only loss change.

Default is a stdlib-only plan check. Execution needs an explicitly ready plan
and --execute. This entry never starts another job, retries or evaluates scores.
"""
import argparse
import fcntl
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/home/ubuntu22/EPO')
AUDIT = ROOT / 'artifacts/research/20260914-dapo-transfer'
BASE_PATH = ROOT / 'scripts/research/run_training_probe.py'
ACTOR_PATH = ROOT / 'scripts/research/dapo_actor.py'
E51_SHA = 'de220625c407b463d41f9242cd652a40f5e48c253b4b9225b7817e452cf0f6be'
QUANTUM, N = 327680, 20480
base = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def configure_execution_root(plan):
    """Only load a hash-bound utility after reading the explicit execution root."""
    global ROOT, AUDIT, BASE_PATH, ACTOR_PATH, base, write, identity, module
    ROOT = Path(plan.get('execution_root', '/home/ubuntu22/EPO')).resolve()
    require(ROOT.is_absolute() and ROOT.is_dir(), 'Missing execution root')
    AUDIT = ROOT / 'artifacts'
    BASE_PATH, ACTOR_PATH = Path(plan['base_source']), Path(plan['actor_source'])
    require(ROOT in BASE_PATH.resolve().parents and ROOT in ACTOR_PATH.resolve().parents,
            'Executing base and actor sources must be inside the isolated root')
    require(BASE_PATH.is_file() and not BASE_PATH.is_symlink(), 'Nonregular base source')
    sha = hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()
    require(plan['files_sha256'].get(str(BASE_PATH)) == sha, 'Unbound bootstrap utility')
    spec = importlib.util.spec_from_file_location('dapo_frozen_probe_utilities', BASE_PATH)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    require(Path(base.ROOT).resolve() == ROOT, 'Utility ROOT differs from execution root')
    write, identity, module = base.write, base.identity, base.module


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_scalar': repr(value)}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def scalar(value):
    if hasattr(value, 'detach'):
        return float(value.detach().item())
    return float(value)


def clocks(plan):
    start, epoch = (5000069120, 15259) if plan['mode'] == 'e51' else (0, 0)
    return start, epoch, start + plan['epochs'] * QUANTUM


def validate_plan(plan, plan_path):
    require(plan['purpose'] == 'dapo_actor_clip_higher_training', 'Wrong plan purpose')
    require(plan['mode'] in ('e51', 'fresh') and plan['arm'] in ('control', 'candidate'), 'Unsupported method identity')
    require(type(plan['epochs']) is int and 1 <= plan['epochs'] <= 15259, 'Explicit bounded complete-epoch budget required')
    require(type(plan['seed']) is int and 0 < plan['seed'] < 2**31, 'Explicit valid seed required')
    require(plan['clip_low'] == .1 and plan['clip_high'] == (.1 if plan['arm'] == 'control' else .2), 'Only the fixed actor upper change is supported')
    require(isinstance(plan['execution_ready'], bool), 'Explicit execution_ready flag required')
    require(isinstance(plan['run_id'], str) and plan['run_id'].isalnum(), 'Explicit own run ID required')
    if plan['mode'] == 'e51':
        require(plan['seed'] == 35001 and bool(plan['checkpoint']), 'E51 screen has a fixed continuation seed/checkpoint')
    else:
        require(plan['checkpoint'] in (None, ''), 'Fresh mode must have no checkpoint')
    require(Path(plan['base_source']).resolve() == BASE_PATH.resolve(), 'Wrong frozen utility source')
    require(Path(plan['actor_source']).resolve() == ACTOR_PATH.resolve(), 'Wrong actor implementation source')
    output = Path(plan['output'])
    require(output.is_absolute() and AUDIT in output.parents and output.resolve() == output,
            'Output must be an explicit non-symlink study path')
    require(not output.exists(), 'Existing output; no resume or retry')
    required = {str(Path(__file__).resolve()), str(BASE_PATH), str(ACTOR_PATH),
                plan['config'], plan['nvml_source'], plan['finite_source']}
    require(all(ROOT in Path(plan[key]).resolve().parents for key in ('config', 'finite_source', 'nvml_source')),
            'Config and executing helpers must be archived within the execution root')
    if plan['mode'] == 'e51':
        required.add(plan['checkpoint'])
        require(plan['files_sha256'].get(plan['checkpoint']) == E51_SHA, 'E51 checkpoint needs its explicit immutable input binding')
    required.update(plan.get('extra_plan_files', {}).values())
    require(required <= set(plan['files_sha256']), 'Unbound executing source/config/attachment')
    for name, path in plan.get('extra_plan_files', {}).items():
        require(Path(name).name == name and name not in ('plan.json', 'config.resolved.yaml'), 'Unsafe attachment name')
    for path, sha in plan['files_sha256'].items():
        require(Path(path).is_absolute() and identity(path)['sha256'] == sha, 'Reviewed input changed: ' + path)
    require(Path(plan_path).is_file(), 'Missing exact plan file')
    return {'status': 'prepared', 'gpu_executed': False, 'api_called': False,
            'mode': plan['mode'], 'arm': plan['arm'], 'epochs': plan['epochs'],
            'start_frame': clocks(plan)[0], 'target_frame': clocks(plan)[2],
            'expected_physical_frames': plan['epochs'] * QUANTUM,
            'expected_actor_attempts': plan['epochs'] * 8,
            'expected_scheduler_calls': plan['epochs'] * 2,
            'execution_ready': plan['execution_ready']}


class Patches:
    def __init__(self):
        self.items = []

    def set(self, owner, name, value):
        original = getattr(owner, name)
        present = name in owner.__dict__
        prior = owner.__dict__.get(name)
        self.items.append((owner, name, present, prior))
        setattr(owner, name, value)
        return original

    def restore(self, remember):
        for owner, name, present, prior in reversed(self.items):
            try:
                if present:
                    setattr(owner, name, prior)
                else:
                    delattr(owner, name)
            except BaseException as exc:
                remember(exc, 'patch_restore.' + name)
        self.items.clear()


class Observations:
    """Bounded scalar observations, no extra forward/backward or RNG sampling."""
    def __init__(self, agent, plan, output, nvml, remember, patches):
        self.agent, self.plan, self.nvml = agent, plan, nvml
        self.remember, self.patches = remember, patches
        self.stream = (output / 'training.jsonl').open('x')
        self.active = self.in_actor = False
        self.counts = {key: 0 for key in ('epoch_entries', 'epoch_returns', 'env_step_attempts',
                       'env_step_returns', 'actor_attempts', 'actor_returns', 'adam_entered',
                       'adam_returned', 'adam_exceptions', 'scheduler_calls', 'scheduler_returns',
                       'lr_calls', 'lr_returns', 'prepare_returns', 'history_releases',
                       'ga_merge_calls', 'ga_merge_returns', 'gene_disable_calls', 'gene_disable_returns',
                       'resource_samples')}
        self.core_seconds = self.epoch_wall_seconds = 0.0
        self.last_resource = None

    def emit(self, kind, **fields):
        row = dict(kind=kind, utc=base.now(), monotonic=time.monotonic(),
                   epoch=int(self.agent.epoch_num), frame=int(self.agent.frame),
                   actor_ordinal=self.counts['actor_attempts'], **fields)
        self.stream.write(json.dumps(clean(row), separators=(',', ':'), allow_nan=False) + '\n')
        self.stream.flush()

    def resource(self, boundary):
        row = dict(boundary=boundary, disk_free_bytes=shutil.disk_usage(ROOT).free, **self.nvml())
        self.emit('resource', **row)
        self.counts['resource_samples'] += 1
        self.last_resource = row
        require(row['driver_free_bytes'] >= 2**30, 'Post-native-return driver free below 1GiB')
        require(row['disk_free_bytes'] >= 4 * 2**30, 'Post-native-return disk free below 4GiB')
        return row

    def counted(self, owner, name, entered, returned):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            self.counts[entered] += 1
            try:
                result = original(*args, **kwargs)
                self.counts[returned] += 1
                return result
            except BaseException as exc:
                if entered == 'adam_entered':
                    self.counts['adam_exceptions'] += 1
                self.remember(exc, name)
                raise
        self.patches.set(owner, name, call)

    def install(self):
        agent = self.agent
        self.counted(agent, 'env_step', 'env_step_attempts', 'env_step_returns')
        self.counted(agent.optimizer, 'step', 'adam_entered', 'adam_returned')
        self.counted(agent, 'update_lr', 'lr_calls', 'lr_returns')
        old_scheduler = agent.scheduler.update

        def scheduler(*args, **kwargs):
            self.counts['scheduler_calls'] += 1
            try:
                result = old_scheduler(*args, **kwargs)
                self.counts['scheduler_returns'] += 1
                self.emit('scheduler', actual_input=list(args), actual_kwargs=kwargs,
                          actual_return=list(result))
                return result
            except BaseException as exc:
                self.remember(exc, 'scheduler.update')
                raise
        self.patches.set(agent.scheduler, 'update', scheduler)
        network = agent.model.a2c_network
        for owner, name, key in ((network, 'merge_extra_params', 'ga_merge'),
                                  (agent, 'disable_gene_optimizer_updates', 'gene_disable')):
            original = getattr(owner, name)

            def event(*args, _original=original, _name=name, _key=key, **kwargs):
                self.counts[_key + '_calls'] += 1
                self.emit(_name + '_entered', in_training_epoch=self.active,
                          arguments=repr(args), counts=dict(self.counts))
                try:
                    result = _original(*args, **kwargs)
                    self.counts[_key + '_returns'] += 1
                    self.emit(_name + '_returned', in_training_epoch=self.active)
                    return result
                except BaseException as exc:
                    self.remember(exc, _name)
                    raise
            self.patches.set(owner, name, event)
        old_prepare = agent.prepare_dataset

        def prepare(*args, **kwargs):
            result = old_prepare(*args, **kwargs)
            self.counts['prepare_returns'] += 1
            require(len(agent.dataset) == 4, 'Native dataset must have exactly four batches')
            self.emit('dataset_prepared', batch_count=len(agent.dataset))
            return result
        self.patches.set(agent, 'prepare_dataset', prepare)
        old_actor = agent.train_actor_critic

        def actor(*args, **kwargs):
            self.counts['actor_attempts'] += 1
            self.in_actor = True
            before = dict(adam_entered=self.counts['adam_entered'], adam_returned=self.counts['adam_returned'],
                          scale=float(agent.scaler.get_scale()), lr=float(agent.last_lr))
            started = time.perf_counter()
            self.emit('actor_entered', **before)
            try:
                result = old_actor(*args, **kwargs)
                self.counts['actor_returns'] += 1  # Native returned even if later telemetry/resource fails.
                self.emit('actor_returned', native_returned=True, scale_before=before['scale'],
                          scale_after=float(agent.scaler.get_scale()), lr_before=before['lr'],
                          native_actor_seconds=time.perf_counter() - started,
                          adam_entered=self.counts['adam_entered'] - before['adam_entered'],
                          adam_returned=self.counts['adam_returned'] - before['adam_returned'],
                          native_actor_loss=scalar(result[0]), native_critic_loss=scalar(result[1]),
                          native_entropy=scalar(result[2]), native_kl=scalar(result[3]),
                          native_return_lr=scalar(result[4]), native_bounds_loss=scalar(result[8]))
                self.resource('after_native_actor_return')
                return result
            except BaseException as exc:
                self.remember(exc, 'actor_or_post_return_observation')
                raise
            finally:
                self.in_actor = False
        self.patches.set(agent, 'train_actor_critic', actor)
        old_epoch = agent.train_epoch

        def epoch(*args, **kwargs):
            before = dict(self.counts)
            start_frame, start_epoch, _ = clocks(self.plan)
            require(before['epoch_entries'] < self.plan['epochs'], 'Unexpected extra epoch')
            require(agent.frame == start_frame + before['epoch_returns'] * QUANTUM and
                    agent.epoch_num == start_epoch + before['epoch_returns'] + 1, 'Native epoch-entry clock differs')
            self.counts['epoch_entries'] += 1
            self.active = True
            started = time.perf_counter()
            self.emit('epoch_entered', scale=float(agent.scaler.get_scale()), lr=float(agent.last_lr))
            try:
                result = old_epoch(*args, **kwargs)
                self.counts['epoch_returns'] += 1
                wall = time.perf_counter() - started
                require(isinstance(result, tuple) and len(result) == 13, 'Unexpected native epoch return, including GA restart')
                for key, n in (('env_step_returns', 16), ('actor_attempts', 8), ('actor_returns', 8),
                               ('scheduler_calls', 2), ('scheduler_returns', 2), ('lr_returns', 2),
                               ('prepare_returns', 1), ('history_releases', 1)):
                    require(self.counts[key] - before[key] == n, 'Native epoch quota differs: ' + key)
                require(agent.curr_frames == QUANTUM, 'Native played-frame count differs')
                core = list(map(float, result[:4]))
                self.core_seconds += core[3]
                self.epoch_wall_seconds += wall
                self.emit('epoch_returned', original_step_seconds=core[0], original_play_seconds=core[1],
                          original_update_seconds=core[2], original_core_seconds=core[3],
                          observed_epoch_wall_seconds=wall, core_fps=QUANTUM / core[3],
                          epoch_wall_fps=QUANTUM / wall, physical_frames_from_returned_env_steps=QUANTUM,
                          frame_note='Native outer loop advances learner frame after this return',
                          expected_outer_frame=start_frame + self.counts['epoch_returns'] * QUANTUM,
                          scale=float(agent.scaler.get_scale()), lr=float(agent.last_lr),
                          delta_counts={k: self.counts[k] - before[k] for k in self.counts})
                self.resource('after_native_epoch_return')
                return result
            except BaseException as exc:
                self.remember(exc, 'train_epoch_or_post_return_observation')
                raise
            finally:
                self.active = False
        self.patches.set(agent, 'train_epoch', epoch)

    def summary(self):
        missing = self.counts['env_step_attempts'] - self.counts['env_step_returns']
        return dict(counts=dict(self.counts), physical_frames_returned_lower_bound=self.counts['env_step_returns'] * N,
                    env_step_attempts_without_return=missing,
                    unknown_additional_control_frames_upper_bound=missing * N,
                    incomplete_step_scope='A failed env_step may have advanced simulation before returning; internal partial physics unknown',
                    original_core_seconds=self.core_seconds, observed_epoch_wall_seconds=self.epoch_wall_seconds,
                    original_core_fps=(self.counts['epoch_returns'] * QUANTUM / self.core_seconds if self.core_seconds else None),
                    last_resource=self.last_resource)



def install_actor_loss(agent, plan, torch_runtime, actor_module, patches, native_actor_loss):
    require(agent.e_clip == .1 and agent.actor_loss_func is native_actor_loss, 'Original actor call differs')
    if plan['arm'] == 'candidate':
        def asymmetric(old, new, advantage, is_ppo, curr_e_clip):
            require(curr_e_clip == .1 and is_ppo, 'Runtime lower clip or PPO changed')
            return actor_module.actor_loss_asymmetric(torch_runtime, old, new, advantage,
                                                       is_ppo, curr_e_clip, plan['clip_high'])
        patches.set(agent, 'actor_loss_func', asymmetric)
    # Control keeps the original function object and original arithmetic.


def is_terminal_save(agent, filename, run_dir, name, plan):
    _, epoch, target = clocks(plan)
    path = Path(filename)
    prefix = 'last_' + str(name) + '_frame_' + str(target) + '_rew_'
    return (path.parent.resolve() == (run_dir / 'nn').resolve() and path.name.startswith(prefix)
            and agent.frame == target and agent.epoch_num == epoch + plan['epochs'])

def validate_config(cfg, plan):
    c = cfg.train.params.config
    start, _, target = clocks(plan)
    require(str(c.name).split('_')[0].isdigit() and cfg.get('full_experiment_name') in (None, '', c.name), 'Legal unique native name required')
    require((c.num_actors, c.horizon_length, c.seq_length, c.minibatch_size, c.mini_epochs) ==
            (N, 16, 16, 81920, 2), 'Fixed recurrent EPO layout differs')
    require(cfg.seed == cfg.train.params.seed == plan['seed'], 'Config seeds differ from plan')
    require(c.max_frames == target and c.max_epochs == -1 and 'score_to_win' not in c, 'Only the fixed frame budget may stop training')
    require(c.epo and c.expl_type == 'mixed_expl_learn_param' and c.use_others_experience == 'lf', 'Native LF EPO required')
    require((c.off_policy_actor_coef, c.off_policy_critic_coef, c.off_policy_support_mode) ==
            (1., 0., 'actor_clip'), 'Actor/critic coefficient or original support mask changed')
    require(c.e_clip == .1 and not c.get('use_smooth_clamp', False) and c.get('ppo', True), 'Original clip/lower/critic semantics changed')
    require(c.get('gradient_clip_scope', 'model') == 'model' and c.truncate_grads and c.mixed_precision,
            'Original model gradient clipping and AMP required')
    require(c.get('release_completed_rollout_history') is True and not c.get('release_cache_before_rollout', False), 'Only audited post-LF history release is allowed')
    require(c.schedule_type == 'standard' and c.lr_schedule == 'adaptive', 'Original scheduler required')
    require(c.expl_reward_coef == 0. and c.expl_coef_block_size == 320, 'Original exploration population differs')
    require(cfg.wandb_activate and cfg.wandb_mode == 'online' and cfg.wandb_resume == 'never', 'Fresh online W&B required')
    require(cfg.wandb_run_id in ('', plan['run_id']), 'Config contains another run ID')
    require(not cfg.test and not cfg.pbt.enabled and not cfg.multi_gpu and cfg.sigma in ('', None), 'Single native training/no sigma override required')
    require(cfg.get('torch_deterministic', False) is False, 'Native nondeterministic backend policy required')
    require(not (ROOT / 'runs' / c.name).exists(), 'Existing native run directory; no resume')
    if plan['mode'] == 'e51':
        require(str(cfg.checkpoint) == str(plan['checkpoint']) and identity(cfg.checkpoint)['sha256'] == E51_SHA, 'Wrong E51 checkpoint')
        require(not cfg.get('checkpoint_yaml'), 'No second configuration loading path allowed')
        if cfg.train.params.get('load_checkpoint', False):
            require(str(cfg.train.params.load_path) == str(cfg.checkpoint), 'Conflicting checkpoint paths')
    else:
        require(cfg.checkpoint in ('', None) and not cfg.get('checkpoint_yaml') and
                not cfg.train.params.get('load_checkpoint', False) and not cfg.train.params.get('load_path'), 'Fresh mode must not restore anything')
        require(c.learning_rate == 1e-4, 'Fresh learning rate differs')
    return start, target


def execute(plan, plan_path, output):
    import isaacgym  # Must precede Torch and project runtime imports.
    import torch
    from omegaconf import OmegaConf
    from isaacgymenvs import train as native
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from rl_games.common import common_losses
    from scripts.evaluate_frozen_checkpoint import source_snapshot
    from scripts.train_clipping_ablation import release_completed_rollout_history

    require(ROOT in Path(native.__file__).resolve().parents and
            ROOT in Path(sys.modules[A2CAgent.__module__].__file__).resolve().parents,
            'Imported numerical project modules are outside execution root')
    require(torch.__version__ == '1.8.1' and torch.version.cuda == '11.1', 'Unreviewed numerical runtime')
    cfg = OmegaConf.load(plan['config'])
    start, target = validate_config(cfg, plan)
    cfg.wandb_run_id = plan['run_id']
    config = cfg.train.params.config
    run_dir = ROOT / 'runs' / config.name
    finite = module('dapo_owned_finish_and_finite', plan['finite_source'])
    actor_module = module('dapo_bound_actor_loss', plan['actor_source'])
    nvml = module('dapo_nvml_readonly', plan['nvml_source']).NvmlMemory()
    snapshot = source_snapshot(output)
    with tarfile.open(str(output / 'source_snapshot.tar.gz'), 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(snapshot['files']) and {m.name for m in members} == set(snapshot['files']),
                'Source archive inventory differs')
        for member in members:
            require(member.isfile(), 'Nonregular source archive member')
            with archive.extractfile(member) as stream:
                sha = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            require(sha.hexdigest() == snapshot['files'][member.name], 'Source archive bytes differ: ' + member.name)
    archive_identity = identity(output / 'source_snapshot.tar.gz')
    (output / 'plan.json').write_bytes(plan_path.read_bytes())
    (output / 'config.resolved.yaml').write_text(OmegaConf.to_yaml(cfg))
    executing_sources = {'entry.py': str(Path(__file__).resolve()), 'base_source.py': plan['base_source'],
                         'actor_source.py': plan['actor_source'], 'finite_source.py': plan['finite_source'],
                         'nvml_source.py': plan['nvml_source'], 'config.input.yaml': plan['config']}
    sources = output / 'executing_sources'
    sources.mkdir()
    for name, path in executing_sources.items():
        shutil.copy2(path, sources / name)
    for name, path in plan.get('extra_plan_files', {}).items():
        require(not (output / name).exists(), 'Plan attachment collides with generated file')
        shutil.copy2(path, output / name)
    write(output / 'runtime.json', {'boundary': 'Before native Runner/set_seed; actual training flags recorded at train start',
          'torch': torch.__version__, 'cuda': torch.version.cuda,
          'cudnn_version': torch.backends.cudnn.version(), 'cudnn_benchmark': torch.backends.cudnn.benchmark,
          'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
          'python': sys.version, 'allocator_disabled': os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),
          'device_name': torch.cuda.get_device_name(0), 'source_file_count': len(snapshot['files'])})
    state = {'agent': None, 'observer': None, 'observations': None, 'restore_returned': False,
             'finish_entered': False, 'checkpoint': None, 'suppressed_saves': 0,
             'first': None, 'errors': [], 'config_capture': None}
    patches = Patches()

    def remember(exc, where):
        if state['first'] is None:
            state['first'] = (exc, exc.__traceback__)
        row = {'utc': base.now(), 'where': where, 'error': repr(exc),
               'primary_error': repr(state['first'][0])}
        state['errors'].append(row)
        try:
            with output.with_name(output.name + '-errors.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
                stream.flush()
        except BaseException as secondary:
            print('Secondary error-journal failure: ' + repr(secondary), file=sys.stderr)

    def source_check():
        require(identity(output / 'source_snapshot.tar.gz') == archive_identity, 'Sealed source archive changed')
        for name, sha in snapshot['files'].items():
            require(identity(ROOT / name)['sha256'] == sha, 'Executing repository source changed: ' + name)
        for path, sha in plan['files_sha256'].items():
            require(identity(path)['sha256'] == sha, 'Bound input changed: ' + path)
        for name, original in executing_sources.items():
            require(identity(sources / name)['sha256'] == plan['files_sha256'][original], 'Archived executing source changed')
        require((output / 'plan.json').read_bytes() == plan_path.read_bytes(), 'Plan changed')
        if state['config_capture'] is not None:
            for path, record in state['config_capture'].items():
                require(identity(path) == record, 'Executed native config changed: ' + path)
        if state['checkpoint'] is not None:
            require(identity(state['checkpoint']['path']) == state['checkpoint']['identity'], 'Terminal checkpoint changed')

    def capture_config(agent):
        require(agent.experiment_name == str(config.name), 'Constructed agent name differs')
        original, latest = run_dir / 'config.yaml', run_dir / 'config.latest.yaml'
        require(original.read_bytes() == latest.read_bytes(), 'Native config/latest differ')
        require(original.read_text() == OmegaConf.to_yaml(cfg), 'Native dumped config differs from executed cfg')
        copied = output / 'executed_training_config.yaml'
        shutil.copy2(original, copied)
        state['config_capture'] = {str(p): identity(p) for p in (original, latest, copied)}
        write(output / 'executed_training_config_manifest.json', {'files': state['config_capture'],
              'mode': plan['mode'], 'boundary': 'Native agent constructed; native config dumped before runner.run'})

    def init(agent, *args, **kwargs):
        try:
            old_init(agent, *args, **kwargs)
            require(state['agent'] is None, 'Multiple agents unsupported')
            state['agent'] = agent
            require(agent.is_rnn and not agent.multi_gpu and not agent.has_central_value and
                    agent.num_agents == 1 and agent.value_size == 1, 'Unexpected actual agent architecture')
            require((agent.num_actors, agent.horizon_length, agent.seq_length, agent.mini_epochs_num,
                     agent.minibatch_size) == (N, 16, 16, 2, 81920), 'Constructed layout differs')
            require(agent.e_clip == .1 and agent.actor_loss_func is common_losses.actor_loss, 'Original actor call differs')
            capture_config(agent)
            obs = Observations(agent, plan, output, nvml, remember, patches)
            state['observations'] = obs
            obs.install()
            install_actor_loss(agent, plan, torch, actor_module, patches, common_losses.actor_loss)
            old_augment = agent.augment_batch_for_mixed_expl
            def augment(batch, extras, *a, **kw):
                result = old_augment(batch, extras, *a, **kw)
                release_completed_rollout_history(extras, result)
                obs.counts['history_releases'] += 1
                return result
            patches.set(agent, 'augment_batch_for_mixed_expl', augment)
            old_train = agent.train
            def train(*a, **kw):
                try:
                    expected_start, expected_epoch, _ = clocks(plan)
                    require(agent.frame == expected_start and agent.epoch_num == expected_epoch, 'Training start clock differs')
                    backend = {'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                               'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
                               'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
                               'cudnn_deterministic': bool(torch.backends.cudnn.deterministic)}
                    require(backend == {'matmul_allow_tf32': True, 'cudnn_allow_tf32': True,
                                        'cudnn_benchmark': True, 'cudnn_deterministic': False},
                            'Actual post-native-construction backend flags differ')
                    gene = agent.model.a2c_network.extra_params
                    owned = any(gene is p for group in agent.optimizer.param_groups for p in group['params'])
                    if plan['mode'] == 'e51':
                        require(state['restore_returned'] and agent.scaler.get_scale() == 131072., 'Native E51 restore/AMP differs')
                        require(not owned, 'E51 optimizer must preserve native restored gene exclusion')
                    else:
                        require(not state['restore_returned'] and not agent.optimizer.state, 'Fresh optimizer must be new')
                        require(owned and gene.requires_grad and agent.scaler.get_scale() == 65536., 'Native fresh gene/AMP differs')
                    write(output / 'training_start.json', {'mode': plan['mode'], 'frame': int(agent.frame),
                          'epoch': int(agent.epoch_num), 'scale': float(agent.scaler.get_scale()),
                          'gene_owned': owned, 'gene_requires_grad': gene.requires_grad,
                          'optimizer_state_entries': len(agent.optimizer.state),
                          'optimizer_lrs': [float(g['lr']) for g in agent.optimizer.param_groups],
                          'restore_returned': state['restore_returned'], 'actual_backend_flags': backend,
                          'backend_boundary': 'After native Runner/set_seed/agent construction and optional full restore',
                          'scope': 'Native simulator created in this process; not historical trajectory replay'})
                    return old_train(*a, **kw)
                except BaseException as exc:
                    remember(exc, 'native_train')
                    raise
            patches.set(agent, 'train', train)
        except BaseException as exc:
            remember(exc, 'native_init_or_observation_install')
            raise

    def restore(agent, filename, *args, **kwargs):
        try:
            require(plan['mode'] == 'e51' and identity(filename)['sha256'] == E51_SHA and
                    Path(filename).resolve() == Path(plan['checkpoint']).resolve(), 'Unexpected checkpoint restoration')
            result = old_restore(agent, filename, *args, **kwargs)
            state['restore_returned'] = True
            require(agent.frame == 5000069120 and agent.epoch_num == 15259 and agent.scaler.get_scale() == 131072., 'Restored E51 clock/scale differs')
            write(output / 'restored.json', {'checkpoint': str(filename), 'sha256': E51_SHA,
                  'frame': int(agent.frame), 'epoch': int(agent.epoch_num), 'scale': float(agent.scaler.get_scale()),
                  'native_full_restore': True, 'scale_modified': False})
            return result
        except BaseException as exc:
            remember(exc, 'native_restore')
            raise

    def save(agent, filename, *args, **kwargs):
        path = Path(filename)
        allowed = is_terminal_save(agent, filename, run_dir, config.name, plan)
        if not allowed:
            state['suppressed_saves'] += 1
            return None
        try:
            require(state['checkpoint'] is None and not Path(str(path) + '.pth').exists(), 'Duplicate or mutable terminal checkpoint')
            result = old_save(agent, filename, *args, **kwargs)
            terminal = Path(str(path) + '.pth')
            state['checkpoint'] = {'path': str(terminal.resolve()), 'identity': identity(terminal),
                                   'frame': int(agent.frame), 'epoch': int(agent.epoch_num)}
            write(output / 'terminal_checkpoint.json', state['checkpoint'])
            return result
        except BaseException as exc:
            remember(exc, 'native_terminal_save')
            raise

    def before_init(observer, *args, **kwargs):
        state['observer'] = observer
        try:
            result = old_before_init(observer, *args, **kwargs)
            run = observer.run
            require(run is not None and str(run.id) == plan['run_id'] and not run.resumed and
                    not run.offline and run.settings.mode == 'online', 'Wrong/resumed/offline W&B run')
            proof = base.publish(run, {str(p.relative_to(output)): p for p in output.rglob('*') if p.is_file()}, 'dapo-training-plan')
            write(output / 'plan-artifact-wait.json', proof)
            run.summary.update({'study/purpose': plan['purpose'], 'study/mode': plan['mode'],
                                'study/arm': plan['arm'], 'study/actor_clip_high': plan['clip_high'],
                                'study/expected_physical_frames': plan['epochs'] * QUANTUM,
                                'study/seed': plan['seed'], 'study/checkpoint_selection_from_score': False})
            return result
        except BaseException as exc:
            remember(exc, 'online_before_init_and_plan_publication')
            raise

    def finish(observer, exit_code=0):
        if observer.finished:
            return
        require(not state['finish_entered'], 'Repeated finalization boundary')
        state['finish_entered'] = True
        active_error = sys.exc_info()[1]
        if exit_code and active_error is not None:
            remember(active_error, 'active_native_exception_before_seal')
        obs, agent = state['observations'], state['agent']
        summary = {'status': 'failed', 'mode': plan['mode'], 'arm': plan['arm'],
                   'native_exit_code': exit_code, 'planned_epochs': plan['epochs'], 'seed': plan['seed'],
                   'status_scope': 'Native quotas and closed payload validation before final upload/finish; parent actual wait and fresh API still required',
                   'start_frame': start, 'target_frame': target, 'terminal_checkpoint': state['checkpoint'],
                   'suppressed_saves': state['suppressed_saves'],
                   'finite_scope': 'Live model/RMS/actual Adam after original terminal save; no checkpoint reload',
                   'resource_scope': 'Post-return samples only, not continuous free-memory guarantee',
                   'timing_scope': 'Native core and observed epoch wall; process wait/monitor window separately owned by parent',
                   'intervention': 'Only actor upper .1->.2 for candidate; original critic/lower/off support/GA/LR unchanged'}
        if obs is not None:
            summary.update(obs.summary())
        try:
            if exit_code == 0 and state['first'] is None:
                require(obs is not None and state['checkpoint'] is not None, 'Missing completed observations/terminal checkpoint')
                for key, n in (('epoch_entries', 1), ('epoch_returns', 1), ('env_step_attempts', 16),
                               ('env_step_returns', 16), ('actor_attempts', 8), ('actor_returns', 8),
                               ('scheduler_calls', 2), ('scheduler_returns', 2), ('lr_calls', 2),
                               ('lr_returns', 2), ('prepare_returns', 1), ('history_releases', 1), ('resource_samples', 9)):
                    require(obs.counts[key] == plan['epochs'] * n, 'Terminal quota differs: ' + key)
                require(obs.counts['adam_entered'] == obs.counts['adam_returned'] and
                        obs.counts['adam_exceptions'] == 0 and obs.counts['adam_returned'] <= plan['epochs'] * 8,
                        'Actual Adam return accounting differs')
                require(agent.frame == target and agent.epoch_num == clocks(plan)[1] + plan['epochs'], 'Terminal learner clock differs')
                saved = [p.resolve() for p in run_dir.rglob('*.pth') if p.is_file() and not p.is_symlink()]
                require(saved == [Path(state['checkpoint']['path'])], 'Expected exactly one original terminal checkpoint file')
                finite.terminal_finite(agent, output / 'terminal-finite.json')
                summary['physical_frames'] = plan['epochs'] * QUANTUM
                summary['actual_adam_updates'] = obs.counts['adam_returned']
                summary['completed_epochs'] = obs.counts['epoch_returns']
            source_check()
        except BaseException as exc:
            remember(exc, 'terminal_validation_finite_or_sources')
        writer = getattr(observer, 'writer', None) or getattr(agent, 'writer', None)
        writer_closed = writer is None
        if writer is not None:
            for method in ('flush', 'close'):
                try:
                    getattr(writer, method)()
                    if method == 'close':
                        writer_closed = True
                except BaseException as exc:
                    remember(exc, 'writer_' + method)
        if writer_closed:
            observer.writer = None
        if obs is not None:
            try:
                obs.stream.close()
            except BaseException as exc:
                remember(exc, 'telemetry_close')
        effective = exit_code or (1 if state['first'] is not None else 0)
        summary.update(status='completed' if effective == 0 else 'failed', effective_exit_code=effective,
                       errors=list(state['errors']), primary_error=repr(state['first'][0]) if state['first'] else None,
                       native_writer_closed=writer_closed,
                       failed_raw_scope='Closed files uploaded; unconfirmed-open TB retained locally for post-exit parent archive')
        try:
            write(output / 'summary.json', summary)
            files = {str(p.relative_to(output)): p for p in output.rglob('*') if p.is_file() and not p.is_symlink()
                     and (obs is None or obs.stream.closed or p != output / 'training.jsonl')}
            for p in run_dir.rglob('*'):
                relative = p.relative_to(run_dir)
                if not p.is_file() or p.is_symlink() or 'wandb' in relative.parts or p.name == 'wandb_run.json':
                    continue
                if not writer_closed and 'summaries' in relative.parts:
                    continue
                files['native/' + str(relative)] = p
            proof = base.publish(observer.run, files, 'dapo-training-result')
            write(output / 'result-artifact-wait.json', proof)
            source_check()
            observer.run.summary.update({'study/status': summary['status'], 'study/physical_frames': summary.get('physical_frames'),
                                         'study/actual_adam_updates': summary.get('actual_adam_updates'),
                                         'study/result_file_count': len(proof['files']),
                                         'study/terminal_checkpoint_sha256': state['checkpoint']['identity']['sha256'] if state['checkpoint'] else None})
        except BaseException as exc:
            remember(exc, 'closed_raw_publication')
        finally:
            effective = exit_code or (1 if state['first'] is not None else 0)
            finite.finish_original_observer(observer, old_finish, plan['run_id'], effective, remember, output)
            if writer_closed:
                observer.writer = writer
        if state['first'] is not None:
            exc, tb = state['first']
            raise exc.with_traceback(tb)

    native_error = None
    try:
        old_init = patches.set(A2CAgent, '__init__', init)
        old_restore = patches.set(A2CAgent, 'restore', restore)
        old_save = patches.set(A2CAgent, 'save', save)
        old_before_init = patches.set(WandbAlgoObserver, 'before_init', before_init)
        old_finish = patches.set(WandbAlgoObserver, 'finish', finish)
        native.launch_rlg_hydra(cfg)
        require(state['finish_entered'] and state['first'] is None, 'Native training did not complete finalization')
    except BaseException as exc:
        native_error = exc
        remember(exc, 'native_launch')
        observer = state['observer']
        if observer is not None and not observer.finished and not state['finish_entered']:
            try:
                finish(observer, exit_code=1)
            except BaseException as secondary:
                remember(secondary, 'secondary_finish')
        if state['first'] is not None:
            original, tb = state['first']
            raise original.with_traceback(tb)
        raise
    finally:
        if state['observations'] is not None:
            try:
                state['observations'].stream.close()
            except BaseException as exc:
                remember(exc, 'cleanup_telemetry_close')
        patches.restore(remember)
        if native_error is None and state['first'] is not None:
            exc, tb = state['first']
            raise exc.with_traceback(tb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    require(hashlib.sha256(plan_bytes).hexdigest() == args.plan_sha256, 'Plan SHA differs')
    plan = json.loads(plan_bytes)
    configure_execution_root(plan)
    prepared = validate_plan(plan, args.plan)
    if not args.execute:
        print(json.dumps(prepared, indent=2, sort_keys=True))
        return
    require(plan['execution_ready'] is True, 'Plan is not explicitly bound for execution')
    require(Path.cwd() == ROOT and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Wrong cwd/allocator/visible GPU')
    require(shutil.disk_usage(ROOT).free >= 4 * 2**30, 'Insufficient startup disk reserve')
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if len(line.split()) == 3}
    require(memory['MemAvailable'] >= 16 * 2**30, 'Insufficient startup host RAM reserve')
    with open('/tmp/epo-gpu-0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        require(not active, 'GPU compute process is active')
        output = Path(plan['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        execute(plan, args.plan, output)


if __name__ == '__main__':
    main()
