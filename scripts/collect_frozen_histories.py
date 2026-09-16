#!/usr/bin/env python3
"""Collect fixed-length CUDA histories for conditional clipped-command diagnostics.

Sampling uses only the original behavior forward; shadow policies are replayed
afterward in a separate process so their construction/forward cannot consume the
behavior sampler's RNG. This is not an episode-performance evaluation.
"""
import argparse
import copy
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def bootstrap(repo):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'rl_games'))
    import isaacgym  # noqa: F401; must precede torch
    import torch
    import isaacgymenvs  # noqa: F401; registers OmegaConf resolvers
    from isaacgymenvs.train import configure_torch_jit_for_modern_gpus
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    configure_torch_jit_for_modern_gpus()
    return torch


def load_policy(checkpoint_path, original_cfg, obs_dim, actions_num, num_envs, device):
    import torch
    from rl_games.algos_torch.model_builder import ModelBuilder
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state = checkpoint[0] if 0 in checkpoint else checkpoint
    model_state = state['model']
    params = copy.deepcopy(original_cfg['train']['params'])
    if params['model']['name'] != 'continuous_a2c_logstd':
        raise ValueError('Only the audited continuous_a2c_logstd model is supported')
    if params['network']['space']['continuous']['fixed_sigma'] != 'coef_cond':
        raise ValueError('Only coefficient-conditioned sigma is supported')
    if not params['config'].get('zero_rnn_on_done', True):
        raise ValueError('The stored history protocol requires zero_rnn_on_done=True')
    if params['config']['expl_type'] != 'mixed_expl_learn_param':
        raise ValueError('Only extra_param EPO checkpoints are supported')
    genes, latent_dim = model_state['a2c_network.extra_params'].shape
    if model_state['a2c_network.sigma'].shape[0] != genes:
        raise ValueError('Gene and sigma rows differ')
    if int(original_cfg['task']['env']['numEnvs']) // int(params['config']['expl_coef_block_size']) != genes:
        raise ValueError('Saved environment/gene layout differs from checkpoint')
    # Reproduce the saved training identifier grid exactly.
    coef_ids = torch.linspace(50.0, 0.0, genes).to(device)
    model = ModelBuilder().load(params).build({
        'actions_num': actions_num, 'input_shape': (obs_dim + 1,), 'num_seqs': num_envs,
        'value_size': 1, 'normalize_value': params['config'].get('normalize_value', False),
        'normalize_input': params['config'].get('normalize_input', False),
        'type': 'extra_param', 'param_size': latent_dim, 'coef_ids': coef_ids, 'coef_id_idx': obs_dim,
    }).to(device)
    model.load_state_dict(model_state, strict=True)
    if params['config'].get('normalize_input') and 'running_mean_std' in state:
        model.running_mean_std.load_state_dict(state['running_mean_std'], strict=True)
    model.eval()
    model.requires_grad_(False)
    if not model.is_rnn():
        raise ValueError('This history protocol requires a recurrent policy')
    origin = {'checkpoint_frame': int(state.get('frame', -1)),
              'checkpoint_epoch': int(state.get('epoch', -1)),
              'gene_count': int(genes), 'latent_dim': int(latent_dim)}
    del checkpoint, state, model_state
    torch.cuda.empty_cache()
    return model, coef_ids, origin


def identifiers_for(model, coef_ids, assigned_genes):
    import torch
    identifiers = coef_ids[assigned_genes].unsqueeze(1)
    if not torch.all((identifiers == coef_ids).sum(dim=1) == 1).item():
        raise RuntimeError('Ambiguous gene identifier')
    for grid in (model.a2c_network.param_ids, model.a2c_network.sigma_ids):
        decoded = (identifiers == grid).float().argmax(dim=1)
        if not torch.equal(decoded, assigned_genes):
            raise RuntimeError('Gene/sigma lookup differs from requested assignment')
    return identifiers


def check_buffers(model, original):
    import torch
    for name, value in model.named_buffers():
        if not torch.equal(value, original[name]):
            raise RuntimeError('Frozen policy buffer changed: ' + name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=Path('/home/ubuntu22/EPO'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval-seed', type=int, required=True)
    parser.add_argument('--panel-seed', type=int, default=20260914)
    parser.add_argument('--objects-per-behavior', type=int, default=32)
    parser.add_argument('--behavior-genes', default='0,61,63')
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb-entity', default='leo-ucbm-university-of-california-berkeley')
    parser.add_argument('--wandb-project', default='EPO-reproduction')
    parser.add_argument('--wandb-group', default='executed-command-diagnostic-20260913')
    parser.add_argument('--gpu-lock', default='/tmp/epo-gpu-0.lock')
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('This one-GPU resource protocol requires cuda:0')
    if args.steps <= 0 or not 1 <= args.objects_per_behavior <= 64:
        parser.error('steps must be positive and objects-per-behavior in [1,64]')
    return args


def run(args):
    torch = bootstrap(args.repo_dir.resolve())
    import numpy as np
    import wandb
    from omegaconf import OmegaConf
    from isaacgymenvs.utils.utils import set_seed
    from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
    from scripts.evaluate_frozen_checkpoint import object_catalog, sha256_file, source_snapshot, write_json
    torch.cuda.set_device(0)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite collected histories: ' + str(output))
    if shutil.disk_usage(str(args.repo_dir)).free < 2 * 2**30:
        raise RuntimeError('At least 2 GiB free disk is required')
    cfg = OmegaConf.to_container(OmegaConf.load(str(args.train_config)), resolve=True)
    if cfg['task']['name'] != 'AllegroKuka' or cfg['task']['env']['subtask'] != 'reorientation':
        raise ValueError('Only AllegroKuka reorientation has been audited')
    if cfg['task']['task']['randomize']:
        raise ValueError('Scheduled domain randomization requires another protocol')
    if not cfg['train']['params']['config'].get('clip_actions', True):
        raise ValueError('Clipped Gaussian diagnostics require the saved clip_actions=True map')
    genes = [int(item) for item in args.behavior_genes.split(',')]
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError('Behavior genes must be nonempty and unique')
    catalog = object_catalog(AllegroKukaReorientation, cfg['task']['env'])
    indices = sorted(np.random.default_rng(args.panel_seed).choice(
        len(catalog), size=args.objects_per_behavior, replace=False).tolist())
    panel = [catalog[index] for index in indices]
    count = len(panel) * len(genes)
    if count > 192:
        raise ValueError('The bounded history diagnostic permits at most 192 environments')
    task_cfg = copy.deepcopy(cfg['task'])
    task_cfg['env'].update(numEnvs=count, successTolerance=0.01, targetSuccessTolerance=0.01,
        toleranceCurriculumIncrement=1.0, toleranceCurriculumInterval=args.steps + 1,
        goodResetBoundary=0, evalStats=False, saveStates=False, loadInitialStates=False, enableDebugVis=False)
    task_cfg['sim']['use_gpu_pipeline'] = True
    task_cfg['sim']['physx']['use_gpu'] = True
    output.mkdir(parents=True)
    source = source_snapshot(output)
    # This script may be staged outside the repo while paired training is frozen.
    shutil.copyfile(__file__, str(output / 'collector_source.py'))
    write_json(output / 'training_config.resolved.json', cfg)
    write_json(output / 'task_config.diagnostic.json', task_cfg)
    write_json(output / 'object_panel.json', {'catalog_size': len(catalog), 'objects': panel})
    checkpoint_hash = sha256_file(args.checkpoint)
    protocol = {
        'protocol_version': 'fixed_history_collection_v1',
        'interpretation': 'Conditional history diagnostic, not endpoint performance or independent training-seed evidence',
        'training_benefit_tested': False, 'actual_scheduler_reproduced': False,
        'checkpoint_path': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'training_config_sha256': sha256_file(args.train_config), 'training_seed': cfg['seed'],
        'source_sha256': source['source_sha256'], 'collector_source_sha256': sha256_file(Path(__file__)),
        'args': {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'behavior_genes': genes, 'num_envs': count, 'steps': args.steps,
        'planned_physical_frames': count * args.steps,
        'object_assignment': 'object=env_id%panel_size; behavior=genes[env_id//panel_size]',
        'statistical_unit': 'fixed-length environment history; includes resets, not first-completed-episode selection',
        'initialization': 'fresh task, zero first observation, deferred first-step physical reset as in audited evaluator',
        'rnn': 'frozen behavior from zero hidden; dones_after_step resets state before the next forward',
        'shadow_execution': 'none during physical sampling; replay in a separate process',
        'action_mapping': 'raw Gaussian action clipped coordinatewise to [-1,1]; exact map checked against environment',
        'normalizer': 'eval, frozen buffers checked after sampling',
        'storage': 'all raw observations/actions/mu/sigma/commands/reset flags, full behavior hidden input and output at each step, sampled forces, final observation',
        'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'git_commit': source['git_commit'],
    }
    write_json(output / 'protocol.json', protocol)
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
        job_type='diagnostic', name=output.name, dir=str(output), config=protocol,
        mode='online', resume='never', tags=['fixed_history', 'recurrent', 'gpu_only', 'no_training_gain_claim'])
    metadata = {'id': run.id, 'entity': run.entity, 'project': run.project,
                'url': run.url, 'state': 'running'}
    write_json(output / 'wandb_run.json', metadata)
    env = None
    try:
        if run.offline or run.settings.mode != 'online' or run.resumed:
            raise RuntimeError('Fresh online W&B recording is required')

        class HistoryTask(AllegroKukaReorientation):
            def _main_object_assets_and_scales(self, root, folder):
                files, scales = super()._main_object_assets_and_scales(root, folder)
                if [Path(path).name for path in files] != [item['asset_name'] for item in catalog]:
                    raise RuntimeError('Object generator order changed')
                return [files[index] for index in indices], [scales[index] for index in indices]

            def _extra_curriculum(self):
                if self.success_tolerance != 0.01:
                    raise RuntimeError('Frozen diagnostic tolerance changed')

        set_seed(args.eval_seed, torch_deterministic=bool(cfg.get('torch_deterministic', False)))
        env = HistoryTask(cfg=task_cfg, rl_device=args.device, sim_device=args.device,
            graphics_device_id=0, headless=True, virtual_screen_capture=False, force_render=False)
        if not str(env.device).startswith('cuda') or not env.sim_params.use_gpu_pipeline or not env.sim_params.physx.use_gpu:
            raise RuntimeError('CUDA PhysX and GPU pipeline are required')
        low = torch.as_tensor(env.action_space.low, device=args.device)
        high = torch.as_tensor(env.action_space.high, device=args.device)
        if not torch.all(low == -1).item() or not torch.all(high == 1).item() or float(env.clip_actions) < 1.0:
            raise ValueError('The executed-command mapping differs from clip(raw,-1,1)')
        if env.num_envs != count or len(env.object_asset_files) != len(panel):
            raise RuntimeError('Environment/object quota mismatch')
        model, coef_ids, origin = load_policy(args.checkpoint, cfg, env.observation_space.shape[0],
                                             env.action_space.shape[0], count, args.device)
        if min(genes) < 0 or max(genes) >= origin['gene_count']:
            raise ValueError('Invalid behavior gene')
        assigned = torch.tensor(genes, device=args.device).repeat_interleave(len(panel))
        identifiers = identifiers_for(model, coef_ids, assigned)
        buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
        states = tuple(state.to(args.device) for state in model.get_default_rnn_state())
        if any(torch.count_nonzero(state).item() for state in states):
            raise RuntimeError('History replay requires a zero recurrent initial state')
        obs = env.reset()['obs']
        reset_before = torch.zeros(count, dtype=torch.bool, device=args.device)
        episode_id = torch.zeros(count, dtype=torch.int64, device=args.device)
        within_episode = torch.zeros(count, dtype=torch.int64, device=args.device)
        fields = ('obs', 'mu', 'sigma', 'raw_action', 'command', 'behavior_neglogp', 'done_after',
                  'reset_before', 'episode_id', 'within_episode_step', 'rb_forces_after', 'random_force_prob_after')
        rows = {key: [] for key in fields}
        hidden_in, hidden_out = [], []
        rng_at_start = torch.cuda.get_rng_state().cpu()
        started = time.monotonic()
        last_report = started
        with torch.no_grad():
            for step in range(args.steps):
                rows['obs'].append(obs.detach().cpu().clone())
                rows['reset_before'].append(reset_before.cpu().clone())
                rows['episode_id'].append(episode_id.cpu().clone())
                rows['within_episode_step'].append(within_episode.cpu().clone())
                hidden_in.append(tuple(state.detach().cpu().clone() for state in states))
                result = model({'is_train': False, 'prev_actions': None,
                                'obs': torch.cat([obs, identifiers], dim=1), 'rnn_states': states})
                states = result['rnn_states']
                hidden_out.append(tuple(state.detach().cpu().clone() for state in states))
                raw = result['actions']
                if not all(torch.isfinite(result[key]).all().item() for key in ('actions', 'mus', 'sigmas')):
                    raise RuntimeError('Nonfinite behavior distribution/action')
                if not torch.all(result['sigmas'] > 0).item():
                    raise RuntimeError('Behavior sigma must be positive')
                command = raw.clamp(-1, 1)
                for key, tensor in [('mu', result['mus']), ('sigma', result['sigmas']),
                                    ('raw_action', raw), ('command', command),
                                    ('behavior_neglogp', result['neglogpacs'])]:
                    rows[key].append(tensor.detach().cpu().clone())
                obs_dict, _, dones, _ = env.step(command)
                if not torch.equal(env.actions, command):
                    raise RuntimeError('Task consumed a different command from the recorded clipping map')
                obs = obs_dict['obs']
                reset_before = dones.bool()
                rows['done_after'].append(reset_before.cpu().clone())
                rows['rb_forces_after'].append(env.rb_forces.detach().cpu().clone())
                rows['random_force_prob_after'].append(env.random_force_prob.detach().cpu().clone())
                episode_id += reset_before.long()
                within_episode = torch.where(reset_before, torch.zeros_like(within_episode), within_episode + 1)
                for state in states:
                    state[:, reset_before, :] = 0
                now = time.monotonic()
                if now - last_report >= 30 or step + 1 == args.steps:
                    run.log({'history/steps': step + 1, 'history/physical_frames': (step + 1) * count,
                             'history/wall_seconds': now - started})
                    print('history steps={} frames={} elapsed={:.2f}s'.format(step + 1, (step + 1) * count, now - started), flush=True)
                    last_report = now
        check_buffers(model, buffers)
        # CPU operations below only serialize already-computed CUDA data.
        dataset = {key: torch.stack(values) for key, values in rows.items()}
        dataset['hidden_in'] = tuple(torch.stack([row[index] for row in hidden_in]) for index in range(len(states)))
        dataset['hidden_out'] = tuple(torch.stack([row[index] for row in hidden_out]) for index in range(len(states)))
        dataset.update(assigned_genes=assigned.cpu(), gene_identifiers=identifiers.cpu(),
            object_indices=torch.tensor([item['original_object_index'] for item in panel] * len(genes)),
            final_obs=obs.detach().cpu(), rng_at_start=rng_at_start,
            rng_at_finish=torch.cuda.get_rng_state().cpu())
        data_path = output / 'histories.pt.gz'
        with gzip.open(str(data_path), 'wb', compresslevel=1) as stream:
            torch.save(dataset, stream)
        summary = {**origin, 'status': 'completed', 'normalizer_unchanged': True,
            'physical_frames': count * args.steps, 'steps': args.steps, 'num_envs': count,
            'wall_seconds': time.monotonic() - started, 'histories_sha256': sha256_file(data_path),
            'histories_bytes': data_path.stat().st_size,
            'training_benefit_tested': False, 'actual_scheduler_reproduced': False}
        write_json(output / 'summary.json', summary)
        run.summary.update(summary)
        artifact = wandb.Artifact('fixed-history-' + run.id, type='fixed-history',
            metadata={'checkpoint_sha256': checkpoint_hash, 'histories_sha256': summary['histories_sha256']})
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'wandb_run.json':
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact).wait()
        run.finish(exit_code=0)
        metadata.update(state='finished', exit_code=0)
        write_json(output / 'wandb_run.json', metadata)
    except BaseException as error:
        metadata.update(state='failed', exit_code=1,
                        error={'type': type(error).__name__, 'message': str(error)})
        write_json(output / 'wandb_run.json', metadata)
        summary_path = output / 'summary.json'
        failed_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        failed_summary.update(status='failed', error=metadata['error'])
        write_json(summary_path, failed_summary)
        try:
            run.summary.update({'status': 'failed'})
            run.finish(exit_code=1)
        except BaseException as finalization_error:
            metadata['finalization_error'] = {'type': type(finalization_error).__name__,
                                              'message': str(finalization_error)}
            write_json(output / 'wandb_run.json', metadata)
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


def main():
    args = parse_args()
    with open(args.gpu_lock, 'a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError('Another GPU computation is active: ' + active)
        run(args)


if __name__ == '__main__':
    main()
