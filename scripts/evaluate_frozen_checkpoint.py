#!/usr/bin/env python3
"""Online, CUDA-only first-episode evaluation of an EPO reorientation checkpoint.

This is a diagnostic conditional on one training seed. Environment replications
are not independent training runs and are never reported as training-seed CIs.
No player, optimizer, training observer, or checkpoint environment state is used.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-config', required=True)
    parser.add_argument('--output-dir', required=True, help='A new directory; existing directories are refused.')
    parser.add_argument('--genes', default='master', help='master (=K-1), all, or comma-separated row indices.')
    parser.add_argument('--objects-per-gene', type=int, default=16,
                        help='Shared random object panel size; 0 evaluates all generated objects.')
    parser.add_argument('--replicas-per-object', type=int, default=1)
    parser.add_argument('--object-seed', type=int, default=20260913)
    parser.add_argument('--eval-seed', type=int, default=10001)
    parser.add_argument('--max-steps', type=int, default=30000,
                        help='Administrative censoring horizon, separate from task timeout.')
    parser.add_argument('--max-envs', type=int, default=4096)
    parser.add_argument('--tolerance', type=float, default=0.01)
    parser.add_argument('--action-mode', choices=['stochastic', 'deterministic'], default='stochastic')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb-entity', default=os.getenv('WANDB_ENTITY'))
    parser.add_argument('--wandb-project', default=os.getenv('WANDB_PROJECT', 'EPO-reproduction'))
    parser.add_argument('--wandb-group', default='frozen_checkpoint_diagnostics_20260913')
    parser.add_argument('--source-run-url', default='')
    args = parser.parse_args()
    if not args.device.startswith('cuda:'):
        parser.error('--device must explicitly select cuda:N; CPU fallback is forbidden')
    if min(args.replicas_per_object, args.max_steps, args.max_envs) <= 0:
        parser.error('replicas, max-steps, and max-envs must be positive')
    if args.objects_per_gene < 0 or args.tolerance <= 0:
        parser.error('objects-per-gene must be nonnegative and tolerance positive')
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_snapshot(output):
    """Capture code actually on disk, including untracked evaluator/test files."""
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=str(ROOT)
    ).decode().split('\0')
    suffixes = {'.py', '.yaml', '.yml', '.sh', '.toml', '.template'}
    generated_roots = {'artifacts', 'runs', 'logs', 'result', 'wandb', '.git', '.pytest_cache'}
    names = sorted({name for name in tracked
                    if name and Path(name).parts[0] not in generated_roots
                    and output not in (ROOT / name).resolve().parents
                    and (ROOT / name).is_file() and Path(name).suffix in suffixes})
    hashes = {name: sha256_file(ROOT / name) for name in names}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip(),
        'source_sha256': source_hash,
        'files': hashes,
    }
    write_json(output / 'source_manifest.json', manifest)
    with (output / 'source_diff.patch').open('wb') as stream:
        subprocess.run(['git', 'diff', 'HEAD', '--binary', '--no-ext-diff'],
                       cwd=str(ROOT), stdout=stream, check=True)
    with tarfile.open(str(output / 'source_snapshot.tar.gz'), 'w:gz') as archive:
        for name in names:
            archive.add(str(ROOT / name), arcname=name)
    return manifest


def select_genes(specification, count):
    if specification == 'master':
        genes = [count - 1]
    elif specification == 'all':
        genes = list(range(count))
    else:
        genes = [int(item) for item in specification.split(',')]
    if not genes or len(set(genes)) != len(genes) or min(genes) < 0 or max(genes) >= count:
        raise ValueError('Gene indices must be unique and in [0, K)')
    return genes


def object_catalog(task_class, env_config):
    """Use the task's exact asset generator without creating a simulator."""
    probe = SimpleNamespace(
        object_type=env_config['objectType'],
        asset_files_dict={'block': env_config['asset']['assetFileNameBlock']},
        object_base_size=env_config['objectBaseSize'],
        with_small_cuboids=env_config['withSmallCuboids'],
        with_big_cuboids=env_config['withBigCuboids'],
        with_sticks=env_config['withSticks'],
        randomize_object_dimensions=env_config['randomizeObjectDimensions'],
    )
    probe._box_asset_files_and_scales = lambda root, folder: task_class._box_asset_files_and_scales(probe, root, folder)
    with tempfile.TemporaryDirectory(prefix='epo_eval_objects_') as folder:
        files, scales = task_class._main_object_assets_and_scales(probe, str(ROOT / 'assets'), folder)
        return [{'original_object_index': index, 'asset_name': Path(path).name,
                 'scale': list(scale), 'asset_sha256': sha256_file(path)}
                for index, (path, scale) in enumerate(zip(files, scales))]


def summarize(rows, genes, success_cap):
    """Bounds retain all assigned episodes; completed-only averages are labeled."""
    summaries = {}
    for gene in genes:
        selected = [row for row in rows if row['gene_id'] == gene]
        completed = [row for row in selected if not row['censored']]
        values = [row['successes_observed'] for row in selected]
        unresolved = sum(row['censored'] for row in selected)
        summaries[str(gene)] = {
            'assigned_episodes': len(selected),
            'completed_episodes': len(completed),
            'censored_episodes': unresolved,
            'censor_fraction': unresolved / len(selected),
            'mean_successes_lower_bound': sum(values) / len(selected),
            'mean_successes_upper_bound': sum(
                success_cap if row['censored'] else row['successes_observed'] for row in selected
            ) / len(selected),
            'completed_only_mean_successes': (
                sum(row['successes_observed'] for row in completed) / len(completed) if completed else None
            ),
            'cap_fraction_lower_bound': sum(value >= success_cap for value in values) / len(selected),
            'mean_successes': sum(values) / len(selected) if not unresolved else None,
        }
    return summaries


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite evaluation directory: ' + str(output))
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.train_config).resolve()
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError('Both checkpoint and training config must exist')
    os.chdir(str(ROOT))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'rl_games'))

    # Isaac Gym's binary bindings require this import order.
    import isaacgym  # noqa: F401
    import torch
    import numpy as np
    import wandb
    import isaacgymenvs  # noqa: F401; registers the saved config's OmegaConf resolvers
    from omegaconf import OmegaConf
    from isaacgymenvs.train import configure_torch_jit_for_modern_gpus
    from isaacgymenvs.utils.utils import set_seed
    from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
    from rl_games.algos_torch.model_builder import ModelBuilder

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
    torch.cuda.set_device(torch.device(args.device))
    configure_torch_jit_for_modern_gpus()
    original_cfg = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    if original_cfg['task']['env']['subtask'] != 'reorientation' or original_cfg['task']['name'] != 'AllegroKuka':
        raise ValueError('This evaluator is validated only for AllegroKuka reorientation')
    params = copy.deepcopy(original_cfg['train']['params'])
    if params['config']['expl_type'] != 'mixed_expl_learn_param':
        raise ValueError('Only mixed_expl_learn_param checkpoints are supported')
    if original_cfg['task']['task']['randomize']:
        raise ValueError('Scheduled domain randomization needs a separate fixed-distribution protocol')

    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=args.device)
    state = checkpoint[0] if 0 in checkpoint else checkpoint
    model_state = state['model']
    extra_params = model_state['a2c_network.extra_params']
    gene_count, latent_dim = tuple(extra_params.shape)
    genes = select_genes(args.genes, gene_count)
    train_envs = int(original_cfg['task']['env']['numEnvs'])
    train_block = int(params['config']['expl_coef_block_size'])
    if train_envs % train_block or train_envs // train_block != gene_count:
        raise ValueError('Training config gene count does not match checkpoint extra_params')
    if model_state['a2c_network.sigma'].shape[0] != gene_count:
        raise ValueError('Checkpoint sigma rows do not match gene count')
    checkpoint_frame = int(state.get('frame', -1))
    checkpoint_epoch = int(state.get('epoch', -1))

    catalog = object_catalog(AllegroKukaReorientation, original_cfg['task']['env'])
    panel_count = args.objects_per_gene or len(catalog)
    if panel_count > len(catalog):
        raise ValueError('objects-per-gene exceeds available object catalog')
    object_indices = sorted(np.random.default_rng(args.object_seed).choice(
        len(catalog), size=panel_count, replace=False).tolist())
    panel = [catalog[index] for index in object_indices]
    num_envs = panel_count * len(genes) * args.replicas_per_object
    if num_envs > args.max_envs:
        raise ValueError('Requested {} environments exceed --max-envs {}; reduce genes or panel size'.format(
            num_envs, args.max_envs))
    task_cfg = copy.deepcopy(original_cfg['task'])
    task_cfg['env'].update({
        'numEnvs': num_envs, 'successTolerance': args.tolerance,
        'targetSuccessTolerance': args.tolerance, 'toleranceCurriculumIncrement': 1.0,
        'toleranceCurriculumInterval': args.max_steps + 1,
        'goodResetBoundary': 0, 'evalStats': False, 'saveStates': False,
        'loadInitialStates': False, 'enableDebugVis': False,
    })
    task_cfg['sim']['use_gpu_pipeline'] = True
    task_cfg['sim']['physx']['use_gpu'] = True
    success_cap = int(task_cfg['env']['maxConsecutiveSuccesses'])
    if success_cap <= 0:
        raise ValueError('Finite success cap required to report censoring bounds')

    output.mkdir(parents=True)
    source = source_snapshot(output)
    write_json(output / 'training_config.resolved.json', original_cfg)
    write_json(output / 'task_config.evaluation.json', task_cfg)
    write_json(output / 'object_panel.json', {'catalog_size': len(catalog), 'objects': panel})
    protocol = {
        'protocol_version': 'frozen_first_episode_v1.1',
        'statistical_scope': 'diagnostic conditional on one checkpoint and one training seed; no training-seed CI',
        'object_distribution': 'fixed shared subset without replacement' if panel_count < len(catalog) else 'full generated object catalog',
        'assignment': 'gene=(env_id//panel_size)%selected_gene_count; object=env_id%panel_size',
        'rng_pairing': 'equal object quotas; trajectories are independent draws, not exact common random numbers',
        'initialization': 'fresh task; zero initial observation and deferred first-step reset as in training',
        'episode_selection': 'exactly first episode per assigned environment; later episodes ignored',
        'termination': 'task done completes endpoint; administrative max_steps produces censoring',
        'task_timeout_interpretation': 'part of the benchmark episode endpoint, reported separately from administrative censoring',
        'no_checkpoint_environment_restore': True,
        'curriculum_disabled': True,
        'normalizer_frozen': True,
        'action_clipping_measurement': 'per-episode fraction of executed action coordinates and steps changed by policy or simulator clipping',
        'gene_count': gene_count, 'latent_dim': latent_dim, 'selected_genes': genes,
        'num_envs': num_envs, 'success_cap': success_cap,
        'training_seed': original_cfg['seed'], 'checkpoint_frame': checkpoint_frame,
        'checkpoint_epoch': checkpoint_epoch, 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_path': str(checkpoint_path), 'training_config_sha256': sha256_file(config_path),
        'source_sha256': source['source_sha256'], 'git_commit': source['git_commit'],
        'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'wandb_version': wandb.__version__, 'gpu': torch.cuda.get_device_name(),
        'args': vars(args),
    }
    write_json(output / 'protocol.json', protocol)
    run = wandb.init(
        entity=args.wandb_entity or original_cfg.get('wandb_entity') or None,
        project=args.wandb_project, group=args.wandb_group, job_type='evaluation',
        name=output.name, tags=['frozen_checkpoint', 'first_episode', 'conditional_diagnostic', 'gpu_only'],
        notes='No training-seed uncertainty claim. ' + args.source_run_url,
        config=protocol, dir=str(output), mode='online', force=True, resume='never',
        settings=wandb.Settings(init_timeout=120),
    )
    metadata = {'id': run.id, 'entity': run.entity, 'project': run.project, 'url': run.url,
                'state': 'running', 'mode': 'online'}
    write_json(output / 'wandb_run.json', metadata)
    print('W&B evaluation:', run.url, flush=True)
    env = None
    try:
        if run.settings.mode != 'online' or run.offline:
            raise RuntimeError('W&B did not start in strict online mode')

        class EvaluationTask(AllegroKukaReorientation):
            def _main_object_assets_and_scales(self, root, folder):
                files, scales = super()._main_object_assets_and_scales(root, folder)
                if [Path(path).name for path in files] != [item['asset_name'] for item in catalog]:
                    raise RuntimeError('Object generator order changed during evaluation construction')
                return ([files[index] for index in object_indices], [scales[index] for index in object_indices])

            def _extra_curriculum(self):
                # Evaluation-only subclass; training task and reward/reset rules are untouched.
                if self.success_tolerance != args.tolerance:
                    raise RuntimeError('Frozen evaluation tolerance changed')

        set_seed(args.eval_seed, torch_deterministic=bool(original_cfg.get('torch_deterministic', False)))
        env = EvaluationTask(
            cfg=task_cfg, rl_device=args.device, sim_device=args.device,
            graphics_device_id=int(args.device.split(':')[1]), headless=True,
            virtual_screen_capture=False, force_render=False,
        )
        if not str(env.device).startswith('cuda') or not env.sim_params.use_gpu_pipeline or not env.sim_params.physx.use_gpu:
            raise RuntimeError('Isaac Gym is not using CUDA PhysX and the GPU pipeline')
        if env.num_envs != num_envs or len(env.object_asset_files) != panel_count:
            raise RuntimeError('Environment/object quotas do not match protocol')

        obs_dim = env.observation_space.shape[0]
        # CPU construction followed by transfer deliberately matches the training grid bit-for-bit.
        coef_ids = torch.linspace(50.0, 0.0, gene_count).to(args.device)
        network = ModelBuilder().load(params)
        model = network.build({
            'actions_num': env.action_space.shape[0], 'input_shape': (obs_dim + 1,),
            'num_seqs': num_envs, 'value_size': 1,
            'normalize_value': params['config'].get('normalize_value', False),
            'normalize_input': params['config'].get('normalize_input', False),
            'type': 'extra_param', 'param_size': latent_dim,
            'coef_ids': coef_ids, 'coef_id_idx': obs_dim,
        }).to(args.device)
        model.load_state_dict(model_state, strict=True)
        if params['config'].get('normalize_input') and 'running_mean_std' in state:
            model.running_mean_std.load_state_dict(state['running_mean_std'], strict=True)
        model.eval()
        model.requires_grad_(False)
        frozen_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        del checkpoint, state, model_state, extra_params
        torch.cuda.empty_cache()
        states = tuple(value.to(args.device) for value in model.get_default_rnn_state()) if model.is_rnn() else None
        env_ids = torch.arange(num_envs, device=args.device)
        chosen_genes = torch.tensor(genes, dtype=torch.long, device=args.device)
        assigned_genes = chosen_genes[(env_ids // panel_count) % len(genes)]
        identifiers = coef_ids[assigned_genes].unsqueeze(1)
        # Fail instead of accepting the network's silent argmax-to-gene-0 fallback.
        if not torch.all((identifiers == coef_ids).sum(dim=1) == 1).item():
            raise RuntimeError('Non-unique or unmatched gene identifier')
        decoded_gene_ids = (identifiers == model.a2c_network.param_ids).float().argmax(dim=1)
        decoded_sigma_ids = (identifiers == model.a2c_network.sigma_ids).float().argmax(dim=1)
        if not torch.equal(decoded_gene_ids, assigned_genes) or not torch.equal(decoded_sigma_ids, assigned_genes):
            raise RuntimeError('Network gene/sigma lookup differs from requested episode assignments')
        actions_low = torch.as_tensor(env.action_space.low, device=args.device)
        actions_high = torch.as_tensor(env.action_space.high, device=args.device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
        lengths = torch.zeros(num_envs, dtype=torch.long, device=args.device)
        returns = torch.zeros(num_envs, dtype=torch.float64, device=args.device)
        clipped_elements = torch.zeros(num_envs, dtype=torch.long, device=args.device)
        clipped_steps = torch.zeros(num_envs, dtype=torch.long, device=args.device)
        rows = []
        obs = env.reset()['obs']
        start = time.monotonic()
        last_report = start
        torch.cuda.reset_peak_memory_stats()
        fields = ['env_id', 'gene_id', 'gene_identifier', 'object_index', 'asset_name',
                  'scale_x', 'scale_y', 'scale_z', 'replica', 'eval_seed', 'training_seed',
                  'episode_index', 'steps_observed', 'successes_observed', 'extrinsic_return_observed',
                  'action_clip_fraction', 'action_clip_step_fraction',
                  'censored', 'termination_reason', 'task_timeout', 'cap_reached', 'object_fell',
                  'hand_far', 'final_tolerance', 'checkpoint_sha256']

        def record(indices, censored, writer):
            if not indices.numel():
                return
            ids = indices.cpu().tolist()
            successes = env.successes[indices].cpu().tolist()
            counts = lengths[indices].cpu().tolist()
            rewards = returns[indices].cpu().tolist()
            clip_counts = clipped_elements[indices].cpu().tolist()
            clip_step_counts = clipped_steps[indices].cpu().tolist()
            timeouts = env.timeout_buf[indices].cpu().tolist()
            fell = (env.object_pos[indices, 2] < 0.1).cpu().tolist()
            far = (env.curr_fingertip_distances[indices].max(dim=-1).values > 1.5).cpu().tolist()
            gene_values = assigned_genes[indices].cpu().tolist()
            identifier_values = identifiers[indices, 0].cpu().tolist()
            for position, env_id in enumerate(ids):
                item = panel[env_id % panel_count]
                cap = successes[position] >= success_cap
                causes = []
                if cap:
                    causes.append('success_cap')
                if timeouts[position]:
                    causes.append('task_timeout')
                if fell[position]:
                    causes.append('object_fell')
                if far[position]:
                    causes.append('hand_far')
                row = dict(zip(fields, [
                    env_id, gene_values[position], identifier_values[position], item['original_object_index'],
                    item['asset_name'], *item['scale'], env_id // (panel_count * len(genes)),
                    args.eval_seed, original_cfg['seed'], 0, counts[position], successes[position], rewards[position],
                    clip_counts[position] / (counts[position] * env.action_space.shape[0]),
                    clip_step_counts[position] / counts[position],
                    bool(censored), 'administrative_max_steps' if censored else '+'.join(causes) or 'other_task_done',
                    bool(timeouts[position]), cap, bool(fell[position]), bool(far[position]), args.tolerance, checkpoint_hash,
                ]))
                writer.writerow(row)
                rows.append(row)

        with (output / 'episodes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            with torch.no_grad():
                for step in range(1, args.max_steps + 1):
                    result = model({'is_train': False, 'prev_actions': None,
                                    'obs': torch.cat([obs, identifiers], dim=1), 'rnn_states': states})
                    states = result['rnn_states']
                    actions = result['mus'] if args.action_mode == 'deterministic' else result['actions']
                    if not torch.isfinite(actions).all().item():
                        raise RuntimeError('Nonfinite policy action before clipping')
                    clip_mask = torch.zeros_like(actions, dtype=torch.bool)
                    if params['config'].get('clip_actions', True):
                        clip_mask |= actions.ne(actions.clamp(-1, 1))
                        actions = (actions_high - actions_low) * 0.5 * actions.clamp(-1, 1) + (actions_high + actions_low) * 0.5
                    clip_mask |= actions.ne(actions.clamp(-env.clip_actions, env.clip_actions))
                    obs_dict, reward, dones, info = env.step(actions)
                    obs = obs_dict['obs']
                    active = ~finished
                    lengths += active.long()
                    returns += reward.double() * active
                    clipped_elements += clip_mask.sum(dim=1) * active.long()
                    clipped_steps += clip_mask.any(dim=1).long() * active.long()
                    new_done = active & dones.bool()
                    record(new_done.nonzero(as_tuple=False).flatten(), False, writer)
                    finished |= new_done
                    if states is not None:
                        for hidden in states:
                            hidden[:, dones.bool(), :] = 0
                    now = time.monotonic()
                    if now - last_report >= 30 or finished.all().item():
                        stream.flush()
                        run.log({'evaluation/control_steps': step,
                                 'evaluation/completed_episodes': int(finished.sum().item()),
                                 'evaluation/assigned_episodes': num_envs,
                                 'evaluation/wall_seconds': now - start})
                        print('eval step={} completed={}/{} elapsed={:.1f}s'.format(
                            step, int(finished.sum().item()), num_envs, now - start), flush=True)
                        last_report = now
                    if finished.all().item():
                        break
                record((~finished).nonzero(as_tuple=False).flatten(), True, writer)
        if len(rows) != num_envs or len({row['env_id'] for row in rows}) != num_envs:
            raise RuntimeError('First-episode quota violated')
        for name, value in model.named_buffers():
            if not torch.equal(value, frozen_buffers[name]):
                raise RuntimeError('Frozen model buffer changed: ' + name)
        result_summary = {
            'statistical_scope': protocol['statistical_scope'],
            'gene_summaries': summarize(rows, genes, success_cap),
            'episode_count': num_envs, 'censored_count': sum(row['censored'] for row in rows),
            'wall_seconds': time.monotonic() - start,
            'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(),
            'episodes_sha256': sha256_file(output / 'episodes.csv'),
            'normalizer_unchanged': True,
            'gene_assignment_validated': True,
            'action_clip_fraction': float(clipped_elements.sum().item()) / (
                int(lengths.sum().item()) * env.action_space.shape[0]),
            'action_clip_step_fraction': float(clipped_steps.sum().item()) / int(lengths.sum().item()),
        }
        write_json(output / 'summary.json', result_summary)
        for gene, values in result_summary['gene_summaries'].items():
            run.log({'evaluation/gene_{}/{}'.format(gene, key): value
                     for key, value in values.items() if value is not None})
        run.summary.update(result_summary)
        artifact = wandb.Artifact('frozen-evaluation-' + run.id, type='evaluation',
                                  metadata={'checkpoint_sha256': checkpoint_hash, 'source_sha256': source['source_sha256']})
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'wandb_run.json':
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact).wait()
        run.finish(exit_code=0)
        metadata.update(state='finished', exit_code=0)
        write_json(output / 'wandb_run.json', metadata)
        print(json.dumps(result_summary, indent=2, sort_keys=True), flush=True)
    except BaseException:
        metadata.update(state='failed', exit_code=1)
        write_json(output / 'wandb_run.json', metadata)
        try:
            run.finish(exit_code=1)
        finally:
            raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
