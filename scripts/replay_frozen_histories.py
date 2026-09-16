#!/usr/bin/env python3
"""CUDA-only frozen-policy replay on exactly recorded behavior histories.

The behavior self replay is a mandatory control. Target policies reconstruct
their own states from the beginning; legacy follower-hidden outputs are kept
separately. No environment is stepped and no shadow action is sampled.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from collect_frozen_histories import bootstrap, check_buffers, identifiers_for, load_policy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=Path('/home/ubuntu22/EPO'))
    parser.add_argument('--history-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-genes', default='0,15,31,47,55,59,60,61,62,63')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-lock', default='/tmp/epo-gpu-0.lock')
    parser.add_argument('--wandb-entity', default='leo-ucbm-university-of-california-berkeley')
    parser.add_argument('--wandb-project', default='EPO-reproduction')
    parser.add_argument('--wandb-group', default='executed-command-diagnostic-20260913')
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('This resource protocol requires cuda:0')
    return args


def load_json(path):
    return json.loads(Path(path).read_text())


def deterministic_forward(model, obs, identifiers, states):
    import torch
    normalized = model.norm_obs(torch.cat([obs, identifiers], dim=1))
    mu, logstd, _, out_states = model.a2c_network({
        'is_train': False, 'prev_actions': None, 'obs': normalized, 'rnn_states': states})
    return mu, logstd.exp(), logstd, out_states


def close_or_fail(name, actual, expected, counters, atol=1e-6, rtol=1e-5):
    import torch
    if actual.shape != expected.shape:
        raise RuntimeError(name + ': shape mismatch')
    finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
    difference = (actual - expected).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    counters[name] = max(counters.get(name, 0.0), maximum)
    if not bool(finite.item()) or bool((difference > atol + rtol * expected.abs()).any().item()):
        raise RuntimeError('{} exceeded declared FP32 replay tolerance; max_abs={}'.format(name, maximum))


def run(args):
    torch = bootstrap(args.repo_dir.resolve())
    import wandb
    from scripts.evaluate_frozen_checkpoint import sha256_file, source_snapshot, write_json
    source_dir = args.history_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite replay output: ' + str(output))
    if shutil.disk_usage(str(args.repo_dir)).free < 2 * 2**30:
        raise RuntimeError('At least 2 GiB free disk is required')
    protocol = load_json(source_dir / 'protocol.json')
    summary = load_json(source_dir / 'summary.json')
    source_run = load_json(source_dir / 'wandb_run.json')
    if summary.get('status') != 'completed' or source_run.get('state') != 'finished':
        raise RuntimeError('The history collection is not finalized successfully')
    history_path = source_dir / 'histories.pt.gz'
    if sha256_file(history_path) != summary['histories_sha256']:
        raise RuntimeError('History data hash differs from collection summary')
    checkpoint = Path(protocol['checkpoint_path'])
    if sha256_file(checkpoint) != protocol['checkpoint_sha256']:
        raise RuntimeError('Frozen behavior checkpoint bytes changed')
    # Verify actual source completion remotely before using its evidence.
    remote = wandb.Api(timeout=30).run('/'.join([source_run['entity'], source_run['project'], source_run['id']]))
    source_artifacts = [artifact for artifact in remote.logged_artifacts() if artifact.type == 'fixed-history']
    if remote.state != 'finished' or not any(
            artifact.state == 'COMMITTED' and artifact.metadata.get('histories_sha256') == summary['histories_sha256']
            and artifact.metadata.get('checkpoint_sha256') == protocol['checkpoint_sha256']
            for artifact in source_artifacts):
        raise RuntimeError('History run or matching artifact is not remotely finalized')
    cfg = load_json(source_dir / 'training_config.resolved.json')
    from omegaconf import OmegaConf
    saved_config_path = Path(protocol['args']['train_config'])
    if sha256_file(saved_config_path) != protocol['training_config_sha256']:
        raise RuntimeError('Original saved training config changed')
    if OmegaConf.to_container(OmegaConf.load(str(saved_config_path)), resolve=True) != cfg:
        raise RuntimeError('Resolved history config differs from the saved training config')
    if sha256_file(Path(__file__).with_name('collect_frozen_histories.py')) != protocol['collector_source_sha256']:
        raise RuntimeError('Collector/load-policy helper source changed since sampling')
    with gzip.open(str(history_path), 'rb') as stream:
        data = torch.load(stream, map_location='cpu')
    steps, count, obs_dim = data['obs'].shape
    actions_num = data['raw_action'].shape[-1]
    if (steps != protocol['steps'] or count != protocol['num_envs'] or actions_num != 23
            or summary['physical_frames'] != steps * count):
        raise RuntimeError('History shape/quota differs from its declared protocol')
    required_shapes = {'mu': (steps, count, actions_num), 'sigma': (steps, count, actions_num),
                       'command': (steps, count, actions_num), 'done_after': (steps, count),
                       'reset_before': (steps, count), 'behavior_neglogp': (steps, count)}
    for key, shape in required_shapes.items():
        if tuple(data[key].shape) != shape:
            raise RuntimeError('Malformed history field: ' + key)
    targets = [int(item) for item in args.target_genes.split(',')]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError('Target genes must be nonempty and unique')
    model, coef_ids, origin = load_policy(checkpoint, cfg, obs_dim, actions_num, count, args.device)
    if min(targets) < 0 or max(targets) >= origin['gene_count']:
        raise ValueError('Target gene is outside the checkpoint')
    assigned = data['assigned_genes'].to(args.device)
    behavior_ids = identifiers_for(model, coef_ids, assigned)
    if not torch.equal(behavior_ids, data['gene_identifiers'].to(args.device)):
        raise RuntimeError('Stored behavior identifiers differ from checkpoint lookup')
    if not torch.equal(data['command'].to(args.device), data['raw_action'].to(args.device).clamp(-1, 1)):
        raise RuntimeError('Stored command is not the declared clipping of its raw action')
    reset = data['reset_before'].to(args.device)
    done = data['done_after'].to(args.device)
    if reset[0].any().item() or not torch.equal(reset[1:], done[:-1]):
        raise RuntimeError('Stored done/reset timing differs from the declared recurrence')
    output.mkdir(parents=True)
    source = source_snapshot(output)
    if source['source_sha256'] != protocol['source_sha256']:
        raise RuntimeError('Repository source changed since history sampling')
    for path in (Path(__file__), Path(__file__).with_name('collect_frozen_histories.py')):
        shutil.copyfile(str(path), str(output / path.name))
    replay_protocol = {
        'protocol_version': 'frozen_history_replay_v1', 'history_dir': str(source_dir),
        'history_sha256': summary['histories_sha256'], 'history_run_url': source_run['url'],
        'checkpoint_sha256': protocol['checkpoint_sha256'], 'checkpoint_path': str(checkpoint),
        'source_sha256': source['source_sha256'], 'source_collector_sha256': protocol['collector_source_sha256'],
        'replay_source_sha256': sha256_file(Path(__file__)), 'target_genes': targets,
        'steps': steps, 'num_envs': count, 'action_dimension': actions_num,
        'new_physical_frames': 0, 'training_benefit_tested': False, 'actual_scheduler_reproduced': False,
        'comparison': 'same frozen checkpoint; each target replays its own complete recorded observation history',
        'legacy_control': 'same target identifier but supplied the stored behavior hidden input at every step',
        'forward_precision': 'FP32 without autocast', 'self_replay_atol': 1e-6, 'self_replay_rtol': 1e-5,
        'rng': 'deterministic raw network forward; CUDA generator checked unchanged',
        'target_hidden_storage': 'full target mu/sigma and hidden-difference norms at every step; full target hidden anchors at steps 0,31,127,255,511 when present. Complete input history/checkpoint retained for reconstruction.',
        'statistical_scope': 'conditional on fixed checkpoint and sampled histories; no independent-training-seed inference',
    }
    write_json(output / 'protocol.json', replay_protocol)
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
        job_type='diagnostic', name=output.name, dir=str(output), config=replay_protocol,
        mode='online', resume='never', tags=['fixed_history_replay', 'recurrent_control', 'gpu_only'])
    metadata = {'id': run.id, 'entity': run.entity, 'project': run.project,
                'url': run.url, 'state': 'running'}
    write_json(output / 'wandb_run.json', metadata)
    counters, output_files = {}, []
    started = time.monotonic()
    try:
        if run.offline or run.settings.mode != 'online' or run.resumed:
            raise RuntimeError('Fresh online W&B recording is required')
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        rng_start = torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            states = tuple(state.to(args.device) for state in model.get_default_rnn_state())
            for step in range(steps):
                for index, state in enumerate(states):
                    close_or_fail('behavior_hidden_in_' + str(index), state,
                                  data['hidden_in'][index][step].to(args.device), counters)
                mu, sigma, logstd, states = deterministic_forward(
                    model, data['obs'][step].to(args.device), behavior_ids, states)
                close_or_fail('behavior_mu', mu, data['mu'][step].to(args.device), counters)
                close_or_fail('behavior_sigma', sigma, data['sigma'][step].to(args.device), counters)
                neglogp = model.neglogp(data['raw_action'][step].to(args.device), mu, sigma, logstd)
                close_or_fail('behavior_neglogp', neglogp, data['behavior_neglogp'][step].to(args.device), counters)
                for index, state in enumerate(states):
                    close_or_fail('behavior_hidden_out_' + str(index), state,
                                  data['hidden_out'][index][step].to(args.device), counters)
                    state[:, done[step], :] = 0
            if not torch.equal(torch.cuda.get_rng_state(), rng_start):
                raise RuntimeError('Behavior deterministic replay consumed CUDA RNG')
            write_json(output / 'self_replay_validation.json', {'status': 'passed', 'max_abs_errors': counters,
                       'atol': 1e-6, 'rtol': 1e-5, 'steps_checked': steps, 'cuda_rng_unchanged': True})
            anchors = {index for index in (0, 31, 127, 255, 511) if index < steps}
            for target in targets:
                states = tuple(state.to(args.device) for state in model.get_default_rnn_state())
                target_ids = identifiers_for(model, coef_ids, torch.full_like(assigned, target))
                traces = {key: [] for key in ('mu', 'sigma', 'legacy_mu', 'legacy_sigma',
                                             'hidden_difference_l2', 'behavior_hidden_l2')}
                hidden_anchors = {}
                self_rows = assigned == target
                for step in range(steps):
                    obs = data['obs'][step].to(args.device)
                    behavior_states = tuple(value[step].to(args.device) for value in data['hidden_in'])
                    difference_squared = sum((state - original).square().sum(dim=(0, 2))
                                             for state, original in zip(states, behavior_states))
                    behavior_squared = sum(state.square().sum(dim=(0, 2)) for state in behavior_states)
                    traces['hidden_difference_l2'].append(difference_squared.sqrt().cpu())
                    traces['behavior_hidden_l2'].append(behavior_squared.sqrt().cpu())
                    if step in anchors:
                        hidden_anchors[step] = {'input': tuple(state.cpu().clone() for state in states)}
                    mu, sigma, _, states = deterministic_forward(model, obs, target_ids, states)
                    legacy_mu, legacy_sigma, _, _ = deterministic_forward(model, obs, target_ids, behavior_states)
                    for key, value in [('mu', mu), ('sigma', sigma), ('legacy_mu', legacy_mu), ('legacy_sigma', legacy_sigma)]:
                        if not torch.isfinite(value).all().item():
                            raise RuntimeError('Nonfinite target replay: ' + key)
                        traces[key].append(value.cpu().clone())
                    if self_rows.any().item():
                        close_or_fail('target_self_mu', mu[self_rows], data['mu'][step].to(args.device)[self_rows], counters)
                        close_or_fail('target_self_sigma', sigma[self_rows], data['sigma'][step].to(args.device)[self_rows], counters)
                    if step in anchors:
                        hidden_anchors[step]['output'] = tuple(state.cpu().clone() for state in states)
                    for state in states:
                        state[:, done[step], :] = 0
                if not torch.equal(torch.cuda.get_rng_state(), rng_start):
                    raise RuntimeError('A target replay consumed CUDA RNG')
                # Stack on CPU for storage only; all model/norm calculations ran on CUDA.
                payload = {key: torch.stack(values) for key, values in traces.items()}
                payload.update(target_gene=target, hidden_anchors=hidden_anchors,
                               checkpoint_sha256=protocol['checkpoint_sha256'], history_sha256=summary['histories_sha256'])
                path = output / ('target_{}.pt.gz'.format(target))
                with gzip.open(str(path), 'wb', compresslevel=1) as stream:
                    torch.save(payload, stream)
                output_files.append({'target_gene': target, 'name': path.name, 'sha256': sha256_file(path),
                                     'bytes': path.stat().st_size})
                run.log({'replay/targets_completed': len(output_files), 'replay/target_gene': target,
                         'replay/wall_seconds': time.monotonic() - started})
        check_buffers(model, buffers)
        result = {'status': 'completed', 'targets_completed': len(output_files), 'target_files': output_files,
                  'self_replay_max_abs_errors': counters, 'cuda_rng_unchanged': True, 'normalizer_unchanged': True,
                  'wall_seconds': time.monotonic() - started, 'new_physical_frames': 0,
                  'training_benefit_tested': False, 'actual_scheduler_reproduced': False}
        write_json(output / 'summary.json', result)
        run.summary.update(result)
        artifact = wandb.Artifact('fixed-history-replay-' + run.id, type='fixed-history-replay',
            metadata={'history_sha256': summary['histories_sha256'], 'checkpoint_sha256': protocol['checkpoint_sha256']})
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'wandb_run.json':
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact).wait()
        run.finish(exit_code=0)
        metadata.update(state='finished', exit_code=0)
        write_json(output / 'wandb_run.json', metadata)
    except BaseException as error:
        error_record = {'type': type(error).__name__, 'message': str(error)}
        write_json(output / 'summary.json', {'status': 'failed', 'error': error_record,
                   'target_files': output_files, 'self_replay_max_abs_errors': counters})
        metadata.update(state='failed', exit_code=1, error=error_record)
        write_json(output / 'wandb_run.json', metadata)
        try:
            run.summary.update({'status': 'failed'})
            run.finish(exit_code=1)
        except BaseException as finalization_error:
            metadata['finalization_error'] = str(finalization_error)
            write_json(output / 'wandb_run.json', metadata)
        raise


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
