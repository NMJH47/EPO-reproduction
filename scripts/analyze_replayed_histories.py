#!/usr/bin/env python3
"""CUDA FP64 latent/command metrics on validated frozen recurrent histories.

All distribution calculations run on CUDA. CPU is used for reference
quadrature, serialization and control-plane checks. No training is performed.
"""
import argparse
import csv
import fcntl
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import time

from collect_frozen_histories import bootstrap


def read_json(path):
    return json.loads(Path(path).read_text())


def load_tensors(path, torch):
    with gzip.open(str(path), 'rb') as stream:
        return torch.load(stream, map_location='cpu')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, default=Path('/home/ubuntu22/EPO'))
    parser.add_argument('--replay-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['smoke', 'development', 'validation'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-lock', default='/tmp/epo-gpu-0.lock')
    parser.add_argument('--wandb-entity', default='leo-ucbm-university-of-california-berkeley')
    parser.add_argument('--wandb-project', default='EPO-reproduction')
    parser.add_argument('--wandb-group', default='executed-command-diagnostic-20260913')
    args = parser.parse_args()
    if args.device != 'cuda:0':
        parser.error('This resource protocol requires cuda:0')
    return args


def run(args):
    torch = bootstrap(args.repo_dir.resolve())
    import wandb
    import clipped_gaussian_metrics as metrics
    from validate_clipped_gaussian_metrics import run_cuda_validation, source_hashes
    from scripts.evaluate_frozen_checkpoint import sha256_file, source_snapshot, write_json
    replay = args.replay_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite command diagnostic: ' + str(output))
    if shutil.disk_usage(str(args.repo_dir)).free < 2 * 2**30:
        raise RuntimeError('At least 2 GiB free disk is required')
    rp = read_json(replay / 'protocol.json')
    rs = read_json(replay / 'summary.json')
    rm = read_json(replay / 'wandb_run.json')
    if rs.get('status') != 'completed' or rm.get('state') != 'finished':
        raise RuntimeError('Replay did not finalize successfully')
    if read_json(replay / 'self_replay_validation.json').get('status') != 'passed':
        raise RuntimeError('Behavior self replay is not validated')
    remote = wandb.Api(timeout=30).run('/'.join([rm['entity'], rm['project'], rm['id']]))
    if remote.state != 'finished' or not any(
            artifact.type == 'fixed-history-replay' and artifact.state == 'COMMITTED'
            and artifact.metadata.get('history_sha256') == rp['history_sha256']
            and artifact.metadata.get('checkpoint_sha256') == rp['checkpoint_sha256']
            for artifact in remote.logged_artifacts()):
        raise RuntimeError('Replay run/artifact is not remotely finalized')
    history = Path(rp['history_dir'])
    hp = read_json(history / 'protocol.json')
    history_path = history / 'histories.pt.gz'
    if sha256_file(history_path) != rp['history_sha256']:
        raise RuntimeError('History bytes changed after replay')
    data = load_tensors(history_path, torch)
    steps, count, actions_num = data['mu'].shape
    if steps != rp['steps'] or count != rp['num_envs'] or actions_num != 23:
        raise RuntimeError('Replay/history dimensions differ')
    p = data['mu'].to(device=args.device, dtype=torch.float64)
    sp = data['sigma'].to(device=args.device, dtype=torch.float64)
    raw = data['raw_action'].to(device=args.device, dtype=torch.float64)
    assigned = data['assigned_genes'].to(args.device)
    e_clip = float(read_json(history / 'training_config.resolved.json')['train']['params']['config']['e_clip'])
    if e_clip != 0.1:
        raise RuntimeError('The predeclared support diagnostic assumes E51 e_clip=0.1')
    target_files = rs['target_files']
    if [item['target_gene'] for item in target_files] != rp['target_genes']:
        raise RuntimeError('Target completion/order differs from the replay protocol')
    actual_cases, case_keys = [], set()
    # Reference validation covers deterministic extreme real coordinates from
    # every target and both hidden conventions before any result inference.
    for item in target_files:
        path = replay / item['name']
        if sha256_file(path) != item['sha256']:
            raise RuntimeError('Target trace bytes changed: ' + item['name'])
        trace = load_tensors(path, torch)
        for prefix in ('', 'legacy_'):
            q = trace[prefix + 'mu'].to(device=args.device, dtype=torch.float64)
            sq = trace[prefix + 'sigma'].to(device=args.device, dtype=torch.float64)
            candidates = [0, int(q.abs().reshape(-1).argmax().item()),
                          int((p - q).abs().reshape(-1).argmax().item()),
                          int(sq.reshape(-1).argmin().item()), int(sq.reshape(-1).argmax().item())]
            for index in candidates:
                values = tuple(float(tensor.reshape(-1)[index].item()) for tensor in (p, sp, q, sq))
                if values in case_keys:
                    continue
                case_keys.add(values)
                actual_cases.append({'name': 'real_target_{}_{}coordinate_{}'.format(item['target_gene'], prefix, index),
                                     'p': list(values[:2]), 'q': list(values[2:])})
        correct_mu = trace['mu'].to(device=args.device, dtype=torch.float64)
        correct_sigma = trace['sigma'].to(device=args.device, dtype=torch.float64)
        legacy_mu = trace['legacy_mu'].to(device=args.device, dtype=torch.float64)
        legacy_sigma = trace['legacy_sigma'].to(device=args.device, dtype=torch.float64)
        candidates = [0, int(legacy_mu.abs().reshape(-1).argmax().item()),
                      int((correct_mu - legacy_mu).abs().reshape(-1).argmax().item()),
                      int(legacy_sigma.reshape(-1).argmin().item()),
                      int(legacy_sigma.reshape(-1).argmax().item())]
        for index in candidates:
            values = tuple(float(tensor.reshape(-1)[index].item())
                           for tensor in (correct_mu, correct_sigma, legacy_mu, legacy_sigma))
            if values not in case_keys:
                case_keys.add(values)
                actual_cases.append({'name': 'real_hidden_target_{}_coordinate_{}'.format(item['target_gene'], index),
                                     'p': list(values[:2]), 'q': list(values[2:])})
        del trace, q, sq, correct_mu, correct_sigma, legacy_mu, legacy_sigma
    output.mkdir(parents=True)
    source = source_snapshot(output)
    for name in ('analyze_replayed_histories.py', 'clipped_gaussian_metrics.py',
                 'clipped_gaussian_reference.py', 'validate_clipped_gaussian_metrics.py'):
        shutil.copyfile(str(Path(__file__).with_name(name)), str(output / name))
    protocol = {
        'protocol_version': 'replayed_command_metrics_v1', 'split': args.split,
        'history_sha256': rp['history_sha256'], 'checkpoint_sha256': rp['checkpoint_sha256'],
        'replay_dir': str(replay), 'replay_run_url': rm['url'], 'history_run_url': rp['history_run_url'],
        'source_sha256': source['source_sha256'], 'metrics_sources': source_hashes(),
        'analysis_source_sha256': sha256_file(Path(__file__)), 'target_genes': rp['target_genes'],
        'behavior_genes': hp['behavior_genes'], 'e_clip': e_clip, 'forward_precision': 'FP32',
        'distribution_precision': 'CUDA FP64', 'new_physical_frames': 0,
        'statistical_unit': 'recorded environment history; output rows are conditional action distributions, not independent training seeds',
        'interpretation': 'Descriptive mechanism screen. No training benefit, trajectory KL or real scheduler disagreement claim.',
        'training_benefit_tested': False, 'actual_scheduler_reproduced': False,
        'directions': {'p_to_q': 'behavior || target', 'q_to_p': 'target || behavior'},
        'hidden_approximation_directions': {'p_to_q': 'correct target history || legacy behavior hidden', 'q_to_p': 'legacy behavior hidden || correct target history'},
        'hidden_conventions': ['correct_full_history', 'legacy_behavior_hidden'],
        'large_contraction_rule': 'latent joint KL>=1e-3 and command joint KL<=0.5*latent joint KL; self rows excluded from pooled rates',
        'real_precision_case_selection': 'first, max absolute target mu, max mu difference, min/max target sigma; each behavior-target hidden convention and correct-target versus legacy-target pair',
        'coordinate_storage': 'full P/Q mu/sigma are archived in history and target traces; joint metrics and coordinate summaries retained here',
    }
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'real_reference_cases.json', actual_cases)
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
        job_type='diagnostic', name=output.name, dir=str(output), config=protocol,
        mode='online', resume='never', tags=['command_kl', 'recurrent', args.split, 'gpu_only', 'no_training_gain_claim'])
    metadata = {'id': run.id, 'entity': run.entity, 'project': run.project, 'url': run.url, 'state': 'running'}
    write_json(output / 'wandb_run.json', metadata)
    started = time.monotonic()
    summary_rows, coordinate_rows, hidden_rows, artifacts = [], [], [], []
    try:
        if run.offline or run.settings.mode != 'online' or run.resumed:
            raise RuntimeError('Fresh online W&B recording is required')
        validation = run_cuda_validation(device=args.device, actual_cases=actual_cases)
        write_json(output / 'precision_validation.json', validation)
        if validation.get('status') != 'passed' or validation.get('gpu_validated') is not True:
            raise RuntimeError('CUDA metric precision validation failed; no inference is allowed')
        windows = [('all', 0, steps), ('cold_start', 0, 1), ('early', 1, min(32, steps)),
                   ('32_127', 32, min(128, steps)), ('128_255', 128, min(256, steps)),
                   ('256_511', 256, min(512, steps))]
        for item in target_files:
            target = item['target_gene']
            if sha256_file(replay / item['name']) != item['sha256']:
                raise RuntimeError('Target trace changed during precision validation')
            trace = load_tensors(replay / item['name'], torch)
            joint_records = {}
            for label, prefix in [('correct_full_history', ''), ('legacy_behavior_hidden', 'legacy_')]:
                q = trace[prefix + 'mu'].to(device=args.device, dtype=torch.float64)
                sq = trace[prefix + 'sigma'].to(device=args.device, dtype=torch.float64)
                kl = metrics.clipped_gaussian_kl(p, sp, q, sq)
                ratio = metrics.joint_logratio_and_support(raw, p, sp, q, sq, e_clip=e_clip)
                if kl['any_violation'].item() or not ratio['all_logratios_finite'].item():
                    raise RuntimeError('A real-data metric violated finiteness or data-processing bounds')
                values = {key: ratio[key] for key in ('raw_logratio', 'command_logratio', 'raw_support',
                          'command_support', 'both_supported', 'raw_only_supported', 'command_only_supported', 'neither_supported')}
                for direction in ('p_to_q', 'q_to_p'):
                    values[direction + '_latent_kl'] = kl[direction]['latent_joint']
                    values[direction + '_command_kl'] = kl[direction]['command_joint']
                joint_records[label] = {key: value.cpu() for key, value in values.items()}
                for behavior in hp['behavior_genes']:
                    mask = assigned == behavior
                    for window, lower, upper in windows:
                        if upper <= lower or lower >= steps:
                            continue
                        raw_lw = ratio['raw_logratio'][lower:upper, mask].reshape(-1)
                        command_lw = ratio['command_logratio'][lower:upper, mask].reshape(-1)
                        row = {'split': args.split, 'target_gene': target, 'behavior_gene': behavior,
                               'is_self': behavior == target, 'hidden_convention': label, 'window': window,
                               'rows': raw_lw.numel(), 'raw_ess_fraction': float(metrics.logspace_ess(raw_lw)['normalized_ess'].item()),
                               'command_ess_fraction': float(metrics.logspace_ess(command_lw)['normalized_ess'].item())}
                        for key in ('raw_support', 'command_support', 'both_supported', 'raw_only_supported',
                                    'command_only_supported', 'neither_supported'):
                            row[key + '_fraction'] = float(ratio[key][lower:upper, mask].double().mean().item())
                        row['support_disagreement_fraction'] = row['raw_only_supported_fraction'] + row['command_only_supported_fraction']
                        for direction in ('p_to_q', 'q_to_p'):
                            latent = kl[direction]['latent_joint'][lower:upper, mask]
                            command = kl[direction]['command_joint'][lower:upper, mask]
                            row[direction + '_latent_mean'] = float(latent.mean().item())
                            row[direction + '_command_mean'] = float(command.mean().item())
                            row[direction + '_gap_mean'] = float((latent - command).mean().item())
                            row[direction + '_large_contraction_fraction'] = float(((latent >= 1e-3) & (command <= 0.5 * latent)).double().mean().item())
                            row[direction + '_negative_command_coordinates'] = int(kl[direction]['negative_command_coordinate'][lower:upper, mask].sum().item())
                        summary_rows.append(row)
                    for direction in ('p_to_q', 'q_to_p'):
                        for coordinate in range(actions_num):
                            coordinate_rows.append({'target_gene': target, 'behavior_gene': behavior,
                                'is_self': target == behavior, 'hidden_convention': label, 'direction': direction,
                                'coordinate': coordinate, 'rows': steps * int(mask.sum().item()),
                                'latent_mean': float(kl[direction]['latent_coordinate'][:, mask, coordinate].mean().item()),
                                'command_mean': float(kl[direction]['command_coordinate'][:, mask, coordinate].mean().item())})
                joint_records[label]['raw_at_exact_boundary_count'] = int(ratio['raw_at_exact_boundary'].sum().item())
                joint_records[label]['tail_underflow_coordinate_counts'] = {
                    side + '_' + key: int(kl[side][key].sum().item()) for side in ('p_components', 'q_components')
                    for key in ('mass_underflow_left', 'mass_underflow_right', 'mass_underflow_interior')}
            # Isolate the recurrent-history approximation while keeping target gene fixed.
            hidden_kl = metrics.clipped_gaussian_kl(
                trace['mu'].to(device=args.device, dtype=torch.float64),
                trace['sigma'].to(device=args.device, dtype=torch.float64),
                trace['legacy_mu'].to(device=args.device, dtype=torch.float64),
                trace['legacy_sigma'].to(device=args.device, dtype=torch.float64))
            if hidden_kl['any_violation'].item():
                raise RuntimeError('Correct-vs-legacy hidden metric failed numerical bounds')
            for behavior in hp['behavior_genes']:
                mask = assigned == behavior
                for window, lower, upper in windows:
                    if upper <= lower or lower >= steps:
                        continue
                    row = {'target_gene': target, 'behavior_gene': behavior, 'is_self': target == behavior,
                           'window': window, 'rows': (upper - lower) * int(mask.sum().item()),
                           'hidden_difference_l2_mean': float(trace['hidden_difference_l2'][lower:upper].to(args.device)[:, mask].double().mean().item()),
                           'behavior_hidden_l2_mean': float(trace['behavior_hidden_l2'][lower:upper].to(args.device)[:, mask].double().mean().item())}
                    for direction in ('p_to_q', 'q_to_p'):
                        for key in ('latent_joint', 'command_joint'):
                            row[direction + '_' + key + '_mean'] = float(hidden_kl[direction][key][lower:upper, mask].mean().item())
                    hidden_rows.append(row)
            joint_records['correct_vs_legacy_hidden'] = {
                direction + '_' + key: hidden_kl[direction][key].cpu() for direction in ('p_to_q', 'q_to_p')
                for key in ('latent_joint', 'command_joint')}
            path = output / ('target_{}_joint_metrics.pt.gz'.format(target))
            with gzip.open(str(path), 'wb', compresslevel=1) as stream:
                torch.save(joint_records, stream)
            artifacts.append({'name': path.name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size})
            run.log({'metrics/targets_completed': len(artifacts), 'metrics/wall_seconds': time.monotonic() - started})
        for filename, rows in [('conditional_summary.csv', summary_rows), ('coordinate_summary.csv', coordinate_rows), ('hidden_approximation_summary.csv', hidden_rows)]:
            with (output / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        final = {'status': 'completed', 'split': args.split, 'targets': len(target_files),
                 'summary_rows': len(summary_rows), 'coordinate_summary_rows': len(coordinate_rows),
                 'joint_files': artifacts, 'precision_validated': True, 'new_physical_frames': 0,
                 'training_benefit_tested': False, 'actual_scheduler_reproduced': False,
                 'wall_seconds': time.monotonic() - started}
        write_json(output / 'summary.json', final)
        run.summary.update(final)
        artifact = wandb.Artifact('command-diagnostic-' + run.id, type='command-diagnostic',
            metadata={'history_sha256': rp['history_sha256'], 'checkpoint_sha256': rp['checkpoint_sha256'], 'split': args.split})
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'wandb_run.json':
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact).wait()
        run.finish(exit_code=0)
        metadata.update(state='finished', exit_code=0)
        write_json(output / 'wandb_run.json', metadata)
    except BaseException as error:
        record = {'type': type(error).__name__, 'message': str(error)}
        write_json(output / 'summary.json', {'status': 'failed', 'error': record, 'joint_files': artifacts})
        metadata.update(state='failed', exit_code=1, error=record)
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
