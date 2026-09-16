#!/usr/bin/env python3
"""Validate first-episode evaluation artifacts and compare each gene to master.

Pure CPU data analysis: no torch, Isaac Gym, policy inference, or simulation.
All comparisons are conditional on the frozen checkpoint and sampled objects.
The script neither selects a champion nor estimates training-seed uncertainty.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


SCOPE = ('Conditional on one frozen checkpoint, one training seed, and this object panel. '
         'Descriptive paired comparisons; no training-seed confidence interval, '
         'no post-selection champion claim, and no multiple-comparison significance claim.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value, name):
    number = float(value)
    require(math.isfinite(number), 'Nonfinite ' + name)
    return number


def integer(value, name):
    number = finite(value, name)
    require(number == int(number), 'Nonintegral ' + name)
    return int(number)


def boolean(value, name):
    require(value in ('True', 'False'), 'Invalid Boolean ' + name)
    return value == 'True'


def close(left, right, message, tolerance=1e-8):
    require(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), message)


def success_bounds(rows, cap):
    count = len(rows)
    if not count:
        return {'episodes': 0, 'censored': 0, 'mean_successes': None,
                'lower_bound': None, 'upper_bound': None}
    lower = sum(row['successes_observed'] for row in rows) / count
    upper = sum(cap if row['censored'] else row['successes_observed'] for row in rows) / count
    censored = sum(row['censored'] for row in rows)
    return {'episodes': count, 'censored': censored,
            'mean_successes': lower if not censored else None,
            'lower_bound': lower, 'upper_bound': upper}


def verify_source(directory, protocol):
    manifest = read_json(directory / 'source_manifest.json')
    digest = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    require(digest == manifest['source_sha256'] == protocol['source_sha256'], 'Source manifest hash mismatch')
    with tarfile.open(str(directory / 'source_snapshot.tar.gz'), 'r:gz') as archive:
        files = {item.name: item for item in archive.getmembers() if item.isfile()}
        require(set(files) == set(manifest['files']), 'Source archive file set differs from manifest')
        for name, expected in manifest['files'].items():
            stream = archive.extractfile(files[name])
            require(hashlib.sha256(stream.read()).hexdigest() == expected, 'Source archive content changed: ' + name)
    return digest


def validate(directory):
    protocol = read_json(directory / 'protocol.json')
    panel = read_json(directory / 'object_panel.json')
    summary = read_json(directory / 'summary.json')
    metadata = read_json(directory / 'wandb_run.json')
    train = read_json(directory / 'training_config.resolved.json')
    task = read_json(directory / 'task_config.evaluation.json')
    require(metadata['state'] == 'finished' and metadata.get('exit_code') == 0,
            'Only fully finished successful evaluations can be analyzed')
    require(metadata['mode'] == 'online', 'Evaluation was not online')
    require(summary['normalizer_unchanged'] is True, 'Normalizer invariance was not established')
    if protocol['protocol_version'] == 'frozen_first_episode_v1.1':
        require(summary.get('gene_assignment_validated') is True, 'Gene assignment was not validated')
    require(sha256_file(directory / 'episodes.csv') == summary['episodes_sha256'], 'Episode CSV checksum mismatch')
    verify_source(directory, protocol)
    genes = protocol['selected_genes']
    require(genes and len(genes) == len(set(genes)), 'Duplicated or empty selected genes')
    gene_count = int(protocol['gene_count'])
    require(all(isinstance(gene, int) and 0 <= gene < gene_count for gene in genes), 'Illegal gene indices')
    objects = panel['objects']
    catalog_count = int(panel['catalog_size'])
    object_ids = [item['original_object_index'] for item in objects]
    require(object_ids and len(set(object_ids)) == len(object_ids), 'Duplicated or empty object panel')
    require(all(isinstance(index, int) and 0 <= index < catalog_count for index in object_ids), 'Illegal object index')
    replicas = int(protocol['args']['replicas_per_object'])
    require(replicas > 0, 'Invalid replica count')
    total = len(genes) * len(objects) * replicas
    require(total == protocol['num_envs'] == summary['episode_count'], 'Episode budget disagreement')
    cap = int(protocol['success_cap'])
    require(cap == task['env']['maxConsecutiveSuccesses'] and cap > 0, 'Success cap disagreement')
    maximum_steps = int(protocol['args']['max_steps'])
    tolerance = float(protocol['args']['tolerance'])
    close(task['env']['successTolerance'], tolerance, 'Task initial tolerance mismatch')
    close(task['env']['targetSuccessTolerance'], tolerance, 'Task target tolerance mismatch')
    require(protocol['curriculum_disabled'] and protocol['normalizer_frozen'], 'Required freeze flags absent')
    train_n = int(train['task']['env']['numEnvs'])
    block_size = int(train['train']['params']['config']['expl_coef_block_size'])
    require(train_n % block_size == 0 and train_n // block_size == gene_count, 'Training N/B differs from K')
    action_count = int(task['env'].get('numActions', 23 + (3 if task['env']['privilegedActions'] else 0)))
    rows = []
    identifiers = {}
    clipping_available = None
    with (directory / 'episodes.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        for line, source in enumerate(reader, 2):
            row = dict(source)
            label = ' at CSV line {}'.format(line)
            for name in ['env_id', 'gene_id', 'object_index', 'replica', 'eval_seed', 'training_seed',
                         'episode_index', 'steps_observed']:
                row[name] = integer(row[name], name + label)
            for name in ['gene_identifier', 'successes_observed', 'extrinsic_return_observed', 'final_tolerance',
                         'scale_x', 'scale_y', 'scale_z']:
                row[name] = finite(row[name], name + label)
            for name in ['censored', 'task_timeout', 'cap_reached', 'object_fell', 'hand_far']:
                row[name] = boolean(row[name], name + label)
            env_id = row['env_id']
            require(0 <= env_id < total, 'Environment ID outside assigned budget' + label)
            expected_gene = genes[(env_id // len(objects)) % len(genes)]
            expected_object = objects[env_id % len(objects)]
            require(row['gene_id'] == expected_gene, 'Gene assignment mismatch' + label)
            require(row['object_index'] == expected_object['original_object_index'], 'Object assignment mismatch' + label)
            require(row['asset_name'] == expected_object['asset_name'], 'Asset basename mismatch' + label)
            for coord, scale in zip(['scale_x', 'scale_y', 'scale_z'], expected_object['scale']):
                close(row[coord], scale, 'Object scale mismatch' + label)
            require(row['replica'] == env_id // (len(objects) * len(genes)), 'Replica assignment mismatch' + label)
            require(row['episode_index'] == 0, 'Later episode entered first-episode quota' + label)
            require(row['training_seed'] == train['seed'] == protocol['training_seed'], 'Training seed mismatch' + label)
            require(row['eval_seed'] == protocol['args']['eval_seed'], 'Evaluation seed mismatch' + label)
            require(row['checkpoint_sha256'] == protocol['checkpoint_sha256'], 'Checkpoint identity mismatch' + label)
            close(row['final_tolerance'], tolerance, 'Evaluation tolerance changed' + label)
            # CUDA/CPU float32 linspace has rounding differences from Python float.
            expected_identifier = 50.0 * (1.0 - expected_gene / (gene_count - 1)) if gene_count > 1 else 50.0
            close(row['gene_identifier'], expected_identifier, 'Gene grid mismatch' + label, tolerance=1e-5)
            if expected_gene in identifiers:
                require(identifiers[expected_gene] == row['gene_identifier'], 'Identifier changes within gene' + label)
            identifiers[expected_gene] = row['gene_identifier']
            steps = row['steps_observed']
            success = row['successes_observed']
            require(1 <= steps <= maximum_steps, 'Impossible episode duration' + label)
            require(0 <= success <= cap and success == int(success), 'Invalid success count' + label)
            require(row['cap_reached'] == (success >= cap), 'Success-cap flag mismatch' + label)
            causes = [name for name, flag in [('success_cap', 'cap_reached'), ('task_timeout', 'task_timeout'),
                                             ('object_fell', 'object_fell'), ('hand_far', 'hand_far')] if row[flag]]
            if row['censored']:
                require(steps == maximum_steps and row['termination_reason'] == 'administrative_max_steps',
                        'Administrative censoring horizon mismatch' + label)
                require(not causes, 'Censored episode already satisfies task termination' + label)
            else:
                require(bool(causes) and row['termination_reason'] == '+'.join(causes), 'Termination reason mismatch' + label)
            if row['task_timeout']:
                require(steps >= task['env']['episodeLength'] - 1, 'Task timeout before minimum duration' + label)
            has_clip = 'action_clip_fraction' in row and 'action_clip_step_fraction' in row
            if clipping_available is None:
                clipping_available = has_clip
            require(clipping_available == has_clip, 'Inconsistent clipping schema')
            if has_clip:
                for name in ['action_clip_fraction', 'action_clip_step_fraction']:
                    row[name] = finite(row[name], name + label)
                    require(0 <= row[name] <= 1, 'Clipping fraction outside [0,1]' + label)
                cf, sf = row['action_clip_fraction'], row['action_clip_step_fraction']
                require(cf <= sf + 1e-10 and sf <= action_count * cf + 1e-10, 'Clipping fractions inconsistent' + label)
                close(cf * steps * action_count, round(cf * steps * action_count), 'Nonintegral clipped-coordinate count' + label)
                close(sf * steps, round(sf * steps), 'Nonintegral clipped-step count' + label)
            rows.append(row)
    require(len(rows) == total and {row['env_id'] for row in rows} == set(range(total)),
            'Missing or duplicated environment first episodes')
    quotas = Counter((row['gene_id'], row['object_index']) for row in rows)
    require(quotas == Counter({(gene, obj): replicas for gene in genes for obj in object_ids}), 'Gene-object quotas differ')
    require(len(set(identifiers.values())) == len(genes), 'Multiple genes share an identifier')
    require(sum(row['censored'] for row in rows) == summary['censored_count'], 'Summary censor count mismatch')
    if protocol['protocol_version'] == 'frozen_first_episode_v1.1':
        require(clipping_available, 'V1.1 CSV has no clipping metrics')
    by_gene = defaultdict(list)
    for row in rows:
        by_gene[row['gene_id']].append(row)
    for gene in genes:
        bounds = success_bounds(by_gene[gene], cap)
        saved = summary['gene_summaries'][str(gene)]
        require(saved['assigned_episodes'] == bounds['episodes'] and saved['censored_episodes'] == bounds['censored'],
                'Per-gene summary count mismatch')
        close(saved['mean_successes_lower_bound'], bounds['lower_bound'], 'Per-gene lower bound mismatch')
        close(saved['mean_successes_upper_bound'], bounds['upper_bound'], 'Per-gene upper bound mismatch')
        require(saved['mean_successes'] == bounds['mean_successes'], 'Censored mean was treated as observed endpoint')
    total_steps = sum(row['steps_observed'] for row in rows)
    if clipping_available:
        close(sum(row['action_clip_fraction'] * row['steps_observed'] for row in rows) / total_steps,
              summary['action_clip_fraction'], 'Aggregate action clipping mismatch')
        close(sum(row['action_clip_step_fraction'] * row['steps_observed'] for row in rows) / total_steps,
              summary['action_clip_step_fraction'], 'Aggregate step clipping mismatch')
    require(finite(summary['wall_seconds'], 'wall_seconds') > 0, 'Invalid evaluation wall time')
    return protocol, panel, summary, metadata, train, task, rows, clipping_available


def analyze(protocol, panel, task, train, rows, master, clipping_available):
    genes = protocol['selected_genes']
    require(master in genes, 'Predeclared master gene was not evaluated; paired comparisons unavailable')
    cap = protocol['success_cap']
    block = int(train['train']['params']['config']['expl_coef_block_size'])
    catalog = panel['catalog_size']
    by_gene, by_gene_object = defaultdict(list), defaultdict(list)
    for row in rows:
        by_gene[row['gene_id']].append(row)
        by_gene_object[(row['gene_id'], row['object_index'])].append(row)
    control_dt = float(task['sim']['dt']) * int(task['env']['controlFrequencyInv'])
    gene_table, object_table, paired_table, composition = [], [], [], []
    for gene in sorted(genes):
        selected = by_gene[gene]
        bounds = success_bounds(selected, cap)
        complete = [row for row in selected if not row['censored']]
        total_steps = sum(row['steps_observed'] for row in selected)
        line = {'gene_id': gene, 'is_predeclared_master': gene == master,
                **bounds, 'completed_episodes': len(complete),
                'cap_fraction_lower_bound': sum(row['cap_reached'] for row in selected) / len(selected),
                'completed_only_median_successes': statistics.median(row['successes_observed'] for row in complete) if complete else None,
                'mean_steps_observed': total_steps / len(selected),
                'mean_sim_seconds_observed': control_dt * total_steps / len(selected),
                'successes_per_sim_second_observed': sum(row['successes_observed'] for row in selected) / (total_steps * control_dt),
                'mean_extrinsic_return_observed': statistics.mean(row['extrinsic_return_observed'] for row in selected),
                'action_clip_fraction': sum(row['action_clip_fraction'] * row['steps_observed'] for row in selected) / total_steps if clipping_available else None,
                'action_clip_step_fraction': sum(row['action_clip_step_fraction'] * row['steps_observed'] for row in selected) / total_steps if clipping_available else None}
        for flag in ['task_timeout', 'object_fell', 'hand_far', 'cap_reached']:
            line[flag + '_count'] = sum(row[flag] for row in selected)
        gene_table.append(line)
        allocated = {index % catalog for index in range(gene * block, (gene + 1) * block)}
        for is_allocated in (True, False):
            subset = [row for row in selected if (row['object_index'] in allocated) == is_allocated]
            composition.append({'gene_id': gene,
                                'composition_group': 'allocated_to_gene_during_training' if is_allocated else 'not_allocated_to_gene_during_training',
                                **success_bounds(subset, cap),
                                'objects': len({row['object_index'] for row in subset})})
        paired = []
        for obj in panel['objects']:
            object_id = obj['original_object_index']
            own = success_bounds(by_gene_object[(gene, object_id)], cap)
            reference = success_bounds(by_gene_object[(master, object_id)], cap)
            delta_low = own['lower_bound'] - reference['upper_bound']
            delta_high = own['upper_bound'] - reference['lower_bound']
            observed = own['mean_successes'] is not None and reference['mean_successes'] is not None
            if gene == master:
                delta_low = delta_high = 0.0
            delta = own['mean_successes'] - reference['mean_successes'] if observed else (0.0 if gene == master else None)
            object_table.append({'gene_id': gene, 'object_index': object_id, 'asset_name': obj['asset_name'],
                                 **own, 'master_gene_id': master, 'master_mean_successes': reference['mean_successes'],
                                 'paired_difference': delta, 'paired_difference_lower_bound': delta_low,
                                 'paired_difference_upper_bound': delta_high,
                                 'allocated_to_gene_during_training': object_id in allocated})
            paired.append((delta, delta_low, delta_high))
        paired_table.append({'gene_id': gene, 'predeclared_master_gene_id': master,
                             'paired_objects': len(paired),
                             'fully_observed_object_pairs': sum(item[0] is not None for item in paired),
                             'mean_paired_difference': statistics.mean(item[0] for item in paired) if all(item[0] is not None for item in paired) else None,
                             'mean_paired_difference_lower_bound': statistics.mean(item[1] for item in paired),
                             'mean_paired_difference_upper_bound': statistics.mean(item[2] for item in paired)})
    return gene_table, object_table, paired_table, composition


def verify_remote(directory, metadata, protocol):
    import wandb
    api = wandb.Api(timeout=30)
    run = api.run('{entity}/{project}/{id}'.format(**metadata))
    require(run.state == 'finished', 'Remote W&B run is not finished: ' + run.state)
    expected_name = 'frozen-evaluation-' + run.id
    candidates = [artifact for artifact in run.logged_artifacts()
                  if artifact.type == 'evaluation' and artifact.name.split(':')[0] == expected_name]
    require(bool(candidates), 'Remote evaluation artifact absent')
    candidates.sort(key=lambda artifact: int(artifact.version[1:]))
    artifact = candidates[-1]
    require(artifact.state == 'COMMITTED', 'Remote artifact is not COMMITTED: ' + artifact.state)
    require(artifact.metadata['checkpoint_sha256'] == protocol['checkpoint_sha256'], 'Remote checkpoint hash mismatch')
    require(artifact.metadata['source_sha256'] == protocol['source_sha256'], 'Remote source hash mismatch')
    entries = artifact.manifest.entries
    needed = ['episodes.csv', 'summary.json', 'protocol.json', 'object_panel.json',
              'source_manifest.json', 'source_snapshot.tar.gz', 'training_config.resolved.json',
              'task_config.evaluation.json']
    for name in needed:
        require(name in entries, 'Remote artifact missing ' + name)
        require(entries[name].size == (directory / name).stat().st_size, 'Remote artifact size mismatch for ' + name)
    return {'verification': 'passed', 'run_state': run.state, 'url': run.url,
            'artifact_name': artifact.name, 'artifact_state': artifact.state,
            'artifact_digest': artifact.digest, 'required_files': needed,
            'content_verification': 'local SHA256/source archive validated; remote manifest identity and file sizes validated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--output-dir', required=True, help='Must not already exist')
    parser.add_argument('--master-gene', type=int, default=None, help='Predeclared reference; default K-1, never chosen from scores')
    parser.add_argument('--skip-remote', action='store_true', help='Explicitly mark online artifact verification incomplete')
    args = parser.parse_args()
    directory = Path(args.evaluation_dir).resolve()
    output = Path(args.output_dir).resolve()
    require(not output.exists(), 'Refusing to overwrite analysis output')
    protocol, panel, summary, metadata, train, task, rows, clipping = validate(directory)
    master = args.master_gene if args.master_gene is not None else protocol['gene_count'] - 1
    tables = analyze(protocol, panel, task, train, rows, master, clipping)
    remote = {'verification': 'skipped', 'reason': '--skip-remote was explicitly specified'}
    if not args.skip_remote:
        remote = verify_remote(directory, metadata, protocol)
    output.mkdir(parents=True)
    for name, table in zip(['gene_metrics.csv', 'gene_object_metrics.csv', 'paired_vs_master.csv', 'training_object_composition.csv'], tables):
        write_csv(output / name, table)
    report = {
        'validation': 'passed', 'analysis_version': 'frozen_evaluation_analysis_v1',
        'statistical_scope': SCOPE, 'master_gene_id': master,
        'rows': len(rows), 'genes': len(protocol['selected_genes']), 'objects': len(panel['objects']),
        'replicas_per_gene_object': protocol['args']['replicas_per_object'],
        'censored_episodes': summary['censored_count'],
        'checkpoint_sha256': protocol['checkpoint_sha256'], 'source_sha256': protocol['source_sha256'],
        'episodes_sha256': summary['episodes_sha256'], 'analyzer_sha256': sha256_file(__file__),
        'wall_seconds': summary['wall_seconds'], 'termination_flag_counts': {
            key: sum(row[key] for row in rows) for key in ['cap_reached', 'task_timeout', 'object_fell', 'hand_far']},
        'termination_counts_overlap': True, 'clipping_metrics_available': clipping,
        'composition_interpretation': (
            'Inferred from original object index modulo catalog size and training gene environment block. '
            'Assumes the historical catalog and ordering match this source. Shared network, off-policy '
            'experience, and GA lineage transfers invalidate treating not-allocated objects as truly unseen/OOD. '
            'Allocated and not-allocated groups contain different objects; differences are descriptive, not causal.'),
        'paired_interpretation': ('Same object panel and equal quotas; independent stochastic trajectories, '
                                  'not identical random-number streams. Gene means are reported in gene-index order. '
                                  'Any candidate chosen here requires separate confirmation data.'),
        'remote': remote,
    }
    write_json(output / 'validation_report.json', report)
    (output / 'README.md').write_text(
        '# Frozen checkpoint analysis\n\n' + SCOPE + '\n\n'
        'Every row is the first episode of one assigned environment. Administrative censoring '
        'retains the observed success lower bound and the task-cap upper bound. An endpoint mean '
        'is blank whenever that gene has unresolved episodes. Task timeouts remain benchmark endpoints.\n\n'
        '`gene_metrics.csv` is ordered by gene index. `paired_vs_master.csv` uses the predeclared '
        'master gene {} and the same objects for every pair. No ranking-based champion is declared.\n\n'.format(master)
        + report['composition_interpretation'] + '\n\n'
        'Action clipping uses episode-step weighting. Successes per simulated second and extrinsic '
        'return are descriptive over observed trajectory time; they do not replace the endpoint metric. '
        'Termination flags can overlap.\n\n'
        'Remote verification: {}.\n'.format(remote['verification']))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
