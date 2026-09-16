#!/usr/bin/env python3
"""Fixed full-catalog long-budget comparisons; statistics never stop training.

Numerical reductions/bootstrap are inherited unchanged from analyze_dapo_pair.py.
The legacy 'pilot' schema denotes a single continuation seed, not a stopping gate.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

N = 947
CAP = 50
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_CHUNK = 250
T_CRITICAL_DF2 = 4.30265272975
FLAGS = ('censored', 'task_timeout', 'cap_reached', 'object_fell', 'hand_far')
IDENTITY_COLUMNS = ('object_index', 'asset_name', 'scale_x', 'scale_y', 'scale_z', 'replica')
FIRST_EPISODE = 'exactly first episode per assigned environment; later episodes ignored'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def integer(value, name):
    number = float(value)
    require(math.isfinite(number) and number == int(number), 'Noninteger ' + name)
    return int(number)


def boolean(value, name):
    require(value in ('True', 'False'), 'Invalid CSV boolean ' + name)
    return value == 'True'


def absolute(path):
    p = Path(path)
    require(p.is_absolute() and str(p) == str(p.resolve()), 'Canonical absolute path required: ' + str(path))
    return p


def validate_plan(plan, plan_path, plan_sha):
    require(sha(plan_path) == plan_sha, 'Plan SHA mismatch')
    root = absolute(plan['execution_root'])
    output = absolute(plan['output'])
    require(root.is_dir() and root in output.parents, 'Output must be below execution_root')
    require(not output.exists(), 'Output exists; no automatic resume/retry')
    require(plan['stage'] == 'pilot', 'Only the single-continuation-seed schema is supported')
    require(plan['contrast'] in ('owned_clip', 'clip_higher'), 'Explicit treatment identity required')
    require(plan['relative_epoch'] in (306, 1526, 3052, 6104), 'Fixed milestone required')
    for key in ('run_id', 'entity', 'project', 'group'):
        require(isinstance(plan[key], str) and re.fullmatch(r'[A-Za-z0-9_.-]+', plan[key]), 'Invalid ' + key)
    expected = {(36001, 17001), (36001, 17002)}
    comparisons = plan['comparisons']
    keys = [(c['training_seed'], c['eval_seed']) for c in comparisons]
    require(len(keys) == len(expected) and set(keys) == expected, 'Exact fixed seed grid required')
    files = plan['files_sha256']
    required = {str(Path(__file__).resolve()), str(root / 'scripts/research/run_training_probe.py')}
    all_eval_ids = []
    for comparison in comparisons:
        for mode in ('control', 'candidate'):
            endpoint = comparison[mode]
            require(re.fullmatch(r'[0-9a-f]{64}', endpoint['checkpoint_sha256']), 'Invalid checkpoint SHA')
            require(re.fullmatch(r'[A-Za-z0-9_-]+', endpoint['run_id']), 'Invalid evaluation run ID')
            all_eval_ids.append(endpoint['run_id'])
            for field in ('episodes_csv', 'summary_json', 'protocol_json'):
                required.add(str(absolute(endpoint[field])))
            receipt = Path(endpoint['episodes_csv']).parent / 'wandb_run.json'
            required.add(str(receipt))
            require(all(Path(endpoint[f]).parent == receipt.parent for f in ('summary_json', 'protocol_json')),
                    'Endpoint files must come from the same original evaluation directory')
    require(len(set(all_eval_ids)) == len(all_eval_ids), 'Evaluation identities must be distinct')
    require(plan['run_id'] not in all_eval_ids, 'Analysis needs its own fresh ID')
    require(required <= set(files), 'Required input, receipt, source or publish helper is unbound')
    require(str(absolute(plan_path)) not in files, 'Plan SHA is separate; no circular self binding')
    for name, digest in files.items():
        path = absolute(name)
        require(path.is_file() and re.fullmatch(r'[0-9a-f]{64}', digest), 'Invalid file binding: ' + name)
        require(output not in path.parents, 'Inputs cannot reside inside fresh output')
        require(sha(path) == digest, 'Bound file SHA mismatch: ' + name)
    return sorted(comparisons, key=lambda c: (c['training_seed'], c['eval_seed']))


def parse_endpoint(endpoint, comparison):
    """CPU performs text/schema validation only; data-dependent reductions use CUDA."""
    protocol = read_json(endpoint['protocol_json'])
    summary = read_json(endpoint['summary_json'])
    receipt = read_json(Path(endpoint['episodes_csv']).parent / 'wandb_run.json')
    require(receipt['id'] == endpoint['run_id'] and receipt['state'] == 'finished'
            and receipt['exit_code'] == 0 and receipt['mode'] == 'online', 'Evaluation is not locally finished')
    require(protocol['protocol_version'] == 'frozen_first_episode_v1.1', 'Unsupported evaluator protocol')
    require(protocol['episode_selection'] == FIRST_EPISODE, 'First-episode scope differs')
    require(protocol['num_envs'] == N and protocol['gene_count'] == 64 and protocol['selected_genes'] == [63]
            and protocol['success_cap'] == CAP, 'Full947 master63/cap50 required')
    require(protocol['object_distribution'] == 'full generated object catalog', 'Full catalog required')
    require(all(protocol[k] is True for k in ('normalizer_frozen', 'curriculum_disabled', 'no_checkpoint_environment_restore')),
            'Evaluation frozen-policy semantics differ')
    require(summary['normalizer_unchanged'] is True and summary['gene_assignment_validated'] is True,
            'Evaluation controls did not pass')
    require(summary['episode_count'] == N and set(summary['gene_summaries']) == {'63'}, 'Summary quota differs')
    require(summary['episodes_sha256'] == sha(endpoint['episodes_csv']), 'Episode bytes differ from summary')
    require(protocol['checkpoint_sha256'] == endpoint['checkpoint_sha256'], 'Checkpoint identity differs')
    require(protocol['training_seed'] == comparison['training_seed'], 'Training seed differs')
    args = protocol['args']
    require(args['eval_seed'] == comparison['eval_seed'] and args['genes'] in ('63', 'master')
            and args['replicas_per_object'] == 1 and args['objects_per_gene'] == 0
            and args['max_envs'] == N and args['max_steps'] == 30000
            and args['action_mode'] == 'stochastic' and args['tolerance'] == .01,
            'Fixed evaluation arguments differ')
    if 'training_run_id' in endpoint:
        require(args['source_run_url'].rstrip('/').split('/')[-1] == endpoint['training_run_id'], 'Training run ID differs')
    with Path(endpoint['episodes_csv']).open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames
    needed = {'env_id', 'gene_id', 'gene_identifier', 'object_index', 'asset_name', 'scale_x', 'scale_y', 'scale_z',
              'replica', 'eval_seed', 'training_seed', 'episode_index', 'steps_observed', 'successes_observed',
              'extrinsic_return_observed', 'action_clip_fraction', 'action_clip_step_fraction', 'censored',
              'termination_reason', 'task_timeout', 'cap_reached', 'object_fell', 'hand_far', 'final_tolerance', 'checkpoint_sha256'}
    require(columns is not None and needed <= set(columns) and len(columns) == len(set(columns)), 'CSV schema differs')
    require(len(rows) == N, 'Exactly947 original rows required')
    parsed = {}
    for original in rows:
        r = dict(original)
        for key in ('env_id', 'gene_id', 'object_index', 'replica', 'eval_seed', 'training_seed', 'episode_index',
                    'steps_observed', 'successes_observed'):
            r[key] = integer(r[key], key)
        for key in FLAGS:
            r[key] = boolean(r[key], key)
        for key in ('gene_identifier', 'scale_x', 'scale_y', 'scale_z', 'extrinsic_return_observed',
                    'action_clip_fraction', 'action_clip_step_fraction', 'final_tolerance'):
            r[key] = float(r[key])
            require(math.isfinite(r[key]), 'Nonfinite CSV field ' + key)
        obj = r['object_index']
        require(obj not in parsed and 0 <= obj < N and r['env_id'] == obj, 'Duplicate/noncatalog object or env assignment')
        require(r['gene_id'] == 63 and r['gene_identifier'] == 0 and r['replica'] == 0 and r['episode_index'] == 0,
                'Master/replica/episode identity differs')
        require(r['eval_seed'] == comparison['eval_seed'] and r['training_seed'] == comparison['training_seed']
                and r['checkpoint_sha256'] == endpoint['checkpoint_sha256'], 'CSV seed/checkpoint identity differs')
        require(0 <= r['successes_observed'] <= CAP and 1 <= r['steps_observed'] <= 30000, 'Invalid count/cap/length')
        require(r['cap_reached'] == (r['successes_observed'] == CAP), 'Cap flag inconsistent')
        require(r['final_tolerance'] == .01 and r['asset_name'] and all(r[k] > 0 for k in ('scale_x', 'scale_y', 'scale_z')),
                'Object/tolerance invalid')
        require(all(0 <= r[k] <= 1 for k in ('action_clip_fraction', 'action_clip_step_fraction')), 'Clipping fraction invalid')
        if r['censored']:
            require(r['steps_observed'] == 30000 and r['termination_reason'] == 'administrative_max_steps'
                    and not any(r[k] for k in ('task_timeout', 'cap_reached', 'object_fell', 'hand_far')),
                    'Administrative censoring inconsistent with loop endpoint')
        else:
            causes = [name for flag, name in (('cap_reached', 'success_cap'), ('task_timeout', 'task_timeout'),
                      ('object_fell', 'object_fell'), ('hand_far', 'hand_far')) if r[flag]]
            require(r['termination_reason'] == ('+'.join(causes) or 'other_task_done'), 'Termination flags/reason mismatch')
        parsed[obj] = r
    require(set(parsed) == set(range(N)), 'Catalog incomplete')
    return {'rows': [parsed[i] for i in range(N)], 'summary': summary, 'protocol': protocol, 'receipt': receipt}


def load_comparisons(comparisons):
    loaded = []
    checkpoint_by_train_mode = {}
    for c in comparisons:
        pair = {m: parse_endpoint(c[m], c) for m in ('control', 'candidate')}
        for a, b in zip(pair['control']['rows'], pair['candidate']['rows']):
            require(all(a[k] == b[k] for k in IDENTITY_COLUMNS), 'Paired object metadata mismatch')
        left, right = pair['control']['protocol'], pair['candidate']['protocol']
        for key in ('checkpoint_frame', 'checkpoint_epoch', 'source_sha256', 'gene_count', 'latent_dim',
                    'initialization', 'assignment', 'rng_pairing', 'termination', 'task_timeout_interpretation'):
            require(left[key] == right[key], 'Paired protocol mismatch: ' + key)
        require(left['args']['object_seed'] == right['args']['object_seed'], 'Object seed mismatch')
        for mode in ('control', 'candidate'):
            p = pair[mode]['protocol']
            fingerprint = (p['checkpoint_sha256'], p['training_config_sha256'], p['checkpoint_frame'],
                           p['checkpoint_epoch'], p['args'].get('source_run_url'))
            key = (c['training_seed'], mode)
            require(key not in checkpoint_by_train_mode or checkpoint_by_train_mode[key] == fingerprint,
                    'Two eval seeds must use the same exact training endpoint')
            checkpoint_by_train_mode[key] = fingerprint
        loaded.append((c, pair))
    return loaded


def gpu_resources(root, reject_processes=False):
    if reject_processes:
        processes = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        require(not processes.strip(), 'GPU has a compute process; no concurrent execution')
    line = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    fields = line.split(',')
    require(len(fields) == 2 and '\n' not in line, 'Exactly one selected GPU required')
    free, total = [int(x.strip()) * 2 ** 20 for x in fields]
    disk = shutil.disk_usage(root).free
    result = {'time_utc': now(), 'gpu_free_bytes_sampled': free, 'gpu_total_bytes': total, 'disk_free_bytes': disk}
    require(free >= 2 ** 30 and disk >= 2 ** 30, 'Sampled GPU/disk1GiB floor failed')
    return result


def percentile_interval(t, samples):
    """CUDA type7/linear-interpolation percentiles at fixed2.5% and97.5%."""
    ordered = t.sort(samples).values
    quantiles = []
    for probability in (.025, .975):
        rank = (BOOTSTRAPS - 1) * probability
        lower = math.floor(rank)
        fraction = rank - lower
        quantiles.append(ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction)
    return t.stack(quantiles)


def bootstrap(t, difference, generator):
    means = t.empty(BOOTSTRAPS, device='cuda:0', dtype=t.float64)
    for start in range(0, BOOTSTRAPS, BOOTSTRAP_CHUNK):
        count = min(BOOTSTRAP_CHUNK, BOOTSTRAPS - start)
        indices = t.randint(N, (count, N), device='cuda:0', generator=generator, dtype=t.int64)
        means[start:start + count] = difference[indices].mean(dim=1)
    return means, percentile_interval(t, means)


def analyze_cuda(t, loaded, stage, output):
    require(t.cuda.is_available() and t.__version__ == '1.8.1' and t.version.cuda == '11.1', 'Expected Torch1.8.1 CUDA11.1')
    generator = t.Generator(device='cuda:0')
    generator.manual_seed(BOOTSTRAP_SEED)
    raw, cases, train_data = {}, [], {}
    raw['generator_state_before'] = generator.get_state()  # RNG state bytes, not algorithm computation on CPU.
    for c, pair in loaded:
        label = 'train{}_eval{}'.format(c['training_seed'], c['eval_seed'])
        tensors, endpoint_results = {}, {}
        for mode in ('control', 'candidate'):
            endpoint = pair[mode]
            rows = endpoint['rows']
            values = t.tensor([r['successes_observed'] for r in rows], dtype=t.float64, device='cuda:0')
            flags = t.tensor([[r[k] for k in FLAGS] for r in rows], dtype=t.bool, device='cuda:0')
            lengths = t.tensor([r['steps_observed'] for r in rows], dtype=t.int64, device='cuda:0')
            returns = t.tensor([r['extrinsic_return_observed'] for r in rows], dtype=t.float64, device='cuda:0')
            censored = flags[:, 0]
            upper = t.where(censored, t.full_like(values, CAP), values)
            n_censored = int(censored.sum().item())
            source = endpoint['summary']
            gs = source['gene_summaries']['63']
            require(source['censored_count'] == n_censored and gs['censored_episodes'] == n_censored
                    and gs['assigned_episodes'] == N and gs['completed_episodes'] == N - n_censored,
                    'Original summary counts differ from CUDA original-row reductions')
            lo_mean, hi_mean = float(values.mean().item()), float(upper.mean().item())
            require(abs(gs['mean_successes_lower_bound'] - lo_mean) <= 1e-12
                    and abs(gs['mean_successes_upper_bound'] - hi_mean) <= 1e-12, 'Original mean bounds mismatch')
            require((gs['mean_successes'] is None) if n_censored else
                    (gs['mean_successes'] is not None and abs(gs['mean_successes'] - lo_mean) <= 1e-12), 'Original point mean differs')
            counts = {k: int(flags[:, i].sum().item()) for i, k in enumerate(FLAGS)}
            endpoint_results[mode] = {'evaluation_run_id': c[mode]['run_id'], 'checkpoint_sha256': c[mode]['checkpoint_sha256'],
                'episode_count': N, 'observed_success_sum': int(values.sum().item()),
                'success_mean': None if n_censored else lo_mean, 'mean_bounds': [lo_mean, hi_mean],
                'flag_counts_overlapping': counts, 'noncap_completed': int(((~censored) & (values < CAP)).sum().item()),
                'episode_length_sum': int(lengths.sum().item()),
                'actual_control_steps_inferred_from_first_episode_lengths': int(lengths.max().item()),
                'full_simulated_environment_frames': N * int(lengths.max().item()),
                'extrinsic_return_observed_sum': float(returns.sum().item())}
            tensors[mode] = (values, upper, censored)
            raw[label + '/' + mode] = {'successes_observed': values, 'upper': upper, 'flags': flags, 'steps': lengths, 'returns': returns}
        cv, cu, cc = tensors['control']
        nv, nu, nc = tensors['candidate']
        lower, upper = nv - cu, nu - cv
        censored = bool((cc | nc).any().item())
        difference = nv - cv
        interval = None
        if not censored:
            samples, ci = bootstrap(t, difference, generator)
            raw[label + '/bootstrap_means'] = samples
            interval = ci.tolist()
        raw[label + '/difference_observed'] = difference
        raw[label + '/difference_lower'] = lower
        raw[label + '/difference_upper'] = upper
        cases.append({'training_seed': c['training_seed'], 'eval_seed': c['eval_seed'], 'endpoints': endpoint_results,
            'censored': censored, 'point_difference': None if censored else float(difference.mean().item()),
            'difference_mean_bounds': [float(lower.mean().item()), float(upper.mean().item())],
            'paired_object_bootstrap_percentile95': interval,
            'bootstrap_replicates_actual': 0 if censored else BOOTSTRAPS})
        train_data.setdefault(c['training_seed'], []).append((difference.mean(), lower.mean(), upper.mean(), censored))
        # Raw CSV values are storage copies; arithmetic above exclusively CUDA.
        with (output / (label + '_object_differences.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['object_index', 'control_successes_observed', 'candidate_successes_observed',
                             'control_censored', 'candidate_censored', 'observed_difference', 'difference_lower', 'difference_upper'])
            stored = t.stack((cv, nv, cc.to(t.float64), nc.to(t.float64), difference, lower, upper), dim=1).cpu().tolist()
            for obj, fields in enumerate(stored):
                writer.writerow([obj] + fields)
    raw['generator_state_after'] = generator.get_state()
    all_complete = not any(case['censored'] for case in cases)
    equal_mean = t.stack([v[0] for seed in sorted(train_data) for v in train_data[seed]]).mean()
    result = {'cases': cases, 'all_uncensored': all_complete,
              'equal_weight_mean_difference': float(equal_mean.item()) if all_complete else None,
              'equal_weight_mean_bounds': [float(t.stack([v[i] for seed in sorted(train_data) for v in train_data[seed]]).mean().item()) for i in (1, 2)]}
    if stage == 'pilot':
        result['pilot_gate'] = {'zero_censoring': all_complete,
            'both_eval_point_differences_positive': all_complete and all(c['point_difference'] > 0 for c in cases),
            'equal_weight_mean_at_least_half': all_complete and float(equal_mean.item()) >= .5,
            'each_object_bootstrap_lower_positive': all_complete and all(c['paired_object_bootstrap_percentile95'][0] > 0 for c in cases)}
        result['pilot_gate_passed'] = all(result['pilot_gate'].values())
    else:
        seed_means = t.stack([t.stack([v[0] for v in train_data[seed]]).mean() for seed in sorted(train_data)])
        seed_bounds = {str(seed): [float(t.stack([v[i] for v in train_data[seed]]).mean().item()) for i in (1, 2)] for seed in sorted(train_data)}
        interval = None
        if all_complete:
            mean = seed_means.mean()
            se = seed_means.std(unbiased=True) / math.sqrt(3)
            interval = t.stack((mean - T_CRITICAL_DF2 * se, mean + T_CRITICAL_DF2 * se)).tolist()
        raw['formal_training_seed_differences_observed'] = seed_means
        result['formal'] = {'training_seeds': sorted(train_data), 'paired_training_seed_differences': seed_means.tolist() if all_complete else None,
            'training_seed_difference_bounds': seed_bounds, 'mean_t_df2_95_interval': interval,
            't_critical': T_CRITICAL_DF2, 'independent_unit': 'three paired training seeds; two eval seeds averaged first',
            'assumption': 'Student t interval assumes independent approximately normal training-seed differences; n=3 cannot establish this assumption'}
        result['pilot_gate_passed'] = None  # No invented formal learning/adoption gate.
    def storage(value):
        if isinstance(value, dict):
            return {k: storage(v) for k, v in value.items()}
        return value.detach().cpu()  # serialization only; no CPU tensor arithmetic
    t.save(storage(raw), output / 'cuda_results.pt')
    return result


def execute(plan, plan_path, plan_sha, comparisons):
    output = Path(plan['output'])
    root = Path(plan['execution_root'])
    run, published, original_finish = None, False, None
    finish_entered, finish_returned, init_entered = 0, 0, False
    receipt = {'status': 'running', 'run_id': plan['run_id'], 'plan_sha256': plan_sha,
               'entry_sha256': sha(__file__), 'started_utc': now(), 'remote_verified_by_parent': False}
    def finish_once(exit_code):
        nonlocal finish_entered, finish_returned
        require(run is not None and run.id == plan['run_id'] and callable(original_finish), 'No owned run finish')
        require(finish_entered == 0, 'Own finish already entered')
        finish_entered = 1
        receipt['effective_wandb_exit_code'] = exit_code
        original_finish(exit_code=exit_code)
        finish_returned = 1
    try:
        resources = [gpu_resources(root, reject_processes=True)]
        require(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '0'), 'Unexpected selected CUDA device')
        require(os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1', 'Original no_cache process required')
        import isaacgym  # noqa:F401 -- must precede Torch
        import torch
        import wandb
        spec = importlib.util.spec_from_file_location('dapo_analysis_publish_base', root / 'scripts/research/run_training_probe.py')
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        init_entered = True
        candidate = wandb.init(id=plan['run_id'], resume='never', mode='online', entity=plan['entity'], project=plan['project'],
                               group=plan['group'], job_type='dapo-paired-analysis', dir=str(output), config=plan)
        if candidate is not None and candidate.id == plan['run_id']:
            run, original_finish = candidate, candidate.finish
        require(run is not None and not run.offline and not run.resumed, 'Fresh owned online analysis run required')
        input_dir = output / 'inputs'
        input_dir.mkdir()
        payload = {'plan.json': plan_path}
        copied = {}
        for index, (source, digest) in enumerate(sorted(plan['files_sha256'].items())):
            target = input_dir / ('{:03d}_'.format(index) + Path(source).name)
            shutil.copyfile(source, target)
            require(sha(target) == digest and sha(source) == digest, 'Input copy changed')
            copied[source] = {'archived_path': str(target), 'sha256': digest, 'size': target.stat().st_size}
            payload['inputs/' + target.name] = target
        write(output / 'input_manifest.json', copied)
        payload['input_manifest.json'] = output / 'input_manifest.json'
        receipt['plan_publication'] = base.publish(run, payload, 'dapo-pair-analysis-plan')
        write(output / 'publication.json', receipt)
        loaded = load_comparisons(comparisons)
        resources.append(gpu_resources(root))
        result = analyze_cuda(torch, loaded, plan['stage'], output)
        torch.cuda.synchronize()
        resources.append(gpu_resources(root))
        require(sha(plan_path) == plan_sha and all(sha(p) == s for p, s in plan['files_sha256'].items()), 'Postcompute source/input SHA mismatch')
        result.update(status='completed', stage=plan['stage'], gpu_validated=True, analysis_run_id=run.id,
                      contrast=plan['contrast'], relative_epoch=plan['relative_epoch'],
                      primary_endpoint=plan['relative_epoch'] == 6104,
                      stopping_policy='Descriptive comparison only; legacy pilot_gate fields never allocate or stop this study',
            actual_comparisons=len(comparisons), bootstrap_seed=BOOTSTRAP_SEED, bootstrap_chunk=BOOTSTRAP_CHUNK,
            bootstrap_order='sorted(training_seed,eval_seed), one private CUDA Generator; censored comparisons draw no samples',
            bootstrap_interval_definition='10000 paired-object resamples; 2.5/97.5 percentiles with linear interpolation (type7)',
            object_bootstrap_scope='conditional on each checkpoint/eval seed; object-composition sensitivity under an iid object-superpopulation interpretation; not a training-seed CI or uncertainty of the enumerated fixed947 catalog mean',
            censoring='All947 retained: unfinished success S in [observed,50]; point differences and CI null if any censoring; pilot gate false',
            pairing='Same object asset/scale; equal eval seeds do not guarantee common trajectory random numbers',
            failure_flag_scope='Task failure flags overlap; success cap is a completed endpoint, not administrative censoring',
            resources=resources, resource_scope='startup/precompute/postcompute sampled1GiB GPU/disk floor, not a continuous peak guarantee',
            runtime={'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
                     'dtype': 'CUDA float64', 'allocator_disabled': True, 'checkpoint_model_loads': 0, 'model_forwards': 0,
                     'backwards': 0, 'optimizer_steps': 0, 'physical_environment_frames': 0},
            ended_utc=now(), entry_sha256=sha(__file__), plan_sha256=plan_sha)
        write(output / 'summary.json', result)
        raw_files = {p.name: p for p in output.iterdir() if p.is_file() and p.name not in ('publication.json',)}
        run.summary.update({'status': 'completed', 'stage': plan['stage'], 'pilot_gate_passed': result['pilot_gate_passed'],
                            'equal_weight_mean_difference': result['equal_weight_mean_difference']})
        # Treat these bytes as sealed before any upload attempt, even if upload then raises.
        published = True
        receipt['raw_publication'] = base.publish(run, raw_files, 'dapo-pair-analysis-results')
        finish_once(0)
        receipt.update(status='completed', exit_code=0, state='finished', finished_utc=now(),
                       own_finish_entered=finish_entered, own_finish_returned=finish_returned)
        write(output / 'publication.json', receipt)
    except BaseException as primary:
        # wandb.init may register a run before throwing. Only the declared own ID is eligible for cleanup.
        if run is None and init_entered:
            active_wandb = sys.modules.get('wandb')
            candidate = getattr(active_wandb, 'run', None) if active_wandb is not None else None
            if candidate is not None and getattr(candidate, 'id', None) == plan['run_id']:
                run, original_finish = candidate, candidate.finish
        error = {'status': 'failed', 'error': repr(primary), 'traceback': traceback.format_exc(), 'time_utc': now(),
                 'raw_publication_attempted': published, 'local_raw_retained': True,
                 'previous_summary_may_be_sealed': published}
        failure_dir = output / 'failure'
        evidence_written = False
        try:
            failure_dir.mkdir(exist_ok=True)
            error['retained_top_level_files'] = {str(p): {'sha256': sha(p), 'size': p.stat().st_size}
                for p in output.iterdir() if p.is_file() and p.name != 'publication.json'}
            write(failure_dir / 'failure.json', error)
            evidence_written = True
        except BaseException as secondary:
            receipt['failure_record_error'] = repr(secondary)
        if run is not None and run.id == plan['run_id']:
            if finish_entered == 0:
                try:
                    if evidence_written:
                        failure_files = {'failure.json': failure_dir / 'failure.json'}
                        if not published:
                            failure_files.update({'partial/' + p.name: p for p in output.iterdir()
                                                  if p.is_file() and p.name != 'publication.json'})
                        receipt['failure_publication'] = base.publish(run, failure_files, 'dapo-pair-analysis-failure')
                except BaseException as secondary:
                    receipt['failure_publication_error'] = repr(secondary)
                try:
                    finish_once(1)
                except BaseException as secondary:
                    receipt['finish_error'] = repr(secondary)
            else:
                receipt['late_failure_scope'] = 'Own finish already entered; no second finish or post-finish upload. Local linked failure retained.'
        receipt.update(status='failed', exit_code=1, ended_utc=now(), error=repr(primary),
                       own_finish_entered=finish_entered, own_finish_returned=finish_returned)
        try:
            write(output / 'publication.json', receipt)
        except BaseException as secondary:
            print('Secondary failure receipt error: ' + repr(secondary), file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    comparisons = validate_plan(plan, plan_path, args.plan_sha256)
    if not args.execute:
        print(json.dumps({'status': 'dry_run', 'stage': plan['stage'], 'comparisons': len(comparisons),
                          'validated_file_bindings': len(plan['files_sha256']), 'torch_imported': 'torch' in sys.modules,
                          'gpu_started': False, 'network_used': False, 'execute_authorized_by_this_check': False}, sort_keys=True))
        return
    require(Path.cwd().resolve() == Path(plan['execution_root']), 'Execution cwd differs')
    with open('/tmp/epo-gpu-0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path(plan['output']).mkdir(parents=True)
        execute(plan, plan_path, args.plan_sha256, comparisons)


if __name__ == '__main__':
    main()
