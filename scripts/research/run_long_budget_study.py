#!/usr/bin/env python3
"""Fixed three-arm long-budget queue; default inert, no learning-based stop/retry."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys

LEGACY_SHA256 = 'd1a34a50663a849f1d45531181d53b36cf68b265e1fa575938c4f0b9daf9c6fd'
ARMS = ('frontier', 'control', 'clip_higher')
MILESTONES = (306, 1526, 3052, 6104)
SEEDS = (17001, 17002)
ROLES = ('train_entry', 'evaluation_entry', 'analysis_entry', 'monitor_entry',
         'monitor_source', 'nvml_source', 'base_source', 'evaluator_source',
         'legacy_controller_source')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Missing/linked input: ' + str(p))
    before = p.stat()
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    after = p.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Input changed')
    return digest.hexdigest()


def load_legacy(study):
    path = study['legacy_controller_source']
    require(study['files_sha256'].get(path) == LEGACY_SHA256 == sha(path), 'Frozen legacy controller differs')
    spec = importlib.util.spec_from_file_location('long_budget_legacy_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_training_sources(study, plan):
    files = study['files_sha256']
    require(plan['execution_root'] == study['execution_root'], 'Training root differs')
    require(plan['files_sha256'].get(study['train_entry']) == files[study['train_entry']], 'Training entry not bound')
    # Executing helper paths are explicit in the training plan, not guessed from filenames.
    for key, value in plan.items():
        if key.endswith('_source') and isinstance(value, str):
            require(value in files and plan['files_sha256'].get(value) == files[value], 'Unbound/different helper: ' + key)
    for source, digest in plan['files_sha256'].items():
        if Path(source).suffix == '.py':
            require(files.get(source) == digest, 'Study does not pin executed training Python: ' + source)
    return {key: (value, files[value]) for key, value in plan.items()
            if key.endswith('_source') and isinstance(value, str)}


def validate(study, path, common):
    root, output = Path(study['execution_root']), Path(study['output'])
    require(study['purpose'] == 'epo_long_budget_sampling_study' and type(study['execution_ready']) is bool, 'Wrong/unbound study')
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'Invalid execution root')
    require(output.resolve() == output and root / 'artifacts' in output.parents and not output.exists(), 'Fresh output required')
    files = study['files_sha256']
    required = {str(Path(__file__).resolve())} | {study[k] for k in ROLES}
    require(str(Path(path).resolve()) not in files, 'Study cannot hash itself')
    for role in ROLES:
        require(root in Path(study[role]).resolve().parents, 'Executing role outside execution root: ' + role)
    require(study['order'] == list(ARMS) and set(study['train_jobs']) == set(ARMS), 'Fixed frontier/control/clip_higher order required')
    require(set(study['evaluation_specs']) == set(ARMS), 'Evaluation arms differ')
    names = [study['run_id']]; outputs = [str(output)]; shared_helpers = None
    for key in ('entity', 'project', 'group'):
        require(isinstance(study[key], str) and study[key] and '/' not in study[key], 'Explicit identity required')
    for arm in ARMS:
        job = study['train_jobs'][arm]; required.add(job['plan_path'])
        require(files.get(job['plan_path']) == job['plan_sha256'] == sha(job['plan_path']), 'Train plan hash differs')
        plan = load(job['plan_path']); helpers = bind_training_sources(study, plan)
        if shared_helpers is None: shared_helpers = helpers
        require(helpers == shared_helpers, 'Primary arms use different helper source identities')
        require((plan['mode'], plan['arm'], plan['seed'], plan['epochs']) == ('e51', arm, 36001, 6104)
                and plan['stage'] == 'primary' and plan['milestone_epochs'] == list(MILESTONES)
                and plan['curriculum']['warmup_epochs'] == 306 and plan['execution_ready'] is True, 'Fixed long training grid differs')
        names.extend([plan['run_id'], job['monitor_run_id']]); outputs.extend([plan['output'], job['monitor_output']])
        specs = study['evaluation_specs'][arm]
        require([(s['relative_epoch'], s['eval_seed']) for s in specs] == [(m, s) for m in MILESTONES for s in SEEDS], 'Fixed8 milestone/seed evaluations required')
        for item in specs:
            names.extend([item['run_id'], item['monitor_run_id']]); outputs.extend([item['output'], item['monitor_output']])
    expected_analysis = {arm + '-' + str(m) for arm in ('frontier', 'clip_higher') for m in MILESTONES}
    require(set(study['analysis_specs']) == expected_analysis, 'Exactly8 descriptive comparisons required')
    for spec in study['analysis_specs'].values():
        names.append(spec['run_id']); outputs.append(spec['output'])
    seen = []
    for item in study['qualification_completed']:
        require(item['kind'] == 'training', 'Only explicit new training qualifications here')
        required.update(item[k] for k in ('plan_path', 'launch_receipt', 'verification_receipt', 'monitor_wait_receipt'))
        require(files.get(item['plan_path']) == item['plan_sha256'] == sha(item['plan_path']), 'Qualification plan changed')
        plan = load(item['plan_path']); helpers = bind_training_sources(study, plan)
        require(helpers == shared_helpers, 'Qualification uses different helper source identities')
        require(plan['mode'] == 'e51' and plan['stage'] == 'qualification' and plan['execution_ready'] is True
                and plan['curriculum']['warmup_epochs'] == 0, 'Qualification mode/ready differs')
        require((plan['arm'], plan['epochs'], plan['seed'], plan['milestone_epochs']) in
                (('control', 1, 36091, [1]), ('frontier', 129, 36092, [64, 129])), 'Qualification grid differs')
        seen.append((plan['arm'], plan['epochs']))
    require(('control', 1) in seen and ('frontier', 129) in seen and len(seen) == len(set(seen)), 'New control1/frontier129 qualifications required')
    for name, source in study.get('extra_plan_files', {}).items():
        require(Path(name).name == name and name != 'study_plan.json', 'Unsafe attachment name')
        required.add(source)
    require(len(names) == len(set(names)) and all(isinstance(n, str) and n.isalnum() for n in names), 'Duplicate/invalid run ID')
    require(len(outputs) == len(set(outputs)), 'Duplicate future output')
    for target in outputs:
        p = Path(target)
        require(p.is_absolute() and p.resolve() == p and root / 'artifacts' in p.parents and not p.exists(), 'Future output exists/outside root: ' + target)
    require(required <= set(files), 'Missing frozen source/plan/qualification input')
    for source, digest in files.items():
        require(sha(source) == digest, 'Frozen input changed: ' + source)
    return {'status': 'prepared', 'gpu_started': False, 'api_called': False,
            'training_jobs': 3, 'evaluations': 24, 'descriptive_analyses': 8,
            'learning_based_early_stop': False, 'automatic_retry': False}


def verify_training(wandb, study, plan, wait, common):
    require(wait['exit_code'] == 0 and wait['actual_exit_code_source'] == 'subprocess.wait', 'Actual training wait0 required')
    out = Path(plan['output']); summary = load(out / 'summary.json')
    require(load(out / 'plan.json') == plan, 'Executed training plan differs')
    require(summary['status'] == 'completed' and summary['effective_exit_code'] == 0
            and (summary['mode'], summary['arm'], summary['seed']) == (plan['mode'], plan['arm'], plan['seed']), 'Training result scope differs')
    finish = load(out.with_name(out.name + '-finish-attempt.json'))
    require(finish['run_finish_entered'] == finish['run_finish_returned'] == 1
            and finish['native_observer_finish_returned'] and not finish['errors'], 'Owned training finish incomplete')
    config_records = load(out / 'executed_training_config_manifest.json')['files']
    native = [Path(p).parent for p in config_records if p.endswith('/config.yaml')]
    require(len(native) == 1, 'One actual native run directory required')
    wb = load(native[0] / 'wandb_run.json')
    require(wb['id'] == plan['run_id'] and wb['state'] == 'finished' and wb['exit_code'] == 0 and wb['mode'] == 'online', 'Native online run differs')
    for path, expected in config_records.items():
        require({k: v for k, v in common.identity(path).items() if k != 'path'} == expected, 'Executed config changed')
    epochs = plan['epochs']; counts = summary['counts']
    for key, per in (('epoch_entries', 1), ('epoch_returns', 1), ('actor_attempts', 8), ('actor_returns', 8),
                     ('env_step_attempts', 16), ('env_step_returns', 16), ('scheduler_calls', 2),
                     ('scheduler_returns', 2), ('lr_calls', 2), ('lr_returns', 2), ('history_releases', 1)):
        require(counts[key] == epochs * per, 'Training quota differs: ' + key)
    require(summary['physical_frames'] == epochs * 327680 and counts['adam_entered'] == counts['adam_returned']
            and counts['adam_exceptions'] == 0 and summary['actual_adam_updates'] == counts['adam_returned'], 'Physical/Adam accounting differs')
    if epochs in (1, 129):
        require(counts['adam_returned'] > 0, 'Qualification did not execute actual Adam')
    require(load(out / 'terminal-finite.json')['status'] == 'completed', 'Terminal live finite failed')
    milestones = summary['milestone_checkpoints']
    require([m['relative_epoch'] for m in milestones] == plan['milestone_epochs'], 'Milestone list/order differs')
    proofs = [load(out / 'plan-artifact-wait.json'), load(out / 'result-artifact-wait.json')]
    require([p['type'] for p in proofs] == ['long-training-plan', 'long-training-result']
            and all(p['run_id'] == plan['run_id'] for p in proofs), 'Wrong long training artifacts')
    bound = {common.normalized_record(v)['path']: common.normalized_record(v) for v in proofs[1]['files'].values()}
    cfg = out / 'executed_training_config.yaml'; expected_paths = set()
    endpoints = {}
    for milestone in milestones:
        relative = milestone['relative_epoch']; checkpoint = Path(milestone['path'])
        require(milestone['new_frames'] == relative * 327680 and milestone['frame'] == 5000069120 + relative * 327680
                and milestone['epoch'] == 15259 + relative, 'Milestone clock differs')
        require(checkpoint.parent == native[0] / 'nn' and checkpoint.suffix == '.pth', 'Checkpoint outside actual nn directory')
        actual = common.identity(checkpoint)
        require({k: actual[k] for k in ('sha256', 'md5', 'size')} == milestone['identity'], 'Milestone bytes differ')
        require(str(checkpoint) in bound and actual == bound[str(checkpoint)], 'Milestone omitted from final immutable artifact')
        expected_paths.add(checkpoint)
        endpoints[relative] = {'status': 'completed', 'run_id': plan['run_id'], 'checkpoint': str(checkpoint),
                              'checkpoint_sha256': actual['sha256'], 'train_config': str(cfg),
                              'train_config_sha256': sha(cfg), 'training_seed': plan['seed'],
                              'arm': plan['arm'], 'relative_epoch': relative, 'new_frames': relative * 327680}
    require(set((native[0] / 'nn').glob('*.pth')) == expected_paths and len(expected_paths) == len(milestones), 'Native nn contains missing/extra/duplicate checkpoints')
    last = milestones[-1]; terminal = summary['terminal_checkpoint']
    require(all(terminal[k] == last[k] for k in ('path', 'identity', 'frame', 'epoch')), 'Terminal is not final milestone')
    for p in (cfg, out / 'summary.json', out / 'terminal-finite.json'):
        require(str(p) in bound and common.identity(p) == bound[str(p)], 'Missing final evidence: ' + str(p))
    remote = common.verify_remote(wandb, study, plan['run_id'], proofs)
    return {'status': 'completed', 'run_id': plan['run_id'], 'arm': plan['arm'], 'training_seed': plan['seed'],
            'milestones': endpoints, 'remote': remote, 'source_summary_sha256': sha(out / 'summary.json')}


def verify_qualifications(wandb, study, common):
    results = []
    for item in study['qualification_completed']:
        plan, wait, review = load(item['plan_path']), load(item['launch_receipt']), load(item['verification_receipt'])
        require(review['status'] == 'completed' and review['actual_wait0'] is True
                and review['plan_sha256'] == item['plan_sha256'] and review['run_id'] == plan['run_id'], 'Qualification review differs')
        result = verify_training(wandb, study, plan, wait, common)
        result['monitor'] = common.verify_monitor(wandb, study, item, load(item['monitor_wait_receipt']), wait, study['train_entry'])
        if plan['arm'] == 'frontier':
            curriculum = load(Path(plan['output']) / 'summary.json')['curriculum']
            require(curriculum['active_proposals'] > 0 and curriculum['teacher_updates'] > 0, 'Frontier qualification did not exercise curriculum')
            result['curriculum'] = curriculum
        results.append(result)
    return results


def analysis_plan(study, arm, milestone, evaluations, common):
    comparisons = []
    for seed in SEEDS:
        pair = {}
        for role, key in [('control', 'control'), ('candidate', arm)]:
            matches = [v for v in evaluations[key] if v['relative_epoch'] == milestone and v['eval_seed'] == seed]
            require(len(matches) == 1, 'Missing/duplicate fixed evaluation')
            pair[role] = {k: matches[0][k] for k in ('episodes_csv', 'summary_json', 'protocol_json', 'checkpoint_sha256', 'run_id', 'training_run_id')}
        comparisons.append(dict(training_seed=36001, eval_seed=seed, **pair))
    paths = {study['analysis_entry'], study['base_source']}
    for pair in comparisons:
        for role in ('control', 'candidate'):
            paths.update(pair[role][k] for k in ('episodes_csv', 'summary_json', 'protocol_json'))
            paths.add(str(Path(pair[role]['episodes_csv']).parent / 'wandb_run.json'))
    spec = study['analysis_specs'][arm + '-' + str(milestone)]
    return {'stage': 'pilot', 'execution_root': study['execution_root'], 'output': spec['output'],
            'run_id': spec['run_id'], 'entity': study['entity'], 'project': study['project'], 'group': study['group'],
            'files_sha256': {p: sha(p) for p in sorted(paths)}, 'comparisons': comparisons,
            'learning_gate_scope': 'Descriptive legacy statistic only; never controls this fixed3-arm queue',
            'contrast': arm, 'relative_epoch': milestone}


def verify_analysis(wandb, study, plan, path, wait, common):
    require(wait['exit_code'] == 0 and wait['actual_exit_code_source'] == 'subprocess.wait', 'Analysis wait0 required')
    out = Path(plan['output']); receipt, summary = load(out / 'publication.json'), load(out / 'summary.json')
    require(load(path) == plan and receipt['plan_sha256'] == sha(path)
            and receipt['entry_sha256'] == study['files_sha256'][study['analysis_entry']], 'Analysis source/plan differs')
    require(receipt['status'] == 'completed' and receipt['exit_code'] == 0 and receipt['state'] == 'finished'
            and receipt['run_id'] == plan['run_id'] and summary['status'] == 'completed'
            and summary['gpu_validated'] is True and summary['stage'] == 'pilot'
            and summary['contrast'] == plan['contrast'] and summary['relative_epoch'] == plan['relative_epoch'], 'Analysis did not complete')
    require(type(summary['pilot_gate_passed']) is bool, 'Descriptive gate field malformed')
    proofs = [receipt['plan_publication'], receipt['raw_publication']]
    require([p['type'] for p in proofs] == ['dapo-pair-analysis-plan', 'dapo-pair-analysis-results'], 'Analysis artifact types differ')
    require(str(out / 'summary.json') in {common.normalized_record(v)['path'] for v in proofs[1]['files'].values()}, 'Analysis raw omits summary')
    return {'status': 'completed', 'run_id': plan['run_id'], 'descriptive_legacy_gate': summary['pilot_gate_passed'],
            'gate_controls_queue': False, 'summary_sha256': sha(out / 'summary.json'),
            'remote': common.verify_remote(wandb, study, plan['run_id'], proofs)}


def execute(study, path, common):
    out = Path(study['output']); out.mkdir(parents=True)
    shutil.copy2(path, out / 'study_plan.json'); sources = out / 'source'; sources.mkdir()
    for role in ROLES:
        shutil.copy2(study[role], sources / (role + '.py'))
    shutil.copy2(__file__, sources / 'controller.py')
    base = common.module('long_study_publisher', study['base_source'])
    import wandb
    require(wandb.__version__ == '0.24.2' and wandb.run is None, 'Fresh online SDK context required')
    run = None; first = []; completed = []; endpoints = {}; evaluations = {}
    journal = (out / 'phases.jsonl').open('x')
    def remember(exc, where):
        if not first: first.append((exc, exc.__traceback__))
        try:
            with out.with_name(out.name + '-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'utc': common.now(), 'where': where, 'error': repr(exc), 'primary': repr(first[0][0])}) + '\n')
        except BaseException as secondary:
            print('Secondary error record: ' + repr(secondary), file=sys.stderr)
    def unchanged():
        for p, digest in study['files_sha256'].items(): require(sha(p) == digest, 'Frozen input changed: ' + p)
        require(sha(path) == sha(out / 'study_plan.json'), 'Study plan changed')
    def mark(kind, key, result):
        completed.append({'kind': kind, 'key': key, 'result': result})
        journal.write(json.dumps(completed[-1], sort_keys=True) + '\n'); journal.flush()
        common.write(out / 'state.json', {'status': 'running', 'utc': common.now(), 'completed_phase_count': len(completed), 'last_phase': key})
        run.log({'study/phase': key, 'study/phase_status': 'completed', 'study/completed_phases': len(completed)})
    def heartbeat(key, record):
        state = {'status': 'running', 'active_phase': key, 'utc': common.now(), 'pid': record['pid'],
                 'start_ticks': record.get('start_ticks'), 'completed_phase_count': len(completed)}
        common.write(out / 'state.json', state)
        run.summary.update({'study/active_phase': key, 'study/heartbeat_utc': state['utc'], 'study/active_pid': record['pid']})
        run.log({'study/phase': key, 'study/phase_status': 'running', 'study/heartbeat_unix': datetime.now(timezone.utc).timestamp()})
    try:
        run = wandb.init(id=study['run_id'], entity=study['entity'], project=study['project'], group=study['group'],
                         job_type='long-budget-study-controller', mode='online', resume='never', force=True, dir=str(out), config=study)
        require(run is not None and run.id == study['run_id'] and not run.offline and not run.resumed, 'Wrong owned online run')
        pre = {'study_plan.json': out / 'study_plan.json'}
        pre.update({'source/' + p.name: p for p in sources.iterdir()})
        for i, (source, digest) in enumerate(sorted(study['files_sha256'].items())):
            if Path(source).suffix in {'.py', '.json', '.yaml', '.yml', '.md', '.diff', '.patch', '.sh'}:
                pre['bound_inputs/' + str(i) + '_' + Path(source).name] = Path(source)
        common.payload_budget(pre, 64 * 2**20)
        for name, source in study.get('extra_plan_files', {}).items():
            require(name not in pre, 'Prospective attachment collision'); pre[name] = Path(source)
        common.payload_budget(pre, 512 * 2**20)
        common.write(out / 'plan-artifact-wait.json', base.publish(run, pre, 'long-study-plan'))
        mark('qualification', 'completed-before-queue', verify_qualifications(wandb, study, common))
        for arm in study['order']:
            unchanged(); job = study['train_jobs'][arm]; plan = load(job['plan_path']); key = 'train-' + arm
            wait, mwait = common.launch_phase(study, key, study['train_entry'], Path(job['plan_path']), job, lambda record: heartbeat(key, record))
            result = verify_training(wandb, study, plan, wait, common)
            result['monitor'] = common.verify_monitor(wandb, study, job, mwait, wait, study['train_entry'])
            endpoints[arm] = result; mark('training', arm, result)
            for spec in study['evaluation_specs'][arm]:
                unchanged(); milestone = spec['relative_epoch']; endpoint = endpoints[arm]['milestones'][milestone]
                ep = common.evaluation_plan(study, spec, endpoint)
                key = 'eval-' + arm + '-' + str(milestone) + '-' + str(spec['eval_seed'])
                resolved = out / 'resolved_plans'; resolved.mkdir(exist_ok=True); pp = resolved / (key + '.json'); common.write(pp, ep)
                wait, mwait = common.launch_phase(study, key, study['evaluation_entry'], pp, spec, lambda record: heartbeat(key, record))
                result = common.verify_evaluation(wandb, study, ep, wait)
                result.update(relative_epoch=milestone, monitor=common.verify_monitor(wandb, study, spec, mwait, wait, study['evaluation_entry']))
                evaluations.setdefault(arm, []).append(result); mark('evaluation', key, result)
        for arm in ('frontier', 'clip_higher'):
            for milestone in MILESTONES:
                unchanged(); ap = analysis_plan(study, arm, milestone, evaluations, common)
                key = 'analysis-' + arm + '-' + str(milestone); pp = out / 'resolved_plans' / (key + '.json'); common.write(pp, ap)
                wait, _ = common.launch_phase(study, key, study['analysis_entry'], pp, heartbeat=lambda record: heartbeat(key, record))
                mark('analysis', key, verify_analysis(wandb, study, ap, pp, wait, common))
        unchanged()
        require(len(endpoints) == 3 and sum(map(len, evaluations.values())) == 24 and len(completed) == 36, 'Fixed full queue did not finish')
    except BaseException as exc:
        if run is None:
            candidate = wandb.run
            if candidate is not None and candidate.id == study['run_id']: run = candidate
        remember(exc, 'study_phase')
    finally:
        try: journal.close()
        except BaseException as exc: remember(exc, 'journal_close')
        try:
            terminal = {'status': 'failed' if first else 'completed_all_budgets', 'utc': common.now(),
                        'completed_phase_count': len(completed), 'completed_training_arms': list(endpoints),
                        'completed_evaluations': sum(map(len, evaluations.values())),
                        'first_error': repr(first[0][0]) if first else None, 'automatic_retry': False,
                        'learning_based_early_stop': False, 'scope': 'All3 budgets and24 evaluations fixed;8 legacy statistical gates descriptive only. Pre-publication local completion; sibling receipt verifies actual remote finish.'}
            common.write(out / 'summary.json', terminal)
            common.write(out / 'state.json', terminal)  # Close the old running heartbeat before immutable publication.
            if run is not None and run.id == study['run_id']:
                run.summary.update({'study/status': terminal['status'], 'study/completed_phases': len(completed), 'study/active_phase': None})
                result = base.publish(run, common.closed_controller_files(out, journal.closed), 'long-study-result')
                common.write(out / 'result-artifact-wait.json', result)
        except BaseException as exc: remember(exc, 'closed_raw_publication')
        if run is not None and run.id == study['run_id']:
            try: run.finish(exit_code=1 if first else 0)
            except BaseException as exc: remember(exc, 'own_finish')
        if not first and run is not None:
            try:
                remote = common.verify_remote(wandb, study, study['run_id'], [load(out / 'plan-artifact-wait.json'), load(out / 'result-artifact-wait.json')])
                common.write(out.with_name(out.name + '-publication.json'), {'status': 'completed', 'state': 'finished',
                             'run_id': study['run_id'], 'plan_sha256': sha(path), 'remote': remote, 'scope': 'completed_all_budgets'})
            except BaseException as exc: remember(exc, 'fresh_controller_readback')
        if first:
            try: common.write(out.with_name(out.name + '-failed.json'), {'status': 'failed', 'run_id': study['run_id'],
                              'first_error': repr(first[0][0]), 'closed_raw_retained': True})
            except BaseException as secondary: print('Failure receipt write: ' + repr(secondary), file=sys.stderr)
    if first:
        exc, tb = first[0]; raise exc.with_traceback(tb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, 'Study plan SHA differs')
    study = load(args.plan); common = load_legacy(study); report = validate(study, args.plan, common)
    if not args.execute:
        print(json.dumps(report, indent=2)); return
    require(study['execution_ready'] is True and Path.cwd() == Path(study['execution_root']), 'Unbound study or cwd mismatch')
    require(shutil.disk_usage(study['execution_root']).free >= 4 * 2**30, 'Startup disk below4GiB')
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(_signal, _frame):
        raise KeyboardInterrupt('Long study SIGTERM; preserve actual child wait then stop')
    with open('/tmp/epo-long-study-controller.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); signal.signal(signal.SIGTERM, terminate)
        try: execute(study, args.plan, common)
        finally: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
