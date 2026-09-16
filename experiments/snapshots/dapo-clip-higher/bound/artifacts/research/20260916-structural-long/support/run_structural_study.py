#!/usr/bin/env python3
"""Fixed two-factor, separate-arm 2B extension with an existing matched control.

Three technical qualifications precede both learning runs. No reward-based gate,
automatic training retry, checkpoint selection, or concurrent GPU execution.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import time

ARMS = ('clip_higher', 'owned_clip')
MILESTONES = (306, 1526, 3052, 6104)
SEEDS = (17001, 17002)


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def validate(s, pp):
    assert s['purpose'] == 'epo_fixed_structural_long_extension'
    assert s['order'] == list(ARMS)
    assert not Path(s['output']).exists()
    assert Path(s['execution_root']).is_absolute()
    assert s['training_seed'] == 36001
    assert s['independent_training_seeds'] == 1
    assert s['reference_training_run_id'] == '4742038629e04897'
    assert s['reference_evaluations'] == load(s['recovery_all_evaluations'])['control']
    original = load(s['reference_training_summary'])
    checkpoints = {v['relative_epoch']: v for v in original['milestone_checkpoints']}
    assert original['status'] == 'completed' and original['arm'] == 'control' and original['seed'] == 36001
    assert len(s['reference_evaluations']) == 8
    assert {(v['relative_epoch'], v['eval_seed']) for v in s['reference_evaluations']} == {(m, seed) for m in MILESTONES for seed in SEEDS}
    assert load(s['recovery_publication'])['status'] == 'completed'
    fixture = load(s['cuda_fixture_result'])
    assert fixture['status'] == 'completed' and fixture['gpu_validated'] is True
    assert fixture['helper_sha256'] == s['files_sha256'][s['clip_source']]
    assert fixture['entry_sha256'] == s['files_sha256'][s['cuda_fixture_entry']]
    assert len(fixture['checks']) == 7 and all(v is True for v in fixture['checks'].values())
    assert fixture['counts'] == {'branches': 15, 'setup_adam_returns': 15, 'scaler_initialization_calls': 15,
        'unscale_returns': 17, 'clip_returns': 17, 'scaler_step_returns': 17,
        'actual_adam_returns': 15, 'scaler_update_returns': 17, 'sampled_hook_rows': 8}
    fixture_wait = load(s['cuda_fixture_actual_wait'])
    assert fixture_wait['exit_code'] == 0 and fixture_wait['actual_exit_code_source'] == 'subprocess.wait'
    roles = ('train_entry', 'analysis_entry', 'evaluation_entry', 'evaluator_source', 'nvml_source',
             'monitor_entry', 'monitor_source', 'base_source', 'legacy_controller_source',
             'long_controller_source', 'clip_source')
    assert {s[k] for k in roles} <= set(s['files_sha256'])
    reference_paths = {s['recovery_all_evaluations'], s['recovery_publication'],
                       s['reference_training_summary'], s['cuda_fixture_result'], s['cuda_fixture_entry'],
                       s['cuda_fixture_actual_wait'], str(Path(__file__).resolve())}
    for v in s['reference_evaluations']:
        assert v['training_run_id'] == s['reference_training_run_id'] and v['status'] == 'completed'
        assert v['checkpoint_sha256'] == checkpoints[v['relative_epoch']]['identity']['sha256']
        for k in ('episodes_csv', 'summary_json', 'protocol_json'): reference_paths.add(v[k])
        reference_paths.add(str(Path(v['episodes_csv']).parent / 'wandb_run.json'))
    assert reference_paths <= set(s['files_sha256'])
    jobs = s['qualification_jobs'] + [s['train_jobs'][a] for a in ARMS]
    ids = [s['run_id']]
    outputs = [s['output']]
    assert len(s['qualification_jobs']) == 3
    for i, j in enumerate(jobs):
        p = load(j['plan_path'])
        assert sha(j['plan_path']) == j['plan_sha256'] == s['files_sha256'][j['plan_path']]
        assert p['execution_root'] == s['execution_root'] and p['execution_ready']
        assert p['purpose'] == 'epo_structural_training'
        assert p['files_sha256'][s['train_entry']] == s['files_sha256'][s['train_entry']]
        if i < 3:
            assert (p['arm'], p['seed'], p['epochs'], p['milestone_epochs'], p['stage']) == (
                ('control', 'clip_higher', 'owned_clip')[i], (36191, 36192, 36193)[i], 1, [1], 'qualification')
        else:
            assert (p['arm'], p['seed'], p['epochs'], p['milestone_epochs'], p['stage']) == (
                ARMS[i - 3], 36001, 6104, list(MILESTONES), 'primary')
        ids += [p['run_id'], j['monitor_run_id']]
        outputs += [p['output'], j['monitor_output']]
        for source, digest in p['files_sha256'].items():
            assert s['files_sha256'].get(source) == digest
    for a in ARMS:
        assert [(v['relative_epoch'], v['eval_seed']) for v in s['evaluation_specs'][a]] == [(m, seed) for m in MILESTONES for seed in SEEDS]
        for v in s['evaluation_specs'][a]:
            ids += [v['run_id'], v['monitor_run_id']]
            outputs += [v['output'], v['monitor_output']]
        for m in MILESTONES:
            v = s['analysis_specs'][a + '-' + str(m)]
            ids.append(v['run_id']); outputs.append(v['output'])
    assert len(ids) == len(set(ids)) and all(v.isalnum() for v in ids)
    assert len(outputs) == len(set(outputs))
    assert all(not Path(p).exists() and Path(s['execution_root']) / 'artifacts' in Path(p).parents for p in outputs)
    assert str(pp.resolve()) not in s['files_sha256']
    for p, h in s['files_sha256'].items():
        assert sha(p) == h, p
    return {'status': 'prepared', 'qualification_jobs': 3, 'training_jobs': 2,
            'new_evaluations': 16, 'descriptive_comparisons': 8,
            'automatic_retry': False, 'learning_gate': False, 'gpu_started': False}


def execute(s, pp):
    c = module('structural_common', s['legacy_controller_source'])
    long = module('structural_long_verify', s['long_controller_source'])
    base = module('structural_publish', s['base_source'])
    out = Path(s['output']); out.mkdir()
    (out / 'resolved_plans').mkdir()
    shutil.copy2(pp, out / 'study_plan.json')
    shutil.copy2(__file__, out / 'controller_source.py')
    import wandb
    native_verify_remote = c.verify_remote
    # Retry visibility only. The original strict byte verifier remains the final
    # authority and a visible inventory/digest/size mismatch is never retried.
    def consistent_remote(wb, study, run_id, proofs):
        deadline = time.monotonic() + 120
        log = out.with_name(out.name + '-remote-readbacks.jsonl')
        expected = {p['artifact'] for p in proofs}
        while True:
            record = {'utc': c.now(), 'run_id': run_id, 'state': None, 'artifacts': [], 'error': None}
            visible = False
            try:
                remote = wb.Api(timeout=10).run('/'.join([s['entity'], s['project'], run_id]))
                artifacts = list(remote.logged_artifacts())
                record['state'] = remote.state
                record['artifacts'] = [{'name': a.name, 'state': a.state} for a in artifacts]
                visible = expected <= {a.name for a in artifacts if a.state == 'COMMITTED'}
            except Exception as exc:
                record['error'] = repr(exc)
            with log.open('a') as f:
                f.write(json.dumps(record, sort_keys=True) + '\n')
            if visible and record['state'] == 'finished':
                return native_verify_remote(wb, study, run_id, proofs)
            if time.monotonic() >= deadline:
                raise TimeoutError('Read-only status visibility budget expired; no experiment retry')
            time.sleep(2)
    c.verify_remote = consistent_remote
    run = None; first = []; completed = []; evaluations = {'control': s['reference_evaluations']}
    journal = (out / 'phases.jsonl').open('x')
    def remember(exc, where):
        if not first: first.append((exc, exc.__traceback__))
        try:
            with out.with_name(out.name + '-errors.jsonl').open('a') as f:
                f.write(json.dumps({'utc': c.now(), 'where': where, 'error': repr(exc)}) + '\n')
        except BaseException as secondary:
            print('Secondary error journal failure: ' + repr(secondary), file=sys.stderr)
    def unchanged():
        for p, h in s['files_sha256'].items(): c.require(sha(p) == h, 'Frozen input changed: ' + p)
        c.require(sha(pp) == sha(out / 'study_plan.json'), 'Study plan changed')
    def mark(kind, key, result):
        completed.append({'kind': kind, 'key': key, 'result': result})
        journal.write(json.dumps(completed[-1], sort_keys=True) + '\n'); journal.flush()
        c.write(out / 'state.json', dict(status='running', utc=c.now(), last_phase=key, completed_phase_count=len(completed)))
        run.log({'study/phase': key, 'study/completed_phases': len(completed)})
    def heartbeat(key, record):
        c.write(out / 'state.json', dict(status='running', utc=c.now(), active_phase=key,
                pid=record['pid'], start_ticks=record.get('start_ticks'), completed_phase_count=len(completed)))
        run.summary.update({'study/active_phase': key, 'study/heartbeat_utc': c.now()})
    def train_job(job, kind):
        unchanged(); p = load(job['plan_path']); key = kind + '-' + p['arm']
        w, mw = c.launch_phase(s, key, s['train_entry'], Path(job['plan_path']), job, lambda r: heartbeat(key, r))
        result = long.verify_training(wandb, s, p, w, c)
        result['monitor'] = c.verify_monitor(wandb, s, job, mw, w, s['train_entry'])
        mark(kind, p['arm'], result)
        return result
    try:
        run = wandb.init(id=s['run_id'], entity=s['entity'], project=s['project'], group=s['group'],
                job_type='structural-long-study-controller', mode='online', force=True, resume='never', dir=str(out), config=s)
        assert run.id == s['run_id'] and not run.offline and not run.resumed
        files = {'study_plan.json': out / 'study_plan.json', 'controller_source.py': out / 'controller_source.py'}
        for i, p in enumerate(sorted(s['files_sha256'])):
            if Path(p).suffix in ('.py', '.json', '.jsonl', '.yaml', '.md', '.patch', '.pt', '.log'):
                files['inputs/%04d_%s' % (i, Path(p).name)] = Path(p)
        c.payload_budget(files, 512 * 2**20)
        c.write(out / 'plan-artifact-wait.json', base.publish(run, files, 'structural-study-plan'))
        for job in s['qualification_jobs']: train_job(job, 'qualification')
        for arm in ARMS:
            trained = train_job(s['train_jobs'][arm], 'training')
            evaluations[arm] = []
            for spec in s['evaluation_specs'][arm]:
                unchanged(); m = spec['relative_epoch']
                p = c.evaluation_plan(s, spec, trained['milestones'][m])
                key = 'eval-%s-%d-%d' % (arm, m, spec['eval_seed'])
                path = out / 'resolved_plans' / (key + '.json'); c.write(path, p)
                w, mw = c.launch_phase(s, key, s['evaluation_entry'], path, spec, lambda r: heartbeat(key, r))
                result = c.verify_evaluation(wandb, s, p, w)
                result.update(relative_epoch=m, monitor=c.verify_monitor(wandb, s, spec, mw, w, s['evaluation_entry']))
                evaluations[arm].append(result); mark('evaluation', key, result)
            for m in MILESTONES:
                unchanged(); p = long.analysis_plan(s, arm, m, evaluations, c)
                key = 'analysis-%s-%d' % (arm, m)
                path = out / 'resolved_plans' / (key + '.json'); c.write(path, p)
                w, _ = c.launch_phase(s, key, s['analysis_entry'], path, heartbeat=lambda r: heartbeat(key, r))
                mark('analysis', key, long.verify_analysis(wandb, s, p, path, w, c))
        unchanged()
        assert len(completed) == 29
        c.write(out / 'all_evaluations.json', evaluations)
    except BaseException as exc:
        if run is None and wandb.run is not None and wandb.run.id == s['run_id']: run = wandb.run
        remember(exc, 'study')
    finally:
        try: journal.close()
        except BaseException as exc: remember(exc, 'journal_close')
        terminal = dict(status='failed' if first else 'completed_all_fixed_budgets', utc=c.now(),
                completed_phase_count=len(completed), first_error=repr(first[0][0]) if first else None,
                automatic_training_retry=False, reward_based_stop=False, independent_training_seeds=1,
                reference_training_run_id=s['reference_training_run_id'])
        try:
            c.write(out / 'summary.json', terminal); c.write(out / 'state.json', terminal)
        except BaseException as exc: remember(exc, 'terminal_write')
        if run is not None and run.id == s['run_id']:
            try:
                run.summary.update(terminal)
                c.write(out / 'result-artifact-wait.json', base.publish(run, c.closed_controller_files(out, journal.closed), 'structural-study-result'))
            except BaseException as exc: remember(exc, 'closed_publication')
            try: run.finish(exit_code=1 if first else 0)
            except BaseException as exc: remember(exc, 'own_finish')
        if not first:
            try:
                remote = c.verify_remote(wandb, s, s['run_id'], [load(out / 'plan-artifact-wait.json'), load(out / 'result-artifact-wait.json')])
                c.write(out.with_name(out.name + '-publication.json'), dict(status='completed', run_id=s['run_id'], remote=remote, plan_sha256=sha(pp)))
            except BaseException as exc: remember(exc, 'final_readback')
    if first:
        exc, tb = first[0]; raise exc.with_traceback(tb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args(); assert sha(a.plan) == a.plan_sha256
    s = load(a.plan); report = validate(s, a.plan)
    if not a.execute: print(json.dumps(report)); return
    assert s['execution_ready'] and Path.cwd() == Path(s['execution_root'])
    assert shutil.disk_usage(s['execution_root']).free >= s['startup_required_free_bytes']
    with open('/tmp/epo-long-study-controller.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(_signal, _frame): raise KeyboardInterrupt('Stop owned study and preserve actual child wait')
        signal.signal(signal.SIGTERM, stop)
        execute(s, a.plan)


if __name__ == '__main__': main()
