#!/usr/bin/env python3
"""Plan-bound native EPO evaluation. Default is CPU-only preparation; no retries."""
import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

EVALUATOR_SHA = '677ced2d8475fbf5a1f553bd3e8ff46ae65ecefcca815acb40b60569abee8b82'
FIXED = dict(genes='63', objects_per_gene=0, replicas_per_object=1, object_seed=20260913,
             max_steps=30000, max_envs=947, tolerance=.01, action_mode='stochastic', device='cuda:0')
FLOORS = dict(driver_free_bytes=2**30, disk_free_bytes=2**30+128*2**20,
              host_available_bytes=4*2**30)
CAP = 32*2**20


def require(value, message):
    if not value:
        raise RuntimeError(message)


def record(path):
    h, m, size = hashlib.sha256(), hashlib.md5(), 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk); m.update(chunk); size += len(chunk)
    return dict(local_path=str(Path(path)), sha256=h.hexdigest(), size=size,
                md5_base64=base64.b64encode(m.digest()).decode())


def write(path, value):
    with Path(path).open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate(plan, plan_path, executing=False):
    root, output = Path(plan['execution_root']), Path(plan['output'])
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'Invalid isolated execution root')
    require(output.is_absolute() and output.resolve() == output and root/'artifacts' in output.parents,
            'Output must be a new path under execution_root/artifacts')
    require(not output.exists(), 'Existing output; no resume or automatic retry')
    require(plan['purpose'] == 'dapo_frozen_master_evaluation', 'Wrong plan purpose')
    require(type(plan['execution_ready']) is bool, 'Explicit execution_ready required')
    require(not executing or plan['execution_ready'], 'Draft plan cannot execute')
    require(plan.get('settings', FIXED) == FIXED, 'Fixed 947/master/stochastic protocol changed')
    require(type(plan['eval_seed']) is int and 0 < plan['eval_seed'] < 2**31, 'Invalid evaluation seed')
    for key in ('run_id', 'source_training_run_id'):
        require(isinstance(plan[key], str) and plan[key].isalnum(), 'Explicit alphanumeric '+key+' required')
    for key in ('entity', 'project', 'group'):
        require(isinstance(plan[key], str) and plan[key] and '/' not in plan[key], 'Explicit W&B '+key+' required')
    evaluator = root/'scripts/evaluate_frozen_checkpoint.py'
    require(Path(plan['evaluator_source']) == evaluator and record(evaluator)['sha256'] == EVALUATOR_SHA,
            'Native evaluator bytes must remain unchanged')
    required = {str(Path(__file__).resolve()), str(evaluator), plan['nvml_source'], plan['train_config']}
    require(required <= set(plan['files_sha256']), 'Unbound adapter/helper/config')
    require(str(Path(plan_path).resolve()) not in plan['files_sha256'], 'Plan must not pin itself')
    for key in ('nvml_source', 'train_config'):
        require(Path(plan[key]).is_absolute(), 'Absolute source/config paths required')
    require(root in Path(plan['nvml_source']).resolve().parents, 'NVML helper must be inside execution root')
    for path, digest in plan['files_sha256'].items():
        if path == plan['checkpoint'] and not executing:
            require(digest == plan['checkpoint_sha256'], 'Checkpoint bindings disagree')
            continue
        require(Path(path).is_absolute() and record(path)['sha256'] == digest, 'Changed bound input: '+path)
    require(plan['files_sha256'][plan['train_config']] == plan['train_config_sha256'], 'Config SHA mismatch')
    require(Path(plan['checkpoint']).is_absolute() and len(plan['checkpoint_sha256']) == 64, 'Explicit checkpoint and SHA required')
    if executing:
        require(record(plan['checkpoint'])['sha256'] == plan['checkpoint_sha256'], 'Checkpoint SHA mismatch')
    limits = plan.get('startup_limits', FLOORS)
    require(set(limits) == set(FLOORS) and all(type(limits[k]) is int and limits[k] >= FLOORS[k] for k in FLOORS),
            'Startup resource floors cannot be reduced')
    return dict(status='prepared', execution_ready=plan['execution_ready'], gpu_executed=False,
                api_called=False, checkpoint_bytes_read=executing, settings=FIXED,
                maximum_environment_control_frames=947*30000, source_count=len(plan['files_sha256']))


def host_fields(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'MemAvailable'):
            result[key] = int(value.split()[0])*1024
    return result


class Journal:
    def __init__(self, output, root, nvml):
        self.path, self.root, self.nvml = output/'evaluation_resources.jsonl', root, nvml
        self.stream = self.path.open('x')
        self.attempts = self.returns = self.samples = 0
        self.closed = False
        self.start = time.monotonic()

    def sample(self, boundary):
        row = dict(boundary=boundary, monotonic=time.monotonic(), elapsed_seconds=time.monotonic()-self.start,
                   env_step_attempts=self.attempts, env_step_returns=self.returns,
                   actual_environment_control_frames=self.returns*947,
                   disk_free_bytes=shutil.disk_usage(self.root).free,
                   host_available_bytes=host_fields('/proc/meminfo')['MemAvailable'],
                   **host_fields('/proc/self/status'), **self.nvml())
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        self.stream.flush(); self.samples += 1
        return row

    def close(self):
        if not self.closed:
            try:
                self.stream.flush(); os.fsync(self.stream.fileno())
            finally:
                self.stream.close(); self.closed = True

    def counts(self):
        return dict(env_step_attempts=self.attempts, env_step_returns=self.returns,
                    global_control_steps=self.returns, actual_environment_control_frames=947*self.returns,
                    env_step_attempts_without_return=self.attempts-self.returns,
                    unreturned_step_may_have_advanced_physics=self.attempts != self.returns,
                    resource_samples=self.samples, resource_journal_closed=self.closed,
                    physical_count_scope='947 environments per actual env.step return, including later episodes; unreturned calls have unknown additional cost.',
                    resource_sampling_scope='NVML whole GPU0 memory and process RSS/HWM, host available RAM and disk; startup/250 returned steps/terminal samples, not continuous peaks or energy; stops before native summary/publication.')


class Patches:
    def __init__(self):
        self.items = []

    def set(self, obj, key, value):
        old, present = getattr(obj, key), key in obj.__dict__
        self.items.append((obj, key, present, obj.__dict__.get(key)))
        setattr(obj, key, value)
        return old

    def restore(self, remember):
        for obj, key, present, old in reversed(self.items):
            try:
                if present: setattr(obj, key, old)
                else: delattr(obj, key)
            except BaseException as exc:
                remember(exc, 'restore.'+key)


def execute(plan, plan_path, plan_sha):
    root, output = Path(plan['execution_root']), Path(plan['output'])
    errors_path = output.with_name(output.name+'-adapter-errors.jsonl')
    receipt_path = output.with_name(output.name+'-adapter-receipt.json')
    state = dict(first=None, run=None, finish_entered=False, finish_returned=False,
                 artifacts=[], sealed=False, payload_frozen=False, plan_waited=False, journal=None)
    patches = Patches()

    def remember(exc, where):
        if state['first'] is None: state['first'] = (exc, exc.__traceback__)
        try:
            with errors_path.open('a') as stream:
                stream.write(json.dumps(dict(where=where, error=repr(exc),
                    first_error=repr(state['first'][0]), monotonic=time.monotonic()))+'\n')
                stream.flush()
        except BaseException as secondary:
            print('Error journal also failed: '+repr(secondary), file=sys.stderr)

    def capture_active(where):
        if sys.exc_info()[1] is not None: remember(sys.exc_info()[1], where)

    def seal_journal(boundary):
        j = state['journal']
        if j is not None and not j.closed:
            try: j.sample(boundary)
            except BaseException as exc: remember(exc, 'resource_sample')
            try: j.close()
            except BaseException as exc: remember(exc, 'resource_close')

    def native_finish(exit_code=0, **kwargs):
        capture_active('native_exception')
        if state['finish_entered']:
            return None
        failed = exit_code != 0 or state['first'] is not None
        if failed:
            seal_journal('failure')
            # A committed native payload is immutable; late failure is only an external receipt.
            if not state['payload_frozen']:
                try:
                    write(output/'adapter_failure.json', dict(status='failed', first_error=repr(state['first'][0]) if state['first'] else 'native failure',
                          counts=state['journal'].counts() if state['journal'] else None))
                    files = [p for p in sorted(output.iterdir()) if p.is_file() and p.name != 'wandb_run.json']
                    require(sum(p.stat().st_size for p in files) <= CAP, 'Failed partial payload exceeds32MiB; originals stay local')
                    art = wandb.Artifact('dapo-evaluation-failure-'+plan['run_id'], type='dapo-evaluation-failure')
                    for p in files: art.add_file(str(p), name=p.name)
                    state['run'].log_artifact(art).wait()
                except BaseException as exc: remember(exc, 'failure_partial_upload')
        state['finish_entered'] = True
        try:
            result = old_finish(exit_code=1 if failed else 0, **kwargs)
            state['finish_returned'] = True
            return result
        except BaseException as exc:
            remember(exc, 'owned_finish'); raise

    try:
        os.chdir(str(root)); sys.path[:0] = [str(root/'rl_games'), str(root)]
        # No CUDA work may precede the caller's nonblocking lock and idle/resource gates.
        import isaacgym  # noqa: F401 -- mandatory before Torch
        import torch
        import wandb
        from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
        require(torch.__version__ == '1.8.1' and torch.version.cuda == '11.1', 'Unexpected numerical runtime')
        require(wandb.__version__ == '0.24.2' and wandb.run is None, 'Unexpected SDK or pre-existing W&B run')
        native = module('dapo_native_frozen_evaluator', plan['evaluator_source'])
        require(native.ROOT.resolve() == root and root in Path(sys.modules[AllegroKukaReorientation.__module__].__file__).resolve().parents,
                'Imported evaluator/task outside execution root')
        nvml = module('dapo_eval_nvml', plan['nvml_source']).NvmlMemory()
        args = dict(FIXED, checkpoint=plan['checkpoint'], train_config=plan['train_config'], output_dir=str(output),
                    eval_seed=plan['eval_seed'], wandb_entity=plan['entity'], wandb_project=plan['project'],
                    wandb_group=plan['group'], source_run_url='https://wandb.ai/{}/{}/runs/{}'.format(plan['entity'],plan['project'],plan['source_training_run_id']))
        from types import SimpleNamespace
        patches.set(native, 'parse_args', lambda: SimpleNamespace(**args))

        def snapshot(directory):
            result = old_snapshot(directory)
            copies = {'evaluation_plan.json': plan_path, 'evaluation_adapter.py': __file__,
                      'native_evaluator.py': plan['evaluator_source'], 'nvml_source.py': plan['nvml_source'],
                      'training_config.input.yaml': plan['train_config']}
            for name, source in copies.items(): shutil.copyfile(str(source), str(directory/name))
            shutil.copyfile(str(output.with_name(output.name+'-startup.json')), str(directory/'adapter_startup.json'))
            write(directory/'adapter_bindings.json', dict(plan_sha256=plan_sha, files_sha256=plan['files_sha256'],
                source_training_run_id=plan['source_training_run_id'], source_copies={k:record(directory/k) for k in copies},
                additional_bound_files_scope='SHA-bound original paths; repository sources are included by native source_snapshot, no checkpoint copied.'))
            state['journal'] = Journal(directory, root, nvml)
            state['journal'].sample('native_output_initialized')
            return result
        old_snapshot = patches.set(native, 'source_snapshot', snapshot)

        def write_native(path, value):
            capture_active('native_write_during_exception')
            name = Path(path).name
            if name == 'protocol.json':
                require(value['num_envs'] == 947 and value['gene_count'] == 64 and value['selected_genes'] == [63]
                        and value['success_cap'] == 50 and value['checkpoint_sha256'] == plan['checkpoint_sha256']
                        and value['training_config_sha256'] == plan['train_config_sha256'], 'Actual native quota/config differs')
                require(value['object_distribution'] == 'full generated object catalog', 'Full947 catalog required')
                value.update(adapter_plan_sha256=plan_sha, adapter_source_sha256=record(__file__)['sha256'],
                    native_evaluator_sha256=EVALUATOR_SHA, source_training_run_id=plan['source_training_run_id'], planned_run_id=plan['run_id'])
            if name == 'summary.json':
                require(state['plan_waited'], 'Evaluation plan was not committed before model use')
                j = state['journal']
                require(value['episode_count'] == 947 and value['normalizer_unchanged'] and value['gene_assignment_validated']
                        and j.attempts == j.returns and 0 < j.returns <= 30000, 'Native evaluation/control-step quota failed')
                seal_journal('before_native_summary')
                require(state['first'] is None, 'Resource journal finalization failed')
                require(record(plan_path)['sha256'] == plan_sha, 'Plan changed during evaluation')
                for source, digest in plan['files_sha256'].items():
                    require(record(source)['sha256'] == digest, 'Bound source/config changed during evaluation')
                require(record(plan['checkpoint'])['sha256'] == plan['checkpoint_sha256'], 'Checkpoint changed during evaluation')
                value.update(j.counts(), adapter_plan_sha256=plan_sha, source_training_run_id=plan['source_training_run_id'])
            return old_write(path, value)
        old_write = patches.set(native, 'write_json', write_native)

        def step(env, *a, **kw):
            j = state['journal']
            require(state['plan_waited'] and j is not None and not j.closed and env.num_envs == 947, 'Unplanned environment step')
            j.attempts += 1
            try: result = old_step(env, *a, **kw)
            except BaseException as exc: remember(exc, 'env.step'); raise
            j.returns += 1
            if j.returns % 250 == 0: j.sample('250_returned_steps')
            return result
        old_step = patches.set(AllegroKukaReorientation, 'step', step)

        def init(**kwargs):
            nonlocal old_finish
            kwargs.update(id=plan['run_id'], entity=plan['entity'], project=plan['project'], group=plan['group'],
                          mode='online', resume='never', force=True)
            try:
                run = old_init(**kwargs)
            except BaseException:
                candidate = wandb.run
                if candidate is not None and candidate.id == plan['run_id']:
                    state['run'] = candidate
                    old_finish = patches.set(candidate, 'finish', native_finish)
                raise
            if run.id == plan['run_id']:
                state['run'] = run
                old_finish = patches.set(run, 'finish', native_finish)
            require(state['run'] is run and run.settings.mode == 'online' and not run.offline and not run.resumed
                    and run.entity == plan['entity'] and run.project == plan['project'], 'Own fresh online evaluation run required')
            original_log = run.log_artifact

            def log_artifact(artifact, *a, **kw):
                before = {name: record(output/name) for name in artifact.manifest.entries}
                if artifact.type == 'evaluation': state['payload_frozen'] = True
                pending = original_log(artifact, *a, **kw)
                class Wait:
                    def wait(self, *wa, **wk):
                        logged = pending.wait(*wa, **wk)
                        require(logged.state == 'COMMITTED', 'Artifact wait did not commit')
                        require(set(logged.manifest.entries) == set(before), 'Waited artifact inventory differs')
                        files = {}
                        for name, entry in logged.manifest.entries.items():
                            local = output/name
                            require(local.is_file(), 'Artifact has no local original: '+name)
                            item = record(local)
                            require(item == before[name], 'Artifact original changed during upload')
                            require(item['md5_base64'] == entry.digest and item['size'] == entry.size, 'Waited artifact differs from local bytes')
                            files[name] = item
                        state['artifacts'].append(dict(name=logged.name, type=logged.type, state=logged.state, files=files))
                        if logged.type == 'evaluation': state['sealed'] = True
                        return logged
                return Wait()
            patches.set(run, 'log_artifact', log_artifact)
            artifact = wandb.Artifact('dapo-evaluation-plan-'+run.id, type='dapo-evaluation-plan')
            for name in ('evaluation_plan.json','evaluation_adapter.py','native_evaluator.py','nvml_source.py','training_config.input.yaml','adapter_bindings.json','adapter_startup.json'):
                artifact.add_file(str(output/name), name=name)
            run.log_artifact(artifact).wait()
            state['plan_waited'] = True
            return run
        old_finish = None
        old_init = patches.set(wandb, 'init', init)
        native.main()
        require(state['finish_returned'] and state['first'] is None and state['sealed'], 'Native evaluator did not fully terminate')
    except BaseException as exc:
        remember(exc, 'adapter_execute')
        if state['run'] is not None and not state['finish_entered']:
            try: native_finish(exit_code=1)
            except BaseException as secondary: remember(secondary, 'fallback_finish')
    finally:
        seal_journal('adapter_cleanup')
        patches.restore(remember)
        receipt = dict(status='failed' if state['first'] else 'completed', exit_code=1 if state['first'] else 0,
            run_id=plan['run_id'], plan_sha256=plan_sha, plan_waited_before_environment=state['plan_waited'],
            owned_finish_entered=state['finish_entered'], owned_finish_returned=state['finish_returned'],
            first_error=repr(state['first'][0]) if state['first'] else None,
            counts=state['journal'].counts() if state['journal'] else None, artifacts=state['artifacts'],
            fresh_api_verified=False, fresh_api_verification_owner='parent controller',
            late_failure_scope='External receipt/errors only after committed native payload; originals retained, no claim every failure partial uploaded.')
        try: write(receipt_path, receipt)
        except BaseException as exc: remember(exc, 'external_receipt')
    if state['first'] is not None:
        exc, tb = state['first']; raise exc.with_traceback(tb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(record(args.plan)['sha256'] == args.plan_sha256, 'Plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    prepared = validate(plan, args.plan, args.execute)
    if not args.execute:
        print(json.dumps(prepared, indent=2)); return
    root, output = Path(plan['execution_root']), Path(plan['output'])
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1', 'Require explicit GPU0/no-cache allocator environment')
    with open('/tmp/epo-gpu-0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()
        require(not active, 'GPU0 has an active compute process; no retry/kill')
        nvml = module('dapo_eval_startup_nvml', plan['nvml_source']).NvmlMemory()
        resource = dict(**nvml(), disk_free_bytes=shutil.disk_usage(root).free,
                        host_available_bytes=host_fields('/proc/meminfo')['MemAvailable'])
        require(all(resource[k] >= v for k,v in plan.get('startup_limits', FLOORS).items()), 'Insufficient startup resource reserve')
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output.with_name(output.name+'-startup.json'), dict(resources=resource, limits=plan.get('startup_limits', FLOORS), pid=os.getpid(), plan_sha256=args.plan_sha256))
        previous = signal.getsignal(signal.SIGTERM)
        def terminated(signum, frame): raise KeyboardInterrupt('SIGTERM: preserve evaluation failure and stop')
        signal.signal(signal.SIGTERM, terminated)
        try: execute(plan, args.plan, args.plan_sha256)
        finally: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
