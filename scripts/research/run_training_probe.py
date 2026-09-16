#!/usr/bin/env python3
"""One source-bound native EPO epoch with passive training-chain probes.

Use a script entrypoint and the epo environment. This command does not launch
other jobs, retry failures, choose checkpoints or change learning objectives.
"""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / 'artifacts/research/20260914-training-chain'
E51_SHA = 'de220625c407b463d41f9242cd652a40f5e48c253b4b9225b7817e452cf0f6be'
START_FRAME = 5000069120
QUANTUM = 327680


def require(value, message):
    if not value:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Changing file: ' + str(path))
    return {'sha256': sha.hexdigest(), 'md5': base64.b64encode(md5.digest()).decode(), 'size': after.st_size}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def secondary_report(path, record):
    try:
        write(path, record)
    except BaseException as exc:
        try:
            print('Secondary report failure: ' + repr(exc), file=sys.stderr)
        except BaseException:
            pass


def publish(run, files, label):
    import wandb
    records = {name: dict(path=str(path), **identity(path)) for name, path in files.items()}
    artifact = wandb.Artifact(label + '-' + run.id, type=label)
    for name, item in records.items():
        artifact.add_file(item['path'], name=name, policy='immutable', skip_cache=True)
    committed = run.log_artifact(artifact).wait()
    require(committed.state == 'COMMITTED', 'Artifact did not commit')
    entries = committed.manifest.entries
    require(set(entries) == set(records), 'Artifact file inventory mismatch')
    for name, item in records.items():
        require(entries[name].digest == item['md5'] and entries[name].size == item['size'], 'Artifact bytes mismatch: ' + name)
        require(identity(item['path']) == {k: item[k] for k in ('sha256', 'md5', 'size')}, 'Local payload changed during upload')
    return {'run_id': run.id, 'artifact': committed.name, 'type': label, 'state': committed.state, 'files': records}


def execute(plan, plan_path, output):
    # Import simulator before Torch, project helpers, or probe modules.
    import isaacgym  # noqa: F401
    import torch
    from omegaconf import OmegaConf
    from isaacgymenvs import train as native
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from rl_games.common.a2c_common import A2CBase
    from scripts.evaluate_frozen_checkpoint import source_snapshot
    from scripts.train_clipping_ablation import release_completed_rollout_history

    require(torch.__version__ == '1.8.1' and torch.version.cuda == '11.1', 'Unexpected numerical environment')
    cfg = OmegaConf.load(plan['config'])
    cfg.wandb_run_id = plan['run_id']
    config = cfg.train.params.config
    require(cfg.get('full_experiment_name') in (None, '', config.name), 'Native run directory name differs')
    require((config.num_actors, config.horizon_length, config.seq_length, config.minibatch_size,
             config.mini_epochs) == (20480, 16, 16, 81920, 2), 'Probe layout differs')
    require(cfg.seed == cfg.train.params.seed == 34000, 'Fixed diagnostic seed differs')
    require(config.max_frames == START_FRAME + QUANTUM and config.max_epochs == -1, 'Probe must end at one complete epoch')
    require(config.epo and config.use_others_experience == 'lf' and config.off_policy_support_mode == 'actor_clip', 'Native EPO control required')
    require(config.get('release_completed_rollout_history') is True, 'Audited history release required')
    require(cfg.wandb_mode == 'online' and cfg.wandb_resume == 'never' and cfg.wandb_activate, 'Online fresh W&B identity required')
    require(not cfg.test and not cfg.pbt.enabled and not cfg.multi_gpu, 'Single native training process required')
    require(identity(cfg.checkpoint)['sha256'] == E51_SHA, 'Checkpoint differs')
    require(not (ROOT / 'runs' / config.name).exists(), 'Existing run; no automatic resume')

    snapshot = source_snapshot(output)
    (output / 'plan.json').write_bytes(plan_path.read_bytes())
    (output / 'config.resolved.yaml').write_text(OmegaConf.to_yaml(cfg))
    for name, path in plan.get('extra_plan_files', {}).items():
        require(Path(name).name == name and not (output / name).exists(), 'Invalid or duplicate plan attachment')
        shutil.copy2(path, output / name)
    probe_module = module('training_chain_passive_probe', HERE / 'chain_probe.py')
    nvml_module = module('training_chain_nvml_reader', plan['nvml_source'])
    finite_module = module('training_chain_finite_sealer', plan['finite_source'])
    for label, path in [('nvml_source.py', plan['nvml_source']), ('finite_source.py', plan['finite_source'])]:
        shutil.copy2(path, output / label)
    nvml = nvml_module.NvmlMemory()
    state = {'agent': None, 'observer': None, 'probe': None, 'resource_samples': 0,
             'suppressed_saves': 0, 'restore_returned': False, 'finish_entered': False}
    resources = (output / 'actor_return_resources.jsonl').open('x')
    patches = []

    def patch(owner, name, replacement):
        old = getattr(owner, name)
        patches.append((owner, name, old))
        setattr(owner, name, replacement)
        return old

    def init(agent, *args, **kwargs):
        old_init(agent, *args, **kwargs)
        require(state['agent'] is None, 'Multiple agents')
        state['agent'] = agent
        state['probe'] = probe_module.install_chain_probe(agent, output / 'probes', expected_epochs=1)
        counted_actor = agent.train_actor_critic

        def observed(batch):
            result = counted_actor(batch)
            row = dict(utc=now(), ordinal=state['resource_samples'] + 1, epoch=int(agent.epoch_num),
                       frame=int(agent.frame), disk_free_bytes=shutil.disk_usage(ROOT).free, **nvml())
            resources.write(json.dumps(row, separators=(',', ':')) + '\n')
            resources.flush()
            state['resource_samples'] += 1
            require(row['driver_free_bytes'] >= 2 ** 30 and row['disk_free_bytes'] >= 2 ** 30, 'Post-actor resource floor failed')
            return result

        agent.train_actor_critic = observed

    def restore(agent, filename, *args, **kwargs):
        require(identity(filename)['sha256'] == E51_SHA, 'Unexpected restore')
        result = old_restore(agent, filename, *args, **kwargs)
        require(agent.frame == START_FRAME and agent.epoch_num == 15259, 'Actual checkpoint clock differs')
        state['restore_returned'] = True
        write(output / 'restored.json', {'frame': int(agent.frame), 'epoch': int(agent.epoch_num),
              'learning_rate': float(agent.optimizer.param_groups[0]['lr']), 'amp_scale': float(agent.scaler.get_scale()),
              'checkpoint_sha256': E51_SHA, 'native_restore': True,
              'scope': 'Native checkpoint restoration with newly constructed simulator; not exact replay of historical simulator/RNG state'})
        return result

    def augment(agent, batch, extras, *args, **kwargs):
        result = old_augment(agent, batch, extras, *args, **kwargs)
        release_completed_rollout_history(extras, result)
        return result

    def save(agent, *args, **kwargs):
        state['suppressed_saves'] += 1
        # A mechanism probe has no learning endpoint for later selection.
        return None

    def before_init(observer, *args, **kwargs):
        state['observer'] = observer
        result = old_before_init(observer, *args, **kwargs)
        require(observer.run.id == plan['run_id'] and not observer.run.resumed and not observer.run.offline, 'Wrong W&B identity')
        files = {p.name: p for p in output.iterdir() if p.is_file() and p.name != 'actor_return_resources.jsonl'}
        proof = publish(observer.run, files, 'training-chain-plan')
        write(output / 'plan-artifact-wait.json', proof)
        observer.run.summary.update({'study/purpose': 'passive_training_chain_probe', 'study/expected_physical_frames': QUANTUM,
                                     'study/selection_from_score': False, 'study/checkpoint_sha256': E51_SHA})
        return result

    def finish(observer, exit_code=0):
        if observer.finished:
            return
        require(not state['finish_entered'], 'Repeated finish boundary')
        state['finish_entered'] = True
        errors = []
        effective = exit_code
        summary = {'status': 'failed' if exit_code else 'completed', 'utc': now(),
                   'native_exit_code': exit_code, 'resource_samples': state['resource_samples'],
                   'suppressed_saves': state['suppressed_saves'], 'checkpoint_saved': False}
        writer = observer.writer

        def remember(exc, where):
            nonlocal effective
            effective = effective or 1
            errors.append({'where': where, 'error': repr(exc)})
            try:
                with (output / 'seal-errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(errors[-1]) + '\n')
            except BaseException as secondary:
                try:
                    print('Secondary seal journal failure: ' + repr(secondary), file=sys.stderr)
                except BaseException:
                    pass

        try:
            if state['probe'] is not None:
                summary['probe'] = state['probe'].finalize(require_complete=exit_code == 0)
            if exit_code == 0:
                agent = state['agent']
                require(state['restore_returned'] and agent.frame == START_FRAME + QUANTUM and agent.epoch_num == 15260, 'Terminal quota differs')
                require(state['resource_samples'] == 8, 'Actor return quota differs')
                finite_module.terminal_finite(agent, output / 'terminal-finite.json')
                summary.update(frame=int(agent.frame), epoch=int(agent.epoch_num), physical_frames=QUANTUM)
            for name, sha in snapshot['files'].items():
                require(identity(ROOT / name)['sha256'] == sha, 'Source changed while executing: ' + name)
        except BaseException as exc:
            remember(exc, 'probe_validation_or_finite_or_source')
        writer_closed = writer is None
        if writer is not None:
            try:
                writer.flush()
                writer.close()
                writer_closed = True
            except BaseException as exc:
                remember(exc, 'native_writer_close')
        observer.writer = None
        try:
            resources.close()
        except BaseException as exc:
            remember(exc, 'resources_close')
        summary.update(status='completed' if effective == 0 else 'failed', effective_exit_code=effective,
                       errors=list(errors), native_writer_closed=writer_closed)
        try:
            write(output / 'summary.json', summary)
            run_dir = ROOT / 'runs' / config.name
            # Do not snapshot a writer with an unconfirmed close. Its raw files
            # remain local for the parent to preserve after the real child exit.
            files = {str(p.relative_to(output)): p for p in output.rglob('*') if p.is_file()
                     and p.name != 'seal-errors.jsonl'}
            files.update({'native/' + str(p.relative_to(run_dir)): p for p in run_dir.rglob('*')
                          if p.is_file() and not p.is_symlink() and 'wandb' not in p.relative_to(run_dir).parts
                          and p.name != 'wandb_run.json' and (writer_closed or 'summaries' not in p.parts)})
            proof = publish(observer.run, files, 'training-chain-probe')
            write(output / 'result-artifact-wait.json', proof)
            observer.run.summary.update({'study/status': summary['status'], 'study/physical_frames': summary.get('physical_frames'),
                                         'study/probe_result_file_count': len(proof['files'])})
        except BaseException as exc:
            remember(exc, 'raw_artifact_publication')
        finally:
            try:
                finite_module.finish_original_observer(observer, old_finish, plan['run_id'],
                                                      effective, remember, output)
            finally:
                observer.writer = writer
        if errors:
            raise RuntimeError('Probe finalization failed; first error: ' + str(errors[0]))

    old_init = patch(A2CAgent, '__init__', init)
    old_restore = patch(A2CAgent, 'restore', restore)
    old_augment = patch(A2CBase, 'augment_batch_for_mixed_expl', augment)
    patch(A2CAgent, 'save', save)
    old_before_init = patch(WandbAlgoObserver, 'before_init', before_init)
    old_finish = patch(WandbAlgoObserver, 'finish', finish)
    native_error = None
    try:
        native.launch_rlg_hydra(cfg)
    except BaseException as exc:
        native_error = exc
        secondary_report(output / 'execution-error.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        observer = state['observer']
        if observer is not None and not observer.finished and not state['finish_entered']:
            try:
                finish(observer, exit_code=1)
            except BaseException as secondary:
                secondary_report(output / 'secondary-finish-error.json', {'error': repr(secondary)})
        raise
    finally:
        cleanup_errors = []
        try:
            resources.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            if state['probe'] is not None:
                state['probe'].uninstall()
        except BaseException as exc:
            cleanup_errors.append(exc)
        for owner, name, original in reversed(patches):
            try:
                setattr(owner, name, original)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            secondary_report(output / 'cleanup-errors.json', {'errors': [repr(exc) for exc in cleanup_errors],
                                                             'native_first_error': repr(native_error)})
            if native_error is None:
                raise cleanup_errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    require(identity(args.plan)['sha256'] == args.plan_sha256, 'Plan SHA differs')
    plan = json.loads(args.plan.read_text())
    require(plan['purpose'] == 'passive_training_chain_probe', 'Wrong plan purpose')
    for path, sha in plan['files_sha256'].items():
        require(identity(path)['sha256'] == sha, 'Reviewed input differs: ' + path)
    require(str(Path(__file__).resolve()) in plan['files_sha256'], 'Runner source not bound')
    required = {str(Path(__file__).resolve()), str(HERE / 'chain_probe.py'),
                plan['config'], plan['nvml_source'], plan['finite_source']}
    required.update(plan.get('extra_plan_files', {}).values())
    require(required <= set(plan['files_sha256']), 'Config or executing helper source is not bound')
    require(Path.cwd() == ROOT and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1', 'Wrong working directory/allocator mode')
    output = Path(plan['output'])
    require(output.parent == AUDIT and not output.exists(), 'Fresh output under study audit required')
    require(shutil.disk_usage(ROOT).free >= 8 * 2 ** 30, 'Insufficient disk reserve')
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) == 3}
    require(mem['MemAvailable'] >= 16 * 2 ** 30, 'Insufficient host RAM reserve')
    with open('/tmp/epo-gpu-0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir()
        execute(plan, args.plan, output)


if __name__ == '__main__':
    main()
