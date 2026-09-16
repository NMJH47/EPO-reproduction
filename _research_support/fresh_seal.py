"""Small native fresh-training sealer. Runtime only; no CLI, models or API at import."""
from datetime import datetime, timezone
import hashlib
import base64
import json
import math
from pathlib import Path
import shutil

ROOT = Path('/home/ubuntu22/EPO')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    # New payload files only; never rewrite a file after artifact submission.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def file_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/nonregular/symlink file: ' + str(path))
    before = path.stat()
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed during hashing: ' + str(path))
    return {'local_path': str(path.resolve()), 'sha256': sha.hexdigest(),
            'md5_base64': base64.b64encode(md5.digest()).decode(),
            'size': after.st_size, 'mtime_ns': after.st_mtime_ns}


def append_error(output, where, error, primary=None):
    """Outside immutable payload and independent of already closed JSONL handles."""
    path = Path(output).with_name(Path(output).name + '-lifecycle-errors.jsonl')
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'where': where,
           'error': repr(error), 'primary_error': primary}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()


def finish_original_observer(observer, original_finish, requested_id, exit_code, remember, output):
    """Native finish first; compensate only if its owned run.finish never entered."""
    run = getattr(observer, 'run', None)
    record = {'native_observer_finish_returned': False, 'run_finish_entered': 0,
              'run_finish_returned': 0, 'fallback_attempted': False, 'errors': []}
    missing = object()
    old_attribute = missing
    original_run_finish = None
    installed = False

    def error(exc, where):
        record['errors'].append({'where': where, 'error': repr(exc)})
        remember(exc, where)

    def tracked_finish(*args, **kwargs):
        record['run_finish_entered'] += 1
        result = original_run_finish(*args, **kwargs)
        record['run_finish_returned'] += 1
        return result

    try:
        require(not observer.finished, 'Native observer was already finished before the one finalization boundary')
        if run is not None:
            require(str(run.id) == str(requested_id) and not run.resumed and run.settings.mode == 'online',
                    'Refusing finalization of an unrelated/resumed/non-online run')
            original_run_finish = run.finish
            old_attribute = run.__dict__.get('finish', missing)
            run.finish = tracked_finish
            installed = True
        try:
            original_finish(observer, exit_code=exit_code)
            record['native_observer_finish_returned'] = True
        except BaseException as exc:
            error(exc, 'native_observer_finish')
            if installed and record['run_finish_entered'] == 0:
                record['fallback_attempted'] = True
                try:
                    tracked_finish(exit_code=1)
                except BaseException as secondary:
                    error(secondary, 'owned_finish1_compensation')
    except BaseException as exc:
        error(exc, 'native_finish_setup')
    finally:
        if installed:
            try:
                if old_attribute is missing:
                    del run.finish
                else:
                    run.finish = old_attribute
            except BaseException as exc:
                error(exc, 'run_finish_hook_restore')
        try:
            write_json(Path(output).with_name(Path(output).name + '-finish-attempt.json'), record)
        except BaseException as exc:
            error(exc, 'finish_attempt_record')
    return record


def capture_sources(output, execution_sources):
    from scripts.evaluate_frozen_checkpoint import source_snapshot
    output = Path(output)
    source_dir = output / 'sources'
    repository = source_dir / 'repository'
    repository.mkdir(parents=True, exist_ok=False)
    manifest = source_snapshot(repository)
    external = {}
    for original in execution_sources:
        original = Path(original)
        require(original.name not in external, 'Duplicate external source name')
        identity = file_identity(original)
        copied = source_dir / original.name
        shutil.copyfile(original, copied)
        require(file_identity(copied)['sha256'] == identity['sha256'], 'Source copy differs')
        external[original.name] = identity
    write_json(source_dir / 'external_manifest.json', external)
    return {'repository_source_sha256': manifest['source_sha256'], 'repository_file_count': len(manifest['files'])}


def verify_sources(output):
    source_dir = Path(output) / 'sources'
    repository = json.loads((source_dir / 'repository/source_manifest.json').read_text())
    for name, sha in repository['files'].items():
        require(file_identity(ROOT / name)['sha256'] == sha, 'Repository source changed: ' + name)
    external = json.loads((source_dir / 'external_manifest.json').read_text())
    for name, original in external.items():
        require(file_identity(original['local_path']) == original, 'Executing source changed: ' + name)
        require(file_identity(source_dir / name)['sha256'] == original['sha256'], 'Archived source changed: ' + name)
    return {'repository_source_sha256': repository['source_sha256'], 'repository_file_count': len(repository['files'])}


def capture_executed_config(output, cfg, agent):
    """Native train.py dumps these before runner.run constructs the fresh agent."""
    from omegaconf import OmegaConf
    name = str(cfg.train.params.config.name)
    require(agent.experiment_name == name, 'Native experiment/config directory mismatch')
    live = ROOT / 'runs' / name / 'config.yaml'
    latest = live.with_name('config.latest.yaml')
    identity = file_identity(live)
    require(live.read_bytes() == latest.read_bytes(), 'Native config and latest differ')
    require(live.read_text() == OmegaConf.to_yaml(cfg), 'Native dumped configuration differs from executing cfg')
    copy = Path(output) / 'executed_training_config.yaml'
    require(not copy.exists(), 'Executed config capture already exists')
    shutil.copyfile(live, copy)
    latest_identity, copied_identity = file_identity(latest), file_identity(copy)
    require(identity['sha256'] == latest_identity['sha256'] == copied_identity['sha256'], 'Config copies changed during capture')
    subset_keys = ('name', 'epo', 'expl_type', 'use_others_experience', 'num_actors', 'horizon_length',
                   'seq_length', 'minibatch_size', 'mini_epochs', 'learning_rate', 'max_frames', 'max_epochs',
                   'mixed_precision', 'truncate_grads', 'grad_norm', 'schedule_type', 'lr_schedule',
                   'kl_threshold', 'off_policy_actor_coef', 'off_policy_critic_coef', 'off_policy_support_mode')
    subset = {key: agent.config[key] for key in subset_keys if key in agent.config}
    write_json(Path(output) / 'executed_training_config_manifest.json',
               {'native_config': identity, 'native_latest': latest_identity,
                'archived_config': copied_identity, 'agent_config_subset': subset,
                'boundary': 'After native fresh A2CAgent construction; no restore'})


def verify_executed_config(output):
    record = json.loads((Path(output) / 'executed_training_config_manifest.json').read_text())
    for key in ('native_config', 'native_latest', 'archived_config'):
        require(file_identity(record[key]['local_path']) == record[key], 'Executed config changed: ' + key)
    return record['native_config']['sha256']


def terminal_finite(agent, path):
    """One live CUDA scan; state collection follows actual ownership, never GA count.

    All registered model parameters/buffers include RMS. Gradients are excluded:
    an ordinary AMP skipped update is not a nonfinite model/Adam-state failure.
    """
    import torch
    result = {'status': 'running', 'model_parameters': [], 'model_buffers': [],
              'owned_parameters': [], 'optimizer_state_tensor_paths': [], 'scalar_metadata_paths': [],
              'cuda_tensor_count': 0, 'cuda_element_count': 0, 'first_failure': None,
              'scope': 'Live model parameters/buffers including RMS, actual optimizer state and group metadata; no saved-checkpoint reload, gradients or extra forward'}
    current = None

    def visit(value, name, step_metadata=False):
        nonlocal current
        current = name
        if torch.is_tensor(value):
            if value.is_cuda:
                require(bool(torch.isfinite(value).all().item()), 'Nonfinite CUDA state: ' + name)
                result['cuda_tensor_count'] += 1
                result['cuda_element_count'] += value.numel()
                if name.startswith('optimizer.'):
                    result['optimizer_state_tensor_paths'].append(name)
            else:
                require(step_metadata and value.numel() == 1, 'Computational state is not CUDA: ' + name)
                require(math.isfinite(float(value.item())), 'Nonfinite CPU step scalar: ' + name)
                result['scalar_metadata_paths'].append(name)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, name + '.' + str(key), step_metadata=(str(key) == 'step'))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, name + '.' + str(index), step_metadata)
        elif isinstance(value, (int, float, bool)):
            require(math.isfinite(float(value)), 'Nonfinite numeric metadata: ' + name)
            result['scalar_metadata_paths'].append(name)
        else:
            require(value is None, 'Unexpected optimizer state type: ' + name)

    try:
        require(agent is not None and torch.cuda.is_available(), 'Live CUDA agent required')
        parameters = dict(agent.model.named_parameters())
        parameter_names = {id(value): name for name, value in parameters.items()}
        owned = []
        with torch.no_grad():
            for kind, items in (('model_parameters', parameters.items()), ('model_buffers', agent.model.named_buffers())):
                for name, value in items:
                    visit(value, kind + '.' + name)
                    result[kind].append({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)})
            for group_index, group in enumerate(agent.optimizer.param_groups):
                for parameter in group['params']:
                    require(id(parameter) in parameter_names, 'Optimizer owns an unregistered model parameter')
                    owned.append(id(parameter))
                    result['owned_parameters'].append(parameter_names[id(parameter)])
                for key, value in group.items():
                    if key != 'params':
                        visit(value, 'optimizer.group.%d.%s' % (group_index, key))
            require(len(set(owned)) == len(owned), 'Duplicate optimizer parameter objects')
            for parameter, state in agent.optimizer.state.items():
                require(id(parameter) in owned, 'Optimizer state contains a non-owned parameter')
                visit(state, 'optimizer.state.' + parameter_names[id(parameter)])
        result['optimizer_state_entries'] = len(agent.optimizer.state)
        result['status'] = 'completed'
    except BaseException as exc:
        result['status'] = 'failed'
        result['first_failure'] = {'path': current, 'error': repr(exc)}
        try:
            write_json(path, result)
        except BaseException as secondary:
            print('Finite record write also failed: ' + repr(secondary))
        raise
    write_json(path, result)
    return result


def seal_fresh(summary, output, observer, agent, cfg):
    """Prepare one immutable artifact and wait; never calls run.finish or a new init.

    The entry calls the original observer.finish exactly once afterwards. Parent
    executor must separately verify the finished run, exact remote inventory and
    actual child exit. Failed training never triggers a new CUDA finite scan.
    """
    output = Path(output)
    first = None
    errors = []

    def attempt(where, function):
        nonlocal first
        try:
            return function()
        except BaseException as exc:
            if first is None:
                first = (exc, exc.__traceback__)
            errors.append({'where': where, 'error': repr(exc)})
            try:
                append_error(output, where, exc, summary.get('error'))
            except BaseException as secondary:
                print('Seal error sidecar also failed: ' + repr(secondary))
            return None

    writer = getattr(observer, 'writer', None) or getattr(agent, 'writer', None)
    closed = []
    if writer is not None:
        attempt('writer_flush', writer.flush)
        # Always try close even when flush failed. Only remove a successfully
        # closed writer from observer; original finish still owns run termination.
        def close_writer():
            writer.close()
            closed.append(True)
        attempt('writer_close', close_writer)
        if closed and observer is not None:
            observer.writer = None
    finite = None
    if summary.get('status') != 'failed' and first is None:
        finite = attempt('terminal_finite', lambda: terminal_finite(agent, output / 'terminal_finite_check.json'))
    else:
        attempt('finite_skipped_record', lambda: write_json(output / 'terminal_finite_check.json',
                {'status': 'not_run_after_failure', 'reason': 'Prior training/cleanup failure; no new CUDA scan'}))
    source = attempt('source_stability', lambda: verify_sources(output))
    config_sha = attempt('config_stability', lambda: verify_executed_config(output))
    finite_path = output / 'terminal_finite_check.json'
    finite_record = attempt('finite_record_read', lambda: json.loads(finite_path.read_text())) if finite_path.is_file() else None
    if summary.get('status') != 'failed' and first is None:
        attempt('finite_record_complete', lambda: require(finite_record is not None and finite_record.get('status') == 'completed',
                'Successful terminal finite scan did not leave a completed record'))
    record = dict(summary, sealing_errors=errors, source=source, executed_config_sha256=config_sha,
                  finite_status=None if finite_record is None else finite_record['status'],
                  writer_close_returned=bool(closed),
                  tensorboard_scope=('Complete local TB files after writer.close returned' if closed else
                      'Best-effort existing TB bytes at pre-upload snapshot; writer was not confirmed closed. Native finish may append later; this is not final complete TB.'),
                  checkpoint_validation='Live state after the original terminal save, with no intervening optimizer/environment call recorded; saved checkpoint is not reloaded. Future strict evaluation loading is a separate check.',
                  remote_completion='Not established by this payload; original finish and fresh executor readback follow artifact wait')
    record['status'] = 'failed' if summary.get('status') == 'failed' or first else 'native_completed_finite'
    attempt('terminal_record', lambda: write_json(output / 'training_summary.json', record))

    def upload():
        import wandb
        run = getattr(observer, 'run', None)
        require(run is not None and not observer.finished, 'No active native W&B run available for archival')
        require(str(run.id) == str(cfg.wandb_run_id) and not run.resumed and run.settings.mode == 'online',
                'Refusing unrelated/resumed/non-online W&B run')
        # Snapshot only the small pre-upload error log; later errors keep appending
        # outside the immutable artifact and never change this historical copy.
        sidecar = output.with_name(output.name + '-lifecycle-errors.jsonl')
        if sidecar.is_file():
            destination = output / 'lifecycle_errors_before_archive.jsonl'
            require(not destination.exists(), 'Pre-upload error snapshot already exists')
            shutil.copyfile(sidecar, destination)
        files = {p.relative_to(output).as_posix(): p for p in output.rglob('*') if p.is_file()}
        # Closed TB originals are uploaded directly: no second local binary copy.
        experiment = ROOT / 'runs' / str(cfg.train.params.config.name)
        tensorboard = experiment / 'summaries'
        if tensorboard.is_dir():
            files.update({'tensorboard/' + p.relative_to(tensorboard).as_posix(): p
                          for p in tensorboard.rglob('*') if p.is_file()})
        for name in ('cmd.txt', 'commands.log'):
            if (experiment / name).is_file():
                files['native/' + name] = experiment / name
        checkpoint = summary.get('terminal_checkpoint') or summary.get('terminal_checkpoint_attempt')
        if checkpoint and Path(checkpoint).is_file():
            files['checkpoint.pth' if summary.get('terminal_checkpoint') else 'checkpoint_attempt.pth'] = Path(checkpoint)
        identities = {name: file_identity(path) for name, path in sorted(files.items())}
        write_json(output / 'payload_manifest.json', identities)
        files['payload_manifest.json'] = output / 'payload_manifest.json'
        identities['payload_manifest.json'] = file_identity(files['payload_manifest.json'])
        artifact = wandb.Artifact('matched-fresh-training-' + str(run.id), type='matched-fresh-training',
                                  metadata={'status': 'failed' if summary.get('status') == 'failed' or first else record['status'],
                                            'primary_training_error': summary.get('error'), 'sealing_errors': list(errors),
                                            'method': summary['expected']['method'],
                                            'writer_close_returned': bool(closed),
                                            'checkpoint_policy': 'immutable_skip_cache',
                                            'physics_control_frames': summary['observed'].get('physics_control_frames', 0)})
        for name, path in sorted(files.items()):
            artifact.add_file(str(path), name=name, policy='immutable', skip_cache=True)
        run.summary.update({'matched_fresh/native_status': record['status'],
                            'matched_fresh/observed': summary['observed'],
                            'matched_fresh/finite_status': record['finite_status']})
        committed = run.log_artifact(artifact).wait()
        require(str(committed.state).upper() == 'COMMITTED', 'Artifact wait did not commit')
        entries = committed.manifest.entries
        require(set(entries) == set(identities), 'Committed manifest inventory differs')
        for name, identity in identities.items():
            require(file_identity(files[name]) == identity, 'Immutable original changed during upload: ' + name)
            require(entries[name].digest == identity['md5_base64'] and entries[name].size == identity['size'], 'Artifact manifest bytes differ: ' + name)
        write_json(output.with_name(output.name + '-artifact-wait.json'),
                   {'status': 'artifact_committed_awaiting_native_finish', 'run_id': run.id, 'run_url': run.url,
                    'artifact_name': committed.qualified_name, 'artifact_type': committed.type,
                    'files': identities, 'checkpoint_copied_for_staging': False,
                    'scope': 'SDK waited artifact manifest and local stability; not a fresh post-finish API readback'})
    attempt('artifact_upload_wait', upload)
    if first is not None:
        raise first[0].with_traceback(first[1])
    return record
