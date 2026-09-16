"""Pilot continuation with one process-local gradient-clipping intervention.

Hydra arguments pass through unchanged. gradient_clip_scope=model invokes the
original update; optimizer clips only optimizer-owned parameters after AMP
unscale. Both arms receive identical passive telemetry and final-only saving.
"""

from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isaacgym  # noqa: F401; binding must precede torch
import torch

from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
from rl_games.common.a2c_common import A2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from scripts.evaluate_frozen_checkpoint import sha256_file, source_snapshot


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def validate_wandb_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('W&B ID must remain a nonempty safe string after Hydra parsing')
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(safe_json(value), indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def cuda_gradient_norm(parameters):
    """One CUDA reduction and one host scalar read; no RNG or gradient writes."""
    norms = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    return torch.stack(norms).norm(2).item() if norms else 0.0


def clipping_update(agent, scope, owned_parameters, original_update):
    """Effective update, also callable by isolated CUDA mechanism checks."""
    if scope == 'model':
        return original_update(agent)
    if scope != 'optimizer':
        raise ValueError('Unknown clipping scope: ' + str(scope))
    agent.scaler.unscale_(agent.optimizer)
    torch.nn.utils.clip_grad_norm_(owned_parameters, agent.grad_norm)
    agent.scaler.step(agent.optimizer)
    agent.scaler.update()


def release_unused_cuda_cache():
    """Return inactive Torch allocator segments to the driver for PhysX."""
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    return {'torch_allocated_before': before_allocated,
            'torch_reserved_before': before_reserved,
            'torch_allocated_after': torch.cuda.memory_allocated(),
            'torch_reserved_after': torch.cuda.memory_reserved(),
            'cache_release_seconds': elapsed}


def release_completed_rollout_history(extras, augmented_batch):
    """Drop completed LF history storage, retaining the training batch tensors."""
    history = extras.get('rnn_states')
    if not isinstance(history, (list, tuple)) or not history:
        raise RuntimeError('Expected recurrent rollout history at the post-augmentation boundary')
    pointers = {state.storage().data_ptr() for state in history}

    def check_no_alias(value):
        if isinstance(value, torch.Tensor):
            if value.storage().data_ptr() in pointers:
                raise RuntimeError('Training batch still aliases the completed rollout history')
        elif isinstance(value, dict):
            for item in value.values():
                check_no_alias(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_no_alias(item)

    check_no_alias(augmented_batch)
    released_bytes = sum(state.numel() * state.element_size() for state in history)
    before = torch.cuda.memory_allocated()
    extras['rnn_states'] = None
    del history
    return {'history_tensor_bytes': released_bytes, 'allocated_before': before,
            'allocated_after': torch.cuda.memory_allocated()}


def install(output, start_frame, target_frame, checkpoint_hash):
    source = source_snapshot(output)
    record = {
        'experiment_kind': os.environ.get('EPO_CLIP_EXPERIMENT_KIND', 'single_checkpoint_single_seed_continuation_pilot'),
        'requested_additional_frames': target_frame - start_frame,
        'cuda_allocator_cache_disabled': 'PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ,
        'allocator_statistics_scope': 'Torch caching-allocator counters exclude PhysX and may be zero when the cache is disabled; consult driver/system telemetry',
        'interpretation': 'One-factor mechanism pilot, not evidence of publication-level learning benefit.',
        'start_frame_required': start_frame, 'target_frame_required': target_frame,
        'input_checkpoint_sha256': checkpoint_hash, 'source_sha256': source['source_sha256'],
        'git_commit': source['git_commit'], 'argv': list(sys.argv),
        'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0),
        'utc_started_at': datetime.now(timezone.utc).isoformat(),
        'telemetry': 'First update of each epoch: scaled owned/unowned norms and actual post-clip owned norm. All updates: AMP scale-drop skip proxy.',
        'amp_skip_proxy_definition': 'A decrease of GradScaler scale after update; labels indicate a proxy, not an optimizer-step hook.',
        'checkpoint_policy': 'Both arms suppress intermediate/best writes and retain only budget-terminal checkpoint.',
        'status': 'installed', 'updates': 0, 'amp_scale_drop_skip_proxy_count': 0,
        'epoch_counters': {}, 'first_update_metrics': [], 'suppressed_checkpoint_writes': 0,
        'rollout_memory_metrics': [],
        'completed_history_release_metrics': [],
    }
    state_path = output / 'training_control.json'
    save_json(state_path, record)
    lifecycle = {'agent': None, 'observer': None, 'finalized': False}
    original_update = A2CBase.trancate_gradients_and_step
    original_restore = A2CAgent.restore
    original_save = A2CAgent.save
    original_before_init = WandbAlgoObserver.before_init
    original_finish = WandbAlgoObserver.finish
    original_play_steps = A2CBase.play_steps
    original_augment = A2CBase.augment_batch_for_mixed_expl

    def augment(agent, batch, extras, repeat_idxs=None):
        result = original_augment(agent, batch, extras, repeat_idxs)
        enabled = bool(agent.config.get('release_completed_rollout_history', False))
        record['release_completed_rollout_history'] = enabled
        if enabled:
            if not agent.is_rnn or agent.use_others_experience != 'lf':
                raise RuntimeError('Completed history release is audited only for recurrent LF')
            row = {'epoch': int(agent.epoch_num), 'frame': int(agent.frame),
                   **release_completed_rollout_history(extras, result)}
            record['completed_history_release_metrics'].append(row)
            save_json(state_path, record)
            observer = lifecycle['observer']
            if observer is None or observer.run is None:
                raise RuntimeError('History release telemetry requires online W&B')
            observer.run.log({**{'completed_history/' + key: value for key, value in row.items()},
                              'global_step': int(agent.frame)})
        return result

    def play_steps(agent):
        enabled = bool(agent.config.get('release_cache_before_rollout', False))
        record['release_cache_before_rollout'] = enabled
        if enabled:
            row = {'epoch': int(agent.epoch_num), 'frame': int(agent.frame),
                   **release_unused_cuda_cache()}
            record['rollout_memory_metrics'].append(row)
            save_json(state_path, record)
            observer = lifecycle['observer']
            if observer is None or observer.run is None:
                raise RuntimeError('Allocator telemetry requires an online W&B run')
            observer.run.log({**{'allocator/' + key: value for key, value in row.items()},
                              'global_step': int(agent.frame)})
        return original_play_steps(agent)

    def restore(agent, filename, *args, **kwargs):
        if sha256_file(filename) != checkpoint_hash:
            raise RuntimeError('Continuation checkpoint differs from the pair manifest')
        expected_cache_disabled = agent.config.get('cuda_allocator_cache_disabled')
        if expected_cache_disabled is not None and (
                not isinstance(expected_cache_disabled, bool)
                or expected_cache_disabled != record['cuda_allocator_cache_disabled']):
            raise RuntimeError('Declared CUDA allocator configuration differs from process environment')
        result = original_restore(agent, filename, *args, **kwargs)
        lifecycle['agent'] = agent
        if int(agent.frame) != start_frame:
            raise RuntimeError('Restored frame {} differs from required {}'.format(agent.frame, start_frame))
        if int(agent.max_frames) != target_frame:
            raise RuntimeError('Agent frame target differs from the paired protocol')
        record['restored_frame'] = int(agent.frame)
        record['restored_epoch'] = int(agent.epoch_num)
        record['restored_learning_rate'] = float(agent.optimizer.param_groups[0]['lr'])
        record['restored_amp_scale'] = float(agent.scaler.get_scale())
        record['gradient_clip_scope'] = agent.config['gradient_clip_scope']
        save_json(state_path, record)
        return result

    def checkpoint_save(agent, filename, override_state=None):
        # save_frequency does not disable hard-coded rolling/best checkpoints.
        # Save suppression is identical in both arms and does not change tensors.
        if '_frame_' not in Path(filename).name or int(agent.frame) < target_frame:
            record['suppressed_checkpoint_writes'] += 1
            return
        result = original_save(agent, filename, override_state)
        checkpoint = Path(str(filename) + '.pth').resolve()
        if not checkpoint.is_file():
            raise RuntimeError('Final checkpoint was not produced')
        record['final_checkpoint_path'] = str(checkpoint)
        record['final_checkpoint_sha256'] = sha256_file(checkpoint)
        record['final_frame'] = int(agent.frame)
        record['final_epoch'] = int(agent.epoch_num)
        save_json(state_path, record)
        return result

    def log_epoch_counter(epoch):
        observer = lifecycle['observer']
        if observer is not None and observer.run is not None:
            counter = record['epoch_counters'][str(epoch)]
            observer.run.log({
                'clipping_ablation/epoch': epoch,
                'clipping_ablation/updates_in_epoch': counter['updates'],
                'clipping_ablation/amp_scale_drop_skip_proxy_count_in_epoch': counter['amp_scale_drop_skip_proxy_count'],
                'global_step': counter['frame_at_updates'],
            })

    def update(agent):
        lifecycle['agent'] = agent
        scope = agent.config.get('gradient_clip_scope')
        if scope not in {'model', 'optimizer'}:
            raise ValueError('gradient_clip_scope must be explicitly model or optimizer')
        if agent.multi_gpu or not agent.truncate_grads or not agent.scaler.is_enabled():
            raise RuntimeError('Both arms require one GPU, gradient clipping, and enabled AMP')
        owned = [parameter for group in agent.optimizer.param_groups for parameter in group['params']]
        owned_ids = {id(parameter) for parameter in owned}
        if len(owned_ids) != len(owned):
            raise RuntimeError('Optimizer contains duplicate parameter objects')
        unowned = [parameter for parameter in agent.model.parameters() if id(parameter) not in owned_ids]
        if not unowned:
            raise RuntimeError('E51 pilot requires the checkpoint gene parameter to remain outside Adam')
        epoch = int(agent.epoch_num)
        first = str(epoch) not in record['epoch_counters']
        if first:
            previous = record.get('last_observed_epoch')
            if previous is not None:
                log_epoch_counter(previous)
            record['last_observed_epoch'] = epoch
            record['epoch_counters'][str(epoch)] = {
                'updates': 0, 'amp_scale_drop_skip_proxy_count': 0, 'frame_at_updates': int(agent.frame)}
        scale_before = float(agent.scaler.get_scale())
        if first:
            owned_scaled = cuda_gradient_norm(owned)
            unowned_scaled = cuda_gradient_norm(unowned)
        # This is the sole algorithmic difference between the paired arms.
        result = clipping_update(agent, scope, owned, original_update)
        scale_after = float(agent.scaler.get_scale())
        skipped_proxy = int(scale_after < scale_before)
        record['updates'] += 1
        record['amp_scale_drop_skip_proxy_count'] += skipped_proxy
        counter = record['epoch_counters'][str(epoch)]
        counter['updates'] += 1
        counter['amp_scale_drop_skip_proxy_count'] += skipped_proxy
        if first:
            row = {'epoch': epoch, 'frame_at_update': int(agent.frame),
                   'amp_scale_before': scale_before, 'amp_scale_after': scale_after,
                   'owned_scaled_gradient_norm': owned_scaled,
                   'unowned_scaled_gradient_norm': unowned_scaled,
                   'actual_postclip_owned_gradient_norm': cuda_gradient_norm(owned),
                   'clip_threshold': float(agent.grad_norm), 'owned_parameter_tensors': len(owned),
                   'unowned_parameter_tensors': len(unowned),
                   'learning_rate': float(agent.optimizer.param_groups[0]['lr'])}
            record['first_update_metrics'].append(row)
            record['status'] = 'running'
            save_json(state_path, record)
            observer = lifecycle['observer']
            if observer is None or observer.run is None:
                raise RuntimeError('Training telemetry requires an online W&B run')
            observer.run.log({**{'clipping_ablation/' + key: value for key, value in row.items()
                                 if isinstance(value, (float, int)) and math.isfinite(value)},
                              'global_step': int(agent.frame)})
        return result

    def before_init(observer, *args, **kwargs):
        lifecycle['observer'] = observer
        requested_id = validate_wandb_id(observer.cfg.get('wandb_run_id'))
        if (str(observer.cfg.get('wandb_mode', '')).lower() != 'online'
                or str(observer.cfg.get('wandb_resume', '')).lower() != 'never'
                or not observer.cfg.get('wandb_run_id')):
            raise RuntimeError('Pilot requires explicit fresh ID, wandb_resume=never, and online mode')
        result = original_before_init(observer, *args, **kwargs)
        if observer.run.resumed:
            raise RuntimeError('Pilot unexpectedly resumed an existing W&B run')
        if observer.run.id != requested_id:
            raise RuntimeError('W&B actual ID differs from the validated request')
        record['wandb'] = {'id': observer.run.id, 'url': observer.run.url}
        save_json(state_path, record)
        return result

    def finish(observer, exit_code=0):
        if observer.finished:
            return
        active_error = sys.exc_info()[1]
        error = active_error
        agent = lifecycle['agent']
        if error is None and not exit_code:
            if (agent is None or int(agent.frame) < target_frame
                    or record.get('restored_frame') != start_frame
                    or not record.get('final_checkpoint_path') or record['updates'] == 0):
                error = RuntimeError('Pilot ended without restored origin, full frame budget, updates, and final checkpoint')
        effective_exit_code = exit_code or (1 if error is not None else 0)
        record.update(status='completed' if effective_exit_code == 0 else 'failed',
                      effective_exit_code=effective_exit_code,
                      observed_frame=int(agent.frame) if agent is not None else None,
                      utc_validated_at=datetime.now(timezone.utc).isoformat())
        if error is not None:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        try:
            if record.get('last_observed_epoch') is not None:
                log_epoch_counter(record['last_observed_epoch'])
            if observer.writer is not None:
                observer.writer.flush()
            save_json(state_path, record)
            observer.run.summary.update({
                'clipping_ablation/status': record['status'],
                'clipping_ablation/scope': record.get('gradient_clip_scope'),
                'clipping_ablation/target_frame': target_frame,
                'clipping_ablation/observed_frame': record['observed_frame'],
                'clipping_ablation/updates': record['updates'],
                'clipping_ablation/amp_scale_drop_skip_proxy_count': record['amp_scale_drop_skip_proxy_count'],
                'clipping_ablation/source_sha256': source['source_sha256'],
            })
            import wandb
            artifact = wandb.Artifact('clipping-ablation-' + observer.run.id, type='clipping-ablation-pilot',
                                      metadata={'status': record['status'], 'source_sha256': source['source_sha256'],
                                                'checkpoint_sha256': record.get('final_checkpoint_sha256'),
                                                'input_checkpoint_sha256': checkpoint_hash})
            for path in sorted(output.iterdir()):
                if path.is_file():
                    artifact.add_file(str(path), name=path.name)
            if record.get('final_checkpoint_path'):
                artifact.add_file(record['final_checkpoint_path'], name='checkpoint.pth')
            observer.run.log_artifact(artifact).wait()
        except BaseException as archive_error:
            error = archive_error
            effective_exit_code = effective_exit_code or 1
            record.update(status='failed', effective_exit_code=effective_exit_code,
                          finalization_error={'type': type(archive_error).__name__, 'message': str(archive_error)})
            save_json(state_path, record)
            try:
                observer.run.summary.update({'clipping_ablation/status': 'failed'})
            except Exception:
                pass
        finally:
            try:
                original_finish(observer, exit_code=effective_exit_code)
            except BaseException as finish_error:
                effective_exit_code = effective_exit_code or 1
                record.update(
                    status='failed', effective_exit_code=effective_exit_code,
                    finalization_error={
                        'type': type(finish_error).__name__,
                        'message': str(finish_error),
                    },
                )
                save_json(state_path, record)
                raise
            finally:
                lifecycle['finalized'] = True
        if error is not None and active_error is None:
            raise RuntimeError('Clipping pilot did not finalize successfully') from error

    A2CBase.trancate_gradients_and_step = update
    A2CBase.play_steps = play_steps
    A2CBase.augment_batch_for_mixed_expl = augment
    A2CAgent.restore = restore
    A2CAgent.save = checkpoint_save
    WandbAlgoObserver.before_init = before_init
    WandbAlgoObserver.finish = finish
    return lifecycle, record, state_path


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is forbidden')
    output = Path(os.environ['EPO_CLIP_ABLATION_OUTPUT_DIR']).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite clipping ablation output: ' + str(output))
    output.mkdir(parents=True)
    lifecycle, record, state_path = install(
        output, int(os.environ['EPO_CLIP_START_FRAME']), int(os.environ['EPO_CLIP_TARGET_FRAME']),
        os.environ['EPO_CLIP_INPUT_CHECKPOINT_SHA256'])
    try:
        runpy.run_module('isaacgymenvs.train', run_name='__main__')
    except BaseException as error:
        observer = lifecycle['observer']
        if observer is not None and observer.run is not None and not observer.finished:
            try:
                observer.finish(exit_code=1)
            except BaseException as finalization_error:
                print('Pilot failure finalization: ' + str(finalization_error), flush=True)
        if not lifecycle['finalized']:
            record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            save_json(state_path, record)
        raise
    if not lifecycle['finalized']:
        raise RuntimeError('Training returned without required online pilot finalization')


if __name__ == '__main__':
    main()
