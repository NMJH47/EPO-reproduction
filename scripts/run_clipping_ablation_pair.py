#!/usr/bin/env python3
"""Serial, fresh-ID 100M continuation pilot followed by matched frozen evaluation.

Run with --dry-run to inspect commands without GPU allocation or W&B writes.
Every failed phase stops the queue. Old runs and evidence are never removed.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
BASE_RUN = ROOT / 'runs/51_full_epo_E8_legacy_equivalent_identity_seed43_N20480'
CHECKPOINT = BASE_RUN / 'nn/last_51_full_epo_E8_legacy_equivalent_identity_seed43_N20480_frame_5000069120_rew__37925.93_.pth'
START_FRAME = 5000069120
TARGET_FRAME = 5100069120
ARMS = [('model', '63_clip_model_control_100m_20260913'),
        ('optimizer', '64_clip_optimizer_candidate_100m_20260913')]
ENTITY = 'leo-ucbm-university-of-california-berkeley'
PROJECT = 'EPO-reproduction'


def now():
    return datetime.now(timezone.utc).isoformat()


def fresh_wandb_id():
    # A letter prefix prevents Hydra from parsing an all-digit/e-containing hex
    # token as an integer or scientific-notation float.
    return 'r' + uuid.uuid4().hex[:15]


def allocator_environment(base, disabled):
    # Torch 1.8.1 checks presence, so assigning '0' still disables its cache.
    environment = dict(base)
    environment.pop('PYTORCH_NO_CUDA_MEMORY_CACHING', None)
    if disabled:
        environment['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    return environment


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=str(ROOT)
    ).decode().split('\0')
    excluded = {'artifacts', 'runs', 'logs', 'result', 'wandb', '.git', '.pytest_cache'}
    suffixes = {'.py', '.yaml', '.yml', '.sh', '.toml', '.template'}
    return {name: sha256(ROOT / name) for name in sorted(set(names))
            if name and Path(name).parts[0] not in excluded and Path(name).suffix in suffixes
            and (ROOT / name).is_file()}


def require_unchanged_source(manifest):
    current = source_inventory()
    expected = manifest['source_files_sha256']
    if current != expected:
        changed = sorted(name for name in set(current) | set(expected) if current.get(name) != expected.get(name))
        raise RuntimeError('Source changed during paired pilot; refusing mixed versions: ' + ', '.join(changed[:20]))


def save_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def require_free_disk(minimum_bytes):
    available = shutil.disk_usage(str(ROOT)).free
    if available < minimum_bytes:
        raise RuntimeError('Free disk {:.2f} GiB is below required {:.2f} GiB'.format(
            available / 2**30, minimum_bytes / 2**30))
    return available


def require_gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError('GPU compute process already active; refusing a competing job: ' + ', '.join(processes))


def training_command(prefix, scope, experiment, run_id):
    return prefix + [str(ROOT / 'scripts/train_clipping_ablation.py'),
        '--config-path', str(BASE_RUN), '--config-name', 'config',
        'checkpoint=' + str(CHECKPOINT), 'experiment=' + experiment,
        'seed=43', 'pipeline=gpu', 'sim_device=cuda:0', 'rl_device=cuda:0',
        'graphics_device_id=0', 'headless=True', 'force_render=False',
        'train.params.config.max_frames=' + str(TARGET_FRAME),
        'train.params.config.max_epochs=1000000',
        'train.params.config.save_frequency=1000000000',
        '++train.params.config.gradient_clip_scope=' + scope,
        'wandb_activate=True', 'wandb_entity=' + ENTITY, 'wandb_project=' + PROJECT,
        'wandb_group=clipping_ablation_100m_pilot_20260913',
        'wandb_name=' + experiment, 'wandb_run_id=' + run_id,
        'wandb_resume=never', 'wandb_mode=online',
        'wandb_tags=[pilot,clipping_ablation,paired_continuation,single_train_seed,gpu_only]',
        'wandb_notes=single_factor_100M_continuation_pilot_no_learning_gain_claim',
        'wandb_log_code=False', 'wandb_log_diff=False',
        'hydra.run.dir=' + str(ROOT), 'hydra.output_subdir=null', 'hydra.job.chdir=False']


def stop_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_phase(name, command, environment, output, manifest, manifest_path, lock_fd, minimum_bytes, gpu):
    require_unchanged_source(manifest)
    require_free_disk(minimum_bytes)
    if gpu:
        require_gpu_idle()
    phase = {'name': name, 'state': 'starting', 'command': command,
             'utc_started_at': now(), 'log_path': str(output / (name + '.log')),
             'free_disk_bytes_at_start': shutil.disk_usage(str(ROOT)).free}
    manifest['phases'].append(phase)
    manifest['current_phase'] = name
    save_manifest(manifest_path, manifest)
    process = None
    try:
        with Path(phase['log_path']).open('x') as log:
            process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock_fd,))
            phase.update(state='running', pid=process.pid)
            save_manifest(manifest_path, manifest)
            print('Started {} pid={} log={}'.format(name, process.pid, phase['log_path']), flush=True)
            while True:
                try:
                    exit_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    phase['last_alive_at'] = now()
                    phase['free_disk_bytes_last'] = require_free_disk(minimum_bytes)
                    save_manifest(manifest_path, manifest)
                    print('Running {} pid={} free_disk={:.2f} GiB'.format(
                        name, process.pid, phase['free_disk_bytes_last'] / 2**30), flush=True)
            phase.update(state='completed' if exit_code == 0 else 'failed',
                         exit_code=exit_code, utc_finished_at=now())
            save_manifest(manifest_path, manifest)
            if exit_code != 0:
                raise RuntimeError('{} exited {}; remaining phases will not start'.format(name, exit_code))
    except BaseException as error:
        if process is not None:
            stop_child(process)
        phase.update(state='failed', error={'type': type(error).__name__, 'message': str(error)},
                     exit_code=process.returncode if process is not None else None, utc_finished_at=now())
        save_manifest(manifest_path, manifest)
        raise


def verify_training(prefix, experiment, control_path, expected_id):
    # This separate interpreter only performs W&B/file control-plane reads.
    verification = '''
import hashlib,json,pathlib,sys,wandb
control=json.loads(pathlib.Path(sys.argv[1]).read_text())
metadata=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert control['status']=='completed' and control['effective_exit_code']==0
assert control['final_frame']>=int(sys.argv[4])
assert metadata['id']==sys.argv[3] and metadata['state']=='finished' and metadata['exit_code']==0
checkpoint=pathlib.Path(control['final_checkpoint_path'])
digest=hashlib.sha256()
with checkpoint.open('rb') as stream:
 for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
assert digest.hexdigest()==control['final_checkpoint_sha256']
run=wandb.Api(timeout=30).run('{entity}/{project}/{id}'.format(**metadata))
assert run.state=='finished'
artifacts=[item for item in run.logged_artifacts() if item.type=='clipping-ablation-pilot']
assert artifacts and all(item.state=='COMMITTED' for item in artifacts)
assert any(item.metadata.get('checkpoint_sha256')==control['final_checkpoint_sha256'] for item in artifacts)
print(json.dumps({'state':run.state,'url':run.url,'artifacts':[a.name for a in artifacts],'checkpoint_sha256':control['final_checkpoint_sha256']}))
'''
    return prefix + ['-c', verification, str(control_path),
                     str(ROOT / 'runs' / experiment / 'wandb_run.json'), expected_id, str(TARGET_FRAME)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'artifacts/research/20260913-audit/clipping-ablation-pair'))
    parser.add_argument('--conda', default='/home/ubuntu22/anaconda3/bin/conda')
    parser.add_argument('--conda-env', default='epo')
    parser.add_argument('--gpu-lock', default='/tmp/epo-gpu-0.lock')
    parser.add_argument('--min-free-gib', type=float, default=2.0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--release-cache-before-rollout', action='store_true',
                        help='Both arms release inactive Torch cache before each rollout for PhysX headroom')
    parser.add_argument('--first-experiment-number', type=int, default=63)
    parser.add_argument('--release-completed-rollout-history', action='store_true',
                        help='Both arms drop consumed LF hidden history before optimizer updates')
    parser.add_argument('--disable-cuda-allocator-cache', action='store_true',
                        help='Both training arms use direct CUDA allocation; frozen evaluation keeps caching enabled')
    args = parser.parse_args()
    if args.min_free_gib < 2.0:
        parser.error('The paired pilot requires at least 2 GiB free disk')
    output = Path(args.output_dir).resolve()
    prefix = [args.conda, 'run', '-n', args.conda_env, '--no-capture-output', 'python']
    if not CHECKPOINT.is_file() or not (BASE_RUN / 'config.yaml').is_file():
        raise FileNotFoundError('E51 source checkpoint/config missing')
    input_hash = sha256(CHECKPOINT)
    arms = [(scope, str(args.first_experiment_number + index) + '_' + experiment.split('_', 1)[1])
            for index, (scope, experiment) in enumerate(ARMS)]
    plans = [{'scope': scope, 'experiment': experiment, 'wandb_run_id': fresh_wandb_id()}
             for scope, experiment in arms]
    manifest = {
        'state': 'planned', 'utc_created_at': now(), 'pid': os.getpid(),
        'interpretation': 'Single-training-seed pilot continuation; 100M does not establish learning gains or publication readiness.',
        'input_checkpoint': str(CHECKPOINT), 'input_checkpoint_sha256': input_hash,
        'input_config': str(BASE_RUN / 'config.yaml'), 'input_config_sha256': sha256(BASE_RUN / 'config.yaml'),
        'start_frame': START_FRAME, 'target_frame': TARGET_FRAME,
        'environment_frames_per_epoch': 327680, 'planned_epochs': 306,
        'expected_final_frame_after_rounding': 5100339200,
        'same_training_seed': 43, 'same_evaluation_seed': 10002,
        'only_algorithm_config_difference': 'train.params.config.gradient_clip_scope',
        'resume_semantics': 'Same saved learner state with a newly constructed simulator; not bitwise continuation of the historical training process.',
        'source_files_sha256': source_inventory(),
        'shared_nonalgorithm_changes': ['fresh output/run IDs', '100M terminal budget', 'final-only checkpoint retention', 'identical passive telemetry'],
        'release_cache_before_rollout': args.release_cache_before_rollout,
        'release_completed_rollout_history': args.release_completed_rollout_history,
        'training_cuda_allocator_cache_disabled': args.disable_cuda_allocator_cache,
        'evaluation_cuda_allocator_cache_disabled': False,
        'gpu_lock': args.gpu_lock, 'minimum_free_disk_bytes': int(args.min_free_gib * 2**30),
        'arms': plans, 'phases': [],
    }
    for plan in plans:
        plan['training_command'] = training_command(prefix, plan['scope'], plan['experiment'], plan['wandb_run_id'])
        plan['training_command'].append('++train.params.config.release_cache_before_rollout=' +
                                        str(args.release_cache_before_rollout))
        plan['training_command'].append('++train.params.config.release_completed_rollout_history=' +
                                        str(args.release_completed_rollout_history))
        plan['training_command'].append('++train.params.config.cuda_allocator_cache_disabled=' +
                                        str(args.disable_cuda_allocator_cache))
    if args.release_cache_before_rollout:
        manifest['shared_nonalgorithm_changes'].append('release unused Torch allocator cache before every rollout; record overhead and reserved bytes')
    if args.release_completed_rollout_history:
        manifest['shared_nonalgorithm_changes'].append('release consumed LF rollout history after unchanged augmentation; no output tensor alias allowed')
    if args.disable_cuda_allocator_cache:
        manifest['shared_nonalgorithm_changes'].append('disable Torch CUDA allocator cache in both training processes; retain original environment and batch sizes')
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if output.exists():
        raise FileExistsError('Refusing to overwrite pair output: ' + str(output))
    for _, experiment in arms:
        if (ROOT / 'runs' / experiment).exists():
            raise FileExistsError('Refusing to reuse training experiment: ' + experiment)
    minimum_bytes = manifest['minimum_free_disk_bytes']
    require_free_disk(minimum_bytes)
    with open(args.gpu_lock, 'a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns the EPO GPU lock: ' + args.gpu_lock)
        require_gpu_idle()
        output.mkdir(parents=True)
        manifest_path = output / 'manifest.json'
        manifest['state'] = 'running'
        save_manifest(manifest_path, manifest)
        environment = allocator_environment(os.environ, False)
        environment.update(CUDA_VISIBLE_DEVICES='0', EPO_CLIP_START_FRAME=str(START_FRAME),
                           EPO_CLIP_TARGET_FRAME=str(TARGET_FRAME),
                           EPO_CLIP_INPUT_CHECKPOINT_SHA256=input_hash,
                           PYTHONHASHSEED='43')
        environment.pop('WANDB_RUN_ID', None)
        environment.pop('WANDB_RESUME', None)
        previous_handler = signal.getsignal(signal.SIGTERM)

        def terminate(_signum, _frame):
            raise KeyboardInterrupt('Queue received SIGTERM')

        signal.signal(signal.SIGTERM, terminate)
        try:
            for plan in plans:
                scope, experiment = plan['scope'], plan['experiment']
                require_unchanged_source(manifest)
                if (ROOT / 'runs' / experiment).exists():
                    raise FileExistsError('Training experiment appeared before its phase: ' + experiment)
                if sha256(CHECKPOINT) != input_hash or sha256(BASE_RUN / 'config.yaml') != manifest['input_config_sha256']:
                    raise RuntimeError('Source checkpoint or saved config changed between arms')
                training_output = output / (scope + '-training')
                environment['EPO_CLIP_ABLATION_OUTPUT_DIR'] = str(training_output)
                training_environment = allocator_environment(environment, args.disable_cuda_allocator_cache)
                run_phase(scope + '-training', plan['training_command'], training_environment, output,
                          manifest, manifest_path, lock.fileno(), minimum_bytes, gpu=True)
                control_path = training_output / 'training_control.json'
                run_phase(scope + '-training-verify', verify_training(prefix, experiment, control_path, plan['wandb_run_id']),
                          environment, output, manifest, manifest_path, lock.fileno(), minimum_bytes, gpu=False)
                control = json.loads(control_path.read_text())
                plan['final_checkpoint_path'] = control['final_checkpoint_path']
                plan['final_checkpoint_sha256'] = control['final_checkpoint_sha256']
                plan['final_frame'] = control['final_frame']
                evaluation = output / (scope + '-master-full947')
                eval_command = prefix + [str(ROOT / 'scripts/evaluate_frozen_checkpoint.py'),
                    '--checkpoint', control['final_checkpoint_path'],
                    '--train-config', str(ROOT / 'runs' / experiment / 'config.yaml'),
                    '--output-dir', str(evaluation), '--genes', 'master', '--objects-per-gene', '0',
                    '--replicas-per-object', '1', '--eval-seed', '10002', '--object-seed', '20260913',
                    '--max-steps', '30000', '--action-mode', 'stochastic',
                    '--wandb-entity', ENTITY, '--wandb-project', PROJECT,
                    '--wandb-group', 'clipping_ablation_100m_pilot_20260913',
                    '--source-run-url', control['wandb']['url']]
                run_phase(scope + '-evaluation', eval_command, environment, output,
                          manifest, manifest_path, lock.fileno(), minimum_bytes, gpu=True)
                analysis = output / (scope + '-analysis')
                analysis_command = prefix + [str(ROOT / 'scripts/analyze_frozen_evaluation.py'),
                    '--evaluation-dir', str(evaluation), '--output-dir', str(analysis), '--master-gene', '63']
                run_phase(scope + '-analysis', analysis_command, environment, output,
                          manifest, manifest_path, lock.fileno(), minimum_bytes, gpu=False)
                plan['evaluation_output'] = str(evaluation)
                plan['analysis_output'] = str(analysis)
                plan['state'] = 'completed'
                save_manifest(manifest_path, manifest)
            manifest.update(state='completed', utc_finished_at=now(), current_phase=None)
            save_manifest(manifest_path, manifest)
        except BaseException as error:
            manifest.update(state='failed', utc_finished_at=now(),
                            error={'type': type(error).__name__, 'message': str(error)})
            save_manifest(manifest_path, manifest)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    print('Paired pilot completed with verified online training/evaluation artifacts. No learning-gain claim.', flush=True)


if __name__ == '__main__':
    main()
