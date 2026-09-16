#!/usr/bin/env python3
"""Read-only resource samples tied to one process identity; no CUDA computation."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/ubuntu22/EPO')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def process_identity(pid):
    try:
        directory = Path('/proc') / str(pid)
        stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        return {'pid': pid, 'start_ticks': int(stat[19]), 'uid': directory.stat().st_uid,
                'argv': [value.decode() for value in argv if value]}
    except FileNotFoundError:
        return None


def memory_fields(path):
    fields = {}
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'MemAvailable'):
                fields[key] = int(value.split()[0]) * 1024
    except FileNotFoundError:
        pass
    return fields


class RemoteContentMismatch(RuntimeError):
    """Committed data differs: never disguise this as temporary invisibility."""


def verify_remote_consistent(wandb, run, artifact_name, output, files, status,
                             timeout_seconds=120.0, poll_seconds=2.0):
    """Only read after the caller's one finish; journal stays outside 4-file payload.

    No new attempt is admitted after 120 seconds. Each synchronous SDK API
    request uses a <=10 second timeout and an in-flight request may overrun
    the admission deadline. No training, upload, or finish is retried.
    """
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError('Positive remote readback time budget required')
    expected_state = 'finished' if status == 'completed' else 'failed'
    expected = {p.name: {'size': p.stat().st_size,
                'md5': base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()}
                for p in files}
    journal = output.with_name(output.name + '-remote-readback.jsonl')
    started = time.monotonic(); attempt = 0
    while True:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Remote visibility did not converge within readback budget; original run already finished')
        attempt += 1
        record = {'utc': now(), 'attempt': attempt, 'elapsed_seconds': time.monotonic()-started,
                  'expected_run_state': expected_state, 'expected_artifact': artifact_name,
                  'run_state': None, 'artifact_candidates': [], 'content_verified': False}
        fatal = None
        success = False
        try:
            remote = wandb.Api(timeout=min(10.0, remaining)).run('/'.join([run.entity, run.project, run.id]))
            record['run_state'] = remote.state
            candidates = list(remote.logged_artifacts())
            record['artifact_candidates'] = [{'name': a.name, 'state': a.state} for a in candidates]
            matches = [a for a in candidates if a.name == artifact_name and a.state == 'COMMITTED']
            if len(matches) > 1:
                raise RemoteContentMismatch('Multiple committed artifacts with the exact published name')
            if len(matches) == 1:
                entries = matches[0].manifest.entries
                if set(entries) != set(expected):
                    raise RemoteContentMismatch('Committed artifact inventory differs')
                for p in files:
                    e = entries[p.name]; original = expected[p.name]
                    if e.size != original['size'] or e.digest != original['md5']:
                        raise RemoteContentMismatch('Committed artifact MD5/size differs: ' + p.name)
                    if p.stat().st_size != original['size'] or base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode() != original['md5']:
                        raise RemoteContentMismatch('Closed local file changed during readback: ' + p.name)
                record['content_verified'] = True
                success = remote.state == expected_state
        except RemoteContentMismatch as error:
            fatal = error
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        except Exception as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
        record['outcome'] = 'verified' if success else ('content_mismatch' if fatal else 'retry_visibility')
        try:
            with journal.open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        except BaseException:
            if fatal is not None:
                raise fatal
            raise
        if fatal is not None:
            raise fatal
        if success:
            return {'remote_state': expected_state, 'artifact': artifact_name,
                    'artifact_state': 'COMMITTED', 'verified_file_count': len(files),
                    'all_md5_sizes_verified': True, 'remote_readback_attempts': attempt,
                    'remote_readback_journal': str(journal)}
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Remote visibility did not converge within readback budget; original run already finished')
        time.sleep(min(poll_seconds, remaining))


def publish(wandb, run, output, status):
    files = [path for path in sorted(output.iterdir()) if path.is_file() and path.name != 'wandb_run.json']
    artifact = wandb.Artifact('job-resources-' + run.id, type='resource-telemetry', metadata={'status': status})
    for path in files:
        artifact.add_file(str(path), name=path.name)
    committed = run.log_artifact(artifact).wait()
    assert committed.state == 'COMMITTED'
    run.finish(exit_code=0 if status == 'completed' else 1)
    return verify_remote_consistent(wandb, run, committed.name, output, files, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--entry', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode': 'dry-run', 'target_pid': args.pid, 'interval_seconds': 2,
                          'output_dir': str(args.output_dir), 'cuda_computation': False}))
        return
    identity = process_identity(args.pid)
    assert identity is not None and identity['uid'] == os.getuid()
    assert str(args.entry.resolve()) in identity['argv']
    assert args.output_dir.resolve().parent == ROOT / 'artifacts/research/20260913-audit'
    output = args.output_dir.resolve()
    output.mkdir()
    shutil.copyfile(__file__, str(output / 'monitor_source.py'))
    protocol = {'started_at': now(), 'target_identity': identity, 'interval_seconds': 2,
        'target_entry_sha256': hashlib.sha256(args.entry.read_bytes()).hexdigest(),
        'scope': 'Samples while this exact process identity remains alive; includes setup and publication if observed. Not a core-training-only measurement.',
        'limits': 'Sampling starts after target launch and may miss initial or between-sample peaks. Board power is whole-GPU power, not process-only power. No CUDA inference/training is performed by this monitor.',
        'terminal_scope': 'Process exit ends sampling; experiment success is independently verified by its own run/artifact.'}
    write(output / 'protocol.json', protocol)
    for key in ('WANDB_RUN_ID', 'WANDB_RESUME'):
        os.environ.pop(key, None)
    import wandb
    run = wandb.init(entity='leo-ucbm-university-of-california-berkeley', project='EPO-reproduction',
        group='capacity-audit-20260913', name=output.name, job_type='resource',
        mode='online', force=True, resume='never', dir=str(output), config=protocol)
    assert not run.offline and not run.resumed
    metadata = {'id': run.id, 'entity': run.entity, 'project': run.project, 'url': run.url, 'state': 'running'}
    write(output / 'wandb_run.json', metadata)
    fields = ['utc', 'elapsed_seconds', 'memory_used_mib', 'memory_free_mib', 'gpu_util_percent',
              'temperature_c', 'power_w', 'disk_free_bytes', 'mem_available_bytes', 'target_rss_bytes']
    rows = []
    start = time.monotonic()
    failure = None
    try:
        with (output / 'driver_resources.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while process_identity(args.pid) == identity:
                values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'], text=True).strip().split(',')
                numeric = [float(value.strip()) for value in values]
                assert len(numeric) == 5 and all(math.isfinite(value) for value in numeric)
                # End-of-query identity check prevents collecting a later job after PID reuse/exit.
                if process_identity(args.pid) != identity:
                    break
                row = dict(zip(fields[2:7], numeric))
                row.update(utc=now(), elapsed_seconds=time.monotonic()-start,
                    disk_free_bytes=shutil.disk_usage(ROOT).free,
                    mem_available_bytes=memory_fields(Path('/proc/meminfo')).get('MemAvailable'),
                    target_rss_bytes=memory_fields(Path('/proc')/str(args.pid)/'status').get('VmRSS'))
                writer.writerow(row); stream.flush(); rows.append(row)
                if len(rows) % 5 == 1:
                    run.log({'resource/'+key:value for key,value in row.items() if isinstance(value,(int,float))})
                time.sleep(2)
        assert rows, 'No live process samples were collected'
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    gaps = [b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(rows,rows[1:])]
    summary = {'status': 'failed' if failure else 'completed', 'finished_at': now(), 'sample_count': len(rows),
        'error': failure, 'target_success_inferred': False,
        'sampled_peak_memory_mib': max((row['memory_used_mib'] for row in rows), default=None),
        'sampled_min_free_memory_mib': min((row['memory_free_mib'] for row in rows), default=None),
        'sampled_peak_temperature_c': max((row['temperature_c'] for row in rows), default=None),
        'max_observed_sample_gap_seconds': max(gaps, default=None),
        'observed_board_energy_wh': sum((b['elapsed_seconds']-a['elapsed_seconds'])*(a['power_w']+b['power_w'])/7200 for a,b in zip(rows,rows[1:])),
        'first_sample_utc': rows[0]['utc'] if rows else None, 'last_sample_utc': rows[-1]['utc'] if rows else None}
    write(output / 'summary.json', summary)
    run.summary.update(summary)
    try:
        publication = publish(wandb, run, output, summary['status'])
        metadata.update(state='failed' if failure else 'finished', exit_code=1 if failure else 0, **publication)
    except BaseException as error:
        metadata.update(state='verification_failed', exit_code=1, error={'type':type(error).__name__,'message':str(error)})
        write(output / 'wandb_run.json', metadata)
        raise
    write(output / 'wandb_run.json', metadata)
    if failure:
        raise RuntimeError(failure['message'])


if __name__ == '__main__':
    main()
