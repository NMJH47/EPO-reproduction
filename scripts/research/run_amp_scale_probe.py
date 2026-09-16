#!/usr/bin/env python3
"""One bounded E51 continuation with a single restored AMP scale intervention.

Composes the frozen native-epoch probe; the original loss, clipping, scaler
growth/backoff, optimizer and scheduler remain in use. Scale affects clipping
of the unowned gene, so this is not a numerically equivalent speed change.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil

from scripts.research import run_training_probe as base


def calibrate_restored_scale(scaler, expected_scale, target_scale):
    before = scaler.state_dict()
    base.require(scaler.is_enabled() and before, 'Enabled restored scaler required')
    base.require(before['scale'] == expected_scale, 'Restored scale differs from protocol')
    base.require(math.isfinite(target_scale) and 0 < target_scale < expected_scale,
                 'Expected a finite positive reduction')
    requested = dict(before, scale=float(target_scale))
    scaler.load_state_dict(requested)
    after = scaler.state_dict()
    base.require(after == requested, 'Scaler state differs from requested single-field change')
    return {'before': before, 'after': after,
            'changed_fields': [key for key in before if before[key] != after[key]],
            'scope': 'Once after native restore; subsequent native dynamic scaler unchanged',
            'coupling': 'Unowned gene remains scaled during full-model gradient clipping'}


def execute(plan, plan_path, output):
    import isaacgym  # noqa: F401; simulator must precede Torch
    import wandb
    from rl_games.algos_torch.a2c_continuous import A2CAgent

    original_restore = A2CAgent.restore
    calls = []

    def restore(agent, filename, *args, **kwargs):
        result = original_restore(agent, filename, *args, **kwargs)
        base.require(not calls, 'Scale may be calibrated only once')
        record = calibrate_restored_scale(agent.scaler, plan['expected_restored_scale'],
                                          plan['initial_amp_scale'])
        calls.append(record)
        base.write(output / 'amp-scale-intervention.json', record)
        base.require(wandb.run is not None and wandb.run.id == plan['run_id'], 'Wrong online run')
        wandb.run.summary.update({'study/purpose': plan['purpose'],
            'study/intervention': 'restored_amp_initial_scale_only',
            'study/control_run_id': plan['control_run_id'],
            'study/restored_amp_scale': record['before']['scale'],
            'study/calibrated_amp_scale': record['after']['scale'],
            'study/learning_gain_claim': False})
        return result

    A2CAgent.restore = restore
    try:
        base.execute(plan, plan_path, output)
        base.require(len(calls) == 1, 'Missing scale calibration')
    finally:
        A2CAgent.restore = original_restore


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    base.require(base.identity(args.plan)['sha256'] == args.plan_sha256, 'Plan SHA differs')
    plan = json.loads(args.plan.read_text())
    base.require(plan['purpose'] == 'amp_scale_initialization_test', 'Wrong purpose')
    required = {str(Path(__file__).resolve()), str(Path(base.__file__).resolve()),
                str(base.HERE / 'chain_probe.py'), plan['config'], plan['nvml_source'],
                plan['finite_source']} | set(plan.get('extra_plan_files', {}).values())
    base.require(required <= set(plan['files_sha256']), 'Unbound executing source/config')
    for path, sha in plan['files_sha256'].items():
        base.require(base.identity(path)['sha256'] == sha, 'Reviewed input changed: ' + path)
    base.require(plan['expected_restored_scale'] == 131072 and plan['initial_amp_scale'] == 8192,
                 'Only the preregistered scale intervention is permitted')
    base.require(Path.cwd() == base.ROOT and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1',
                 'Wrong working directory or allocator mode')
    output = Path(plan['output'])
    base.require(output.parent == base.AUDIT and not output.exists(), 'Fresh study output required')
    base.require(shutil.disk_usage(base.ROOT).free >= 8 * 2**30, 'Insufficient disk reserve')
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024
           for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) == 3}
    base.require(mem['MemAvailable'] >= 16 * 2**30, 'Insufficient host RAM reserve')
    with open('/tmp/epo-gpu-0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir()
        execute(plan, args.plan, output)


if __name__ == '__main__':
    main()
