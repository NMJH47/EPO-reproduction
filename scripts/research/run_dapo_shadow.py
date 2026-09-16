#!/usr/bin/env python3
"""Source-bound DAPO Clip-Higher shadow on the original E51 training path."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from scripts.research import run_training_probe as base

ROOT = base.ROOT
STUDY = ROOT / 'artifacts/research/20260914-dapo-transfer'


def execute(plan, plan_path, output):
    import isaacgym  # noqa: F401
    import wandb
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from scripts.research.dapo_shadow import DapoShadow
    state = {'shadow': None}
    original_init, original_restore, original_publish = A2CAgent.__init__, A2CAgent.restore, base.publish

    def init(agent, *args, **kwargs):
        original_init(agent, *args, **kwargs)
        base.require(state['shadow'] is None, 'More than one agent')
        state['shadow'] = DapoShadow(agent, output / 'dapo-shadow')

    def restore(agent, *args, **kwargs):
        result = original_restore(agent, *args, **kwargs)
        base.require(wandb.run is not None and wandb.run.id == plan['run_id'], 'Wrong run')
        wandb.run.summary.update({'study/purpose': plan['purpose'],
            'study/intervention': 'none; actor .1/.2 counterfactual only',
            'study/learning_gain_claim': False})
        return result

    def publish(run, files, label):
        invalid_success = False
        if label == 'training-chain-probe' and state['shadow'] is not None:
            # Close shadow writes before the inherited immutable publication.
            summary = state['shadow'].finalize(require_complete=False)
            files.update({'dapo-shadow/' + p.name: p for p in (output / 'dapo-shadow').iterdir() if p.is_file()})
            run.summary.update({'dapo_shadow/' + k: v for k, v in summary.items()
                                if isinstance(v, (bool, int, float, str))})
            native_summary = json.loads((output / 'summary.json').read_text())
            invalid_success = native_summary['status'] == 'completed' and not (
                summary['status'] == 'completed' and summary['all_rows_finite'])
            if invalid_success:
                native_summary.update(status='failed', effective_exit_code=1,
                    shadow_qualification_error='Incomplete/nonfinite shadow before original finish')
                base.write(output / 'summary.json', native_summary)
                run.summary['study/status'] = 'failed'
        proof = original_publish(run, files, label)
        if invalid_success:
            base.write(output / 'shadow-failure-publication.json', proof)
            raise RuntimeError('Shadow qualification failed before original observer finish')
        return proof

    A2CAgent.__init__, A2CAgent.restore, base.publish = init, restore, publish
    try:
        base.execute(plan, plan_path, output)
        base.require(state['shadow'] is not None, 'Missing shadow')
        state['shadow'].finalize(require_complete=True)
        base.require(state['shadow'].finalized['status'] == 'completed'
                     and state['shadow'].finalized['all_rows_finite'], 'Shadow completion failed')
    finally:
        primary = sys.exc_info()[1]
        A2CAgent.__init__, A2CAgent.restore, base.publish = original_init, original_restore, original_publish
        errors = []
        if state['shadow'] is not None:
            try:
                state['shadow'].finalize(require_complete=False)
            except BaseException as exc:
                errors.append(exc)
            try:
                state['shadow'].uninstall()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            base.secondary_report(output / 'shadow-cleanup-errors.json',
                {'primary': repr(primary), 'errors': [repr(exc) for exc in errors]})
            if primary is None:
                raise errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    base.require(base.identity(args.plan)['sha256'] == args.plan_sha256, 'Plan differs')
    plan = json.loads(args.plan.read_text())
    base.require(plan['purpose'] == 'dapo_clip_higher_shadow', 'Wrong purpose')
    required = {str(Path(__file__).resolve()), str(Path(base.__file__).resolve()),
        str(base.HERE / 'chain_probe.py'), str(base.HERE / 'dapo_shadow.py'),
        str(base.HERE / 'dapo_actor.py'), plan['config'], plan['nvml_source'], plan['finite_source']}
    required.update(plan.get('extra_plan_files', {}).values())
    base.require(required <= set(plan['files_sha256']), 'Unbound input')
    for path, sha in plan['files_sha256'].items():
        base.require(base.identity(path)['sha256'] == sha, 'Source/input changed: ' + path)
    base.require(Path.cwd() == ROOT and os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1',
                 'Wrong cwd or allocator')
    output = Path(plan['output'])
    base.require(output.parent == STUDY and not output.exists(), 'Fresh study output required')
    base.require(shutil.disk_usage(ROOT).free >= 8 * 2**30, 'Disk reserve failed')
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024
           for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) == 3}
    base.require(mem['MemAvailable'] >= 16 * 2**30, 'Host RAM reserve failed')
    with open('/tmp/epo-gpu-0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir()
        execute(plan, args.plan, output)


if __name__ == '__main__':
    main()
