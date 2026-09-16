#!/usr/bin/env python3
"""Bind the existing CPU resource monitor to an isolated study and own run ID."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == args.plan_sha256, 'Plan changed')
    plan = json.loads(raw)
    root, output = Path(plan['execution_root']), Path(plan['output'])
    require(plan['purpose'] == 'dapo_resource_monitor', 'Wrong purpose')
    require(root.is_absolute() and root.resolve() == root, 'Execution root must be real')
    require(output.resolve() == output and output.parent == root / 'artifacts/research/20260913-audit', 'Wrong monitor output')
    require(not output.exists(), 'Existing output refused')
    required = {str(Path(__file__).resolve()), plan['monitor_source'], plan['entry']}
    require(required <= set(plan['files_sha256']), 'Missing executing source binding')
    for path, digest in plan['files_sha256'].items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'Source changed: ' + path)
    require(plan['run_id'].isalnum(), 'Own run ID required')
    if not args.execute:
        print(json.dumps({'prepared': True, 'gpu_computation': False, 'api_called': False}))
        return
    require(plan['execution_ready'] is True, 'Unbound plan')
    spec = importlib.util.spec_from_file_location('dapo_bound_original_monitor', plan['monitor_source'])
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    target = monitor.process_identity(plan['pid'])
    require(target is not None and target['start_ticks'] == plan['start_ticks'], 'Target process identity changed')
    require(str(Path(plan['entry']).resolve()) in target['argv'], 'Wrong target command')
    original_identity = monitor.process_identity

    def bound_identity(pid):
        current = original_identity(pid)
        return current if current == target else None

    monitor.process_identity = bound_identity
    monitor.ROOT = root
    output.parent.mkdir(parents=True, exist_ok=True)
    import wandb
    original_init = wandb.init
    state = {'run': None, 'finish': None, 'finish_entered': False}

    def own_init(*pos, **kwargs):
        kwargs.update(id=plan['run_id'], entity=plan['entity'], project=plan['project'],
                      group=plan['group'], resume='never', mode='online')
        kwargs['config'] = dict(kwargs.get('config', {}), adapter_plan_sha256=args.plan_sha256,
                                adapter_sha256=plan['files_sha256'][str(Path(__file__).resolve())])
        run = original_init(*pos, **kwargs)
        if run.id == plan['run_id']:
            state['run'], state['finish'] = run, run.finish

            def owned_finish(*a, **k):
                state['finish_entered'] = True
                return state['finish'](*a, **k)
            run.finish = owned_finish
        require(run.id == plan['run_id'] and not run.resumed and not run.offline, 'Unexpected W&B identity')
        return run

    previous = sys.argv
    wandb.init = own_init
    sys.argv = [plan['monitor_source'], '--pid', str(plan['pid']), '--entry', plan['entry'],
                '--output-dir', str(output), '--execute']
    try:
        monitor.main()
    except BaseException as error:
        if state['run'] is None:
            possible = getattr(wandb, 'run', None)
            if possible is not None and possible.id == plan['run_id']:
                state['run'], state['finish'] = possible, possible.finish
        if state['run'] is not None and not state['finish_entered']:
            state['finish_entered'] = True
            try:
                state['finish'](exit_code=1)
            except BaseException as secondary:
                print('Secondary monitor finish failure: ' + repr(secondary), file=sys.stderr)
        raise
    finally:
        monitor.process_identity = original_identity
        if state['run'] is not None:
            state['run'].finish = state['finish']
        wandb.init = original_init
        sys.argv = previous


if __name__ == '__main__':
    main()
