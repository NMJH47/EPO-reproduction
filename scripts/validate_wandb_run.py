import argparse
import json

import torch
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--expect-state', default='finished')
    parser.add_argument('--require-metric', action='append', default=[])
    parser.add_argument('--forbid-metric', action='append', default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')

    with open(args.metadata, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    run_path = f"{metadata['entity']}/{metadata['project']}/{metadata['id']}"
    run = wandb.Api().run(run_path)
    history = list(run.scan_history(page_size=1000))
    metric_keys = sorted({
        key
        for row in history
        for key in row
        if not key.startswith('_')
    })
    missing_metrics = sorted(set(args.require_metric).difference(metric_keys))
    forbidden_metrics = sorted(set(args.forbid_metric).intersection(metric_keys))
    result = {
        'forbidden_metrics': forbidden_metrics,
        'gpu': torch.cuda.get_device_name(0),
        'history_rows': len(history),
        'metric_key_count': len(metric_keys),
        'missing_metrics': missing_metrics,
        'run_id': run.id,
        'state': run.state,
        'url': run.url,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if run.state != args.expect_state:
        raise RuntimeError(f'Expected W&B state {args.expect_state}, received {run.state}')
    if missing_metrics:
        raise RuntimeError(f'Missing required W&B metrics: {missing_metrics}')
    if forbidden_metrics:
        raise RuntimeError(f'Unexpected W&B metrics: {forbidden_metrics}')


if __name__ == '__main__':
    main()
