#!/usr/bin/env python3
"""CPU-only descriptive audit of raw historical EPO TensorBoard events.

No training code is imported. Time points, blocks and episodes within one
training run are not treated as independent training seeds.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path('/home/ubuntu22/EPO')
WORKTREES = Path('/home/ubuntu22/EPO-worktrees')
RUNS = {
    'E25': WORKTREES / 'b2-control-seed43/runs/25_full_epo_B2_offactor_support_seed43_N20480',
    'E29': WORKTREES / 'b2-control-seed43/runs/29_full_epo_B3_support_critic0_seed43_N20480',
    'E37': WORKTREES / 'b2-control-seed43/runs/37_full_epo_B6_control_seed43_N20480',
    'E47': WORKTREES / 'e47-gene-ga-only/runs/47_full_epo_E7_gene_ga_only_seed43_N20480',
    'E48': WORKTREES / 'e48-gene-learnable-lineage/runs/48_full_epo_E7_gene_learnable_lineage_seed43_N20480',
    'E49': WORKTREES / 'e49-transactional-ga/runs/49_full_epo_E7_transactional_ga_seed43_N20480',
    'E50': WORKTREES / 'e50-latest-fitness/runs/50_full_epo_E7_latest_fitness_seed43_N20480',
    'E51': ROOT / 'runs/51_full_epo_E8_legacy_equivalent_identity_seed43_N20480',
}


def thresholds(steps, values, threshold):
    passing = values >= threshold
    five = next((int(steps[i]) for i in range(len(passing) - 4)
                 if passing[i:i + 5].all()), None)
    # Suffix minimum: persistence is only guaranteed at the logged observations
    # through the finite observation cutoff, never between logs or in the future.
    suffix_minimum = np.minimum.accumulate(values[::-1])[::-1]
    indices = np.flatnonzero(suffix_minimum >= threshold)
    sustained = int(steps[indices[0]]) if len(indices) else None
    return five, sustained


def read_run(run):
    paths = sorted((run / 'summaries').glob('events.out.tfevents.*'))
    if not paths:
        return {'error': 'raw event file missing', 'run_dir': str(run)}
    accumulator = EventAccumulator(str(run / 'summaries'), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()['scalars'])
    series = []
    for block in range(64):
        events = accumulator.Scalars(f'successes_per_block/block_{block}/frame')
        series.append({int(event.step): float(event.value) for event in events})
    steps = np.asarray(sorted(set.intersection(*(set(s) for s in series))))
    if not len(steps):
        raise ValueError(f'No exact common frame observations across 64 blocks: {run}')
    values = np.asarray([[s[int(step)] for s in series] for step in steps])
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite block metric: {run}')
    population = values.mean(axis=1)
    tail = steps >= 4e9
    five, strict = {}, {}
    for threshold in (5, 10, 20, 25, 30, 35):
        five[str(threshold)], strict[str(threshold)] = thresholds(steps, population, threshold)

    def scalar_summary(tag):
        if tag not in tags:
            return None
        observations = [(int(e.step), float(e.value)) for e in accumulator.Scalars(tag)]
        late = [value for step, value in observations if step >= 4e9]
        return {'points': len(observations), 'first': observations[0],
                'last': observations[-1],
                'late_logged_mean': float(np.mean(late)) if late else None}

    means = {}
    for start, end in ((0, 2e8), (2e8, 5e8), (4e9, 5e9)):
        selected = (steps >= start) & (steps <= end)
        means[f'{int(start)}_{int(end)}'] = (
            float(np.trapz(population[selected], steps[selected]) /
                  (steps[selected][-1] - steps[selected][0]))
            if selected.sum() > 1 else None)
    result = {
        'run_dir': str(run),
        'events': [{'path': str(path), 'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'n_common_block_points': len(steps),
        'first_frame': int(steps[0]), 'last_frame': int(steps[-1]),
        'population_final': float(population[-1]),
        'population_final_minblock': float(values[-1].min()),
        'population_final_maxblock': float(values[-1].max()),
        'master_final': float(values[-1, -1]),
        'population_full_timeweighted_mean': float(np.trapz(population, steps) /
                                                  (steps[-1] - steps[0])),
        'population_tail100_logged_mean': float(population[-100:].mean()),
        'population_4b_end_slope_per_billion': (
            float(np.polyfit(steps[tail] / 1e9, population[tail], 1)[0])
            if tail.sum() > 1 else None),
        'population_window_means': means,
        'five_logpoint_threshold': five,
        'strict_until_end_threshold': strict,
        'cap_rate': scalar_summary('population/new_success_at_cap_rate'),
        'lr': scalar_summary('info/last_lr'),
        'fps': scalar_summary('performance/step_inference_rl_update_fps'),
    }
    del accumulator
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', default=','.join(RUNS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'audit_date': '2026-09-13',
        'method': ('Independent CPU-only raw TensorBoard read; exact common block '
                   'steps, no interpolation between blocks; per-run AUC over '
                   'recorded interval. Strict threshold uses suffix minima through '
                   'the observation cutoff only. OLS slope is descriptive, without '
                   'independence or asymptotic-convergence claims.'),
        'runs': {},
    }
    for label in args.labels.split(','):
        if label not in RUNS:
            parser.error(f'Unknown label: {label}')
        run_result = read_run(RUNS[label])
        result['runs'][label] = run_result
        print(label, json.dumps({key: run_result[key] for key in (
            'population_final', 'population_4b_end_slope_per_billion',
            'five_logpoint_threshold', 'strict_until_end_threshold')
            if key in run_result}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
