#!/usr/bin/env python3
"""Regenerate the exact object catalog and audit fixed policy/object assignment.

CPU only: import the asset generator directly; execute only two extracted task
methods, without importing Isaac Gym, torch, or a task package initializer.
"""
import ast
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT = Path('/home/ubuntu22/EPO')
OUT = ROOT / 'docs/research/object_assignment_audit_20260913.json'
CONFIG = ROOT / 'artifacts/research/20260913-audit/e51-eval-smoke/training_config.resolved.json'
TASK_SOURCE = ROOT / 'isaacgymenvs/tasks/allegro_kuka/allegro_kuka_base.py'
GENERATOR = ROOT / 'isaacgymenvs/tasks/allegro_kuka/generate_cuboids.py'
LOG = Path('/home/ubuntu22/EPO-baselines/e7-fundamental-corrections/logs/51_full_epo_E8_legacy_equivalent_identity_seed43_N20480.log')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads(CONFIG.read_text())
    ec = cfg['task']['env']
    n = int(ec['numEnvs'])
    b = int(cfg['train']['params']['config']['expl_coef_block_size'])
    k = n // b
    spec = importlib.util.spec_from_file_location('epo_asset_generator_only', GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    namespace = {'os': os, 'join': os.path.join, 'np': np}
    namespace.update({name: getattr(generator, name) for name in (
        'generate_default_cube', 'generate_small_cuboids', 'generate_big_cuboids', 'generate_sticks')})
    cls = next(node for node in ast.parse(TASK_SOURCE.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'AllegroKukaBase')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_main_object_assets_and_scales', '_box_asset_files_and_scales')]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(TASK_SOURCE), 'exec'), namespace)
    probe = SimpleNamespace(
        object_type=ec['objectType'], asset_files_dict={'block': ec['asset']['assetFileNameBlock']},
        object_base_size=ec['objectBaseSize'], with_small_cuboids=ec['withSmallCuboids'],
        with_big_cuboids=ec['withBigCuboids'], with_sticks=ec['withSticks'],
        randomize_object_dimensions=ec['randomizeObjectDimensions'])
    probe._box_asset_files_and_scales = lambda root, folder: namespace['_box_asset_files_and_scales'](probe, root, folder)
    with tempfile.TemporaryDirectory(prefix='epo_object_audit_') as folder:
        with contextlib.redirect_stdout(io.StringIO()):
            files, scales = namespace['_main_object_assets_and_scales'](probe, str(ROOT / 'assets'), folder)
        catalog = [{'index': index, 'asset_name': Path(path).name, 'scale': list(scale),
                    'asset_sha256': digest(Path(path))} for index, (path, scale) in enumerate(zip(files, scales))]
    smoke = json.loads((CONFIG.parent / 'object_panel.json').read_text())
    assert len(catalog) == smoke['catalog_size']
    for item in smoke['objects']:
        actual = catalog[item['original_object_index']]
        assert all(actual[key] == item[key] for key in ('asset_name', 'scale', 'asset_sha256'))
    c = len(catalog)
    assignment = np.arange(n).reshape(k, b) % c
    sets = [set(row.tolist()) for row in assignment]
    overlap = np.asarray([[len(left & right) for right in sets] for left in sets])
    scale_values = np.asarray([item['scale'] for item in catalog])
    volume = scale_values.prod(axis=1)
    sorted_scales = np.sort(scale_values, axis=1)
    aspect = sorted_scales[:, -1] / sorted_scales[:, 0]
    # Geometry labels are explicit derived descriptions, not asset-family provenance.
    elongated = sorted_scales[:, -1] > 3 * sorted_scales[:, -2]
    small = volume <= 2.5 + 1e-12
    geometry = [tuple(item['scale']) for item in catalog]

    def summary(indices):
        indices = np.asarray(indices, dtype=int)
        return {'n_assets': len(indices), 'unique_geometries': len({geometry[i] for i in indices}),
                'mean_scale_xyz': scale_values[indices].mean(axis=0).tolist(),
                'mean_volume_relative_cube': float(volume[indices].mean()),
                'mean_aspect_ratio': float(aspect[indices].mean()),
                'small_volume_le_2p5_fraction': float(small[indices].mean()),
                'elongated_max_gt_3x_middle_fraction': float(elongated[indices].mean())}

    blocks = []
    for gene, row in enumerate(assignment):
        candidates = sorted((other for other in range(k) if other != gene),
                            key=lambda other: (abs(other - gene), other))[:8]
        blocks.append({'gene': gene, 'object_indices': row.tolist(), **summary(row),
                       'master_overlap_count': int(overlap[gene, k - 1]),
                       'local8_candidates': candidates,
                       'local8_overlap_counts': overlap[gene, candidates].tolist()})
    counts = np.bincount(assignment.reshape(-1), minlength=c)
    all_pair = overlap[~np.eye(k, dtype=bool)]
    local = [overlap[block['gene'], donor] for block in blocks for donor in block['local8_candidates']]
    ga = []
    for match in re.finditer(r'Best_policies=\[(\d+), (\d+)\], worst_policies=(\d+): Start replacing embeddings', LOG.read_text()):
        p0, p1, child = map(int, match.groups())
        ga.append({'parent0': p0, 'parent1': p1, 'child': child,
                   'parent_child_overlap': [int(overlap[child, p0]), int(overlap[child, p1])],
                   'parent_pair_overlap': int(overlap[p0, p1]),
                   'parent_child_gene_distance': [abs(child - p0), abs(child - p1)]})
    master = blocks[-1]
    master_geometry = {geometry[i] for i in assignment[-1]}
    all_geometry = set(geometry)
    result = {
        'audit_date': '2026-09-13',
        'interpretation': 'Fixed object composition is confounded with gene/entropy slot. These are exact assignment facts, not causal evidence that object imbalance explains performance.',
        'source_sha256': {str(path): digest(path) for path in (CONFIG, TASK_SOURCE, GENERATOR, LOG, Path(__file__))},
        'num_envs': n, 'num_genes': k, 'block_size': b, 'catalog_size': c,
        'catalog_unique_geometries': len(all_geometry),
        'catalog_matches_smoke_indices_and_hashes': True,
        'catalog': catalog, 'per_gene': blocks,
        'full_catalog_geometry_summary': summary(np.arange(c)),
        'training_population_geometry_summary': summary(assignment.reshape(-1)),
        'full_catalog_environment_multiplicities': dict(sorted(Counter(counts.tolist()).items())),
        'each_gene_distinct_asset_count': sorted(set(len(s) for s in sets)),
        'master_unseen_asset_indices': sorted(set(range(c)) - sets[-1]),
        'master_asset_coverage_fraction': len(sets[-1]) / c,
        'master_geometry_coverage_fraction': len(master_geometry) / len(all_geometry),
        'master_uniform_catalog_total_variation_asset_indices': 1 - len(sets[-1]) / c,
        'pairwise_overlap_counts': overlap.tolist(),
        'distance_overlap': {str(d): sorted(set(int(overlap[i, i + d]) for i in range(k - d))) for d in range(1, k)},
        'all_distinct_pair_mean_overlap': float(np.mean(all_pair)),
        'local8_candidate_mean_overlap': float(np.mean(local)),
        'gene_geometry_summary_ranges': {field: [min(block[field] for block in blocks), max(block[field] for block in blocks)] for field in (
            'mean_volume_relative_cube', 'mean_aspect_ratio', 'small_volume_le_2p5_fraction', 'elongated_max_gt_3x_middle_fraction')},
        'e51_ga_events': ga,
        'e51_ga_parent_child_mean_overlap': float(np.mean([value for event in ga for value in event['parent_child_overlap']])) if ga else None,
        'e51_ga_parent_child_distance_counts': dict(sorted(Counter(value for event in ga for value in event['parent_child_gene_distance']).items())),
        'e51_ga_parent_child_overlap_ge_90pct_fraction': float(np.mean([value >= 0.9 * b for event in ga for value in event['parent_child_overlap']])) if ga else None,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'catalog_size', 'catalog_unique_geometries', 'full_catalog_environment_multiplicities',
        'master_asset_coverage_fraction', 'master_geometry_coverage_fraction',
        'full_catalog_geometry_summary', 'gene_geometry_summary_ranges',
        'all_distinct_pair_mean_overlap', 'local8_candidate_mean_overlap',
        'e51_ga_parent_child_mean_overlap', 'e51_ga_parent_child_distance_counts',
        'e51_ga_parent_child_overlap_ge_90pct_fraction')}, indent=2))
    print('master', json.dumps({key: value for key, value in master.items() if key != 'object_indices'}))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
