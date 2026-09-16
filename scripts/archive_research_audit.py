"""Archive an audit, its exact source, and its evaluated checkpoint to W&B."""
import argparse
import json
from pathlib import Path

import wandb

from evaluate_frozen_checkpoint import ROOT, sha256_file, source_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--checkpoint-frame', type=int, required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--source-run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--entity', default='leo-ucbm-university-of-california-berkeley')
    parser.add_argument('--project', default='EPO-reproduction')
    parser.add_argument('--group', default='ceiling-audit-20260913')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite archive metadata: ' + str(output))
    for path in (args.audit_dir, args.checkpoint, args.train_config):
        if not path.exists():
            raise FileNotFoundError(str(path))
    output.mkdir(parents=True)
    source = source_snapshot(output)
    checkpoint_hash = sha256_file(args.checkpoint)
    config = {'experiment_kind': 'reproducibility_archive_not_training',
              'checkpoint_sha256': checkpoint_hash, 'checkpoint_path': str(args.checkpoint.resolve()),
              'source_sha256': source['source_sha256'], 'git_commit': source['git_commit'],
              'scope': 'Local E51 checkpoint and diagnostic evidence; no claim of improved algorithm performance.'}
    write_json(output / 'config.json', config)
    run = wandb.init(entity=args.entity, project=args.project, group=args.group,
                     job_type='audit', name=output.name, dir=str(output), config=config,
                     mode='online', resume='never', tags=['audit', 'source_archive', 'checkpoint_archive'])
    metadata = {'id': run.id, 'url': run.url, 'state': 'running'}
    write_json(output / 'wandb_run.json', metadata)
    succeeded = False
    try:
        if run.settings.mode != 'online' or run.offline:
            raise RuntimeError('Online W&B is required')
        evidence = wandb.Artifact('epo-ceiling-audit-' + run.id, type='research-audit')
        included = []
        for path in sorted(args.audit_dir.resolve().rglob('*')):
            relative = path.relative_to(args.audit_dir.resolve())
            if not path.is_file() or 'wandb' in relative.parts or output in path.parents:
                continue
            evidence.add_file(str(path), name='evidence/' + relative.as_posix())
            included.append({'path': 'evidence/' + relative.as_posix(), 'sha256': sha256_file(path)})
        for path in sorted((ROOT / 'docs/research').glob('*')):
            if path.is_file():
                evidence.add_file(str(path), name='reports/' + path.name)
                included.append({'path': 'reports/' + path.name, 'sha256': sha256_file(path)})
        for name in ('source_manifest.json', 'source_snapshot.tar.gz', 'source_diff.patch', 'config.json'):
            path = output / name
            if path.exists():
                evidence.add_file(str(path), name='source/' + name)
        write_json(output / 'evidence_manifest.json', included)
        evidence.add_file(str(output / 'evidence_manifest.json'), name='evidence_manifest.json')
        run.log_artifact(evidence).wait()

        model = wandb.Artifact('epo-e51-evaluated-checkpoint-' + run.id, type='model',
                               metadata={'sha256': checkpoint_hash, 'frame': args.checkpoint_frame,
                                         'training_seed': args.training_seed, 'original_run_id': args.source_run_id})
        model.add_file(str(args.checkpoint), name='model.pth')
        model.add_file(str(args.train_config), name='training_config.yaml')
        run.log_artifact(model).wait()
        run.summary.update({'archive/evidence_files': len(included),
                            'archive/checkpoint_sha256': checkpoint_hash,
                            'archive/checkpoint_uploaded': True,
                            'archive/algorithm_improvement_claimed': False})
        succeeded = True
    finally:
        run.finish(exit_code=0 if succeeded else 1)
        metadata['state'] = 'finished' if succeeded else 'failed'
        write_json(output / 'wandb_run.json', metadata)
    remote = wandb.Api(timeout=60).run('{}/{}/{}'.format(args.entity, args.project, run.id))
    artifacts = [{'name': item.name, 'type': item.type, 'state': item.state}
                 for item in remote.logged_artifacts()]
    assert remote.state == 'finished'
    assert all(any(item['type'] == kind and item['state'] == 'COMMITTED' for item in artifacts)
               for kind in ('research-audit', 'model'))
    validation = {'state': remote.state, 'url': remote.url, 'artifacts': artifacts,
                  'checkpoint_sha256': checkpoint_hash}
    write_json(output / 'online_validation.json', validation)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
