import json
import os
import subprocess
from datetime import datetime, timezone

from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.reformat import omegaconf_to_dict


class WandbSummaryWriter:
    def __init__(self, summary_writer, run):
        self.writer = summary_writer
        self.run = run
        self.pending = {}
        self.defined_metrics = set()

        self.run.define_metric('global_step')
        self.run.define_metric('iteration')
        self.run.define_metric('training_time')
        self.run.define_metric('*', step_metric='global_step')

    @staticmethod
    def _scalar(value):
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'item'):
            value = value.item()
        return value

    @staticmethod
    def _step_metric(tag):
        suffix = tag.rsplit('/', 1)[-1]
        if suffix == 'iter':
            return 'iteration'
        if suffix == 'time':
            return 'training_time'
        return 'global_step'

    def _buffer(self, tag, value, step):
        step_metric = self._step_metric(tag)
        if step_metric != 'global_step' and tag not in self.defined_metrics:
            self.run.define_metric(tag, step_metric=step_metric)
            self.defined_metrics.add(tag)
        step = self._scalar(step)
        pending = self.pending.get(step_metric)
        if pending is not None and pending[step_metric] != step:
            self._flush_step_metric(step_metric)
            pending = None
        if pending is None:
            pending = {step_metric: step}
            self.pending[step_metric] = pending
        pending[tag] = value

    def _flush_step_metric(self, step_metric):
        pending = self.pending.pop(step_metric, None)
        if pending is not None:
            self.run.log(pending)

    def add_scalar(self, tag, value, step, *args, **kwargs):
        self.writer.add_scalar(tag, value, step, *args, **kwargs)
        self._buffer(tag, self._scalar(value), step)

    def add_histogram(self, tag, values, step, *args, **kwargs):
        self.writer.add_histogram(tag, values, step, *args, **kwargs)

        import wandb

        wandb_values = values
        if hasattr(wandb_values, 'detach'):
            wandb_values = wandb_values.detach()
        if hasattr(wandb_values, 'cpu'):
            wandb_values = wandb_values.cpu()
        if hasattr(wandb_values, 'numpy'):
            wandb_values = wandb_values.numpy()
        self._buffer(tag, wandb.Histogram(wandb_values), step)

    def flush(self):
        self.writer.flush()
        for step_metric in list(self.pending):
            self._flush_step_metric(step_metric)

    def close(self):
        self.flush()
        self.writer.close()

    def __getattr__(self, attr):
        return getattr(self.writer, attr)


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.run = None
        self.writer = None
        self.finished = False
        self.metadata_path = None
        self.requested_run_id = None
        self.replaced_run_id = None

    @staticmethod
    def _write_json(path, payload):
        temporary_path = f'{path}.tmp'
        with open(temporary_path, 'w') as metadata_file:
            json.dump(payload, metadata_file, indent=2, sort_keys=True)
            metadata_file.write('\n')
        os.replace(temporary_path, path)

    @staticmethod
    def _read_json(path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as metadata_file:
                return json.load(metadata_file)
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _is_missing_run_error(exc):
        response = getattr(exc, 'response', None)
        if getattr(response, 'status_code', None) == 404:
            return True
        error_text = str(exc).lower()
        return 'could not find run' in error_text or 'run not found' in error_text

    def _preflight_run_id(self, wandb, run_id, resume, mode, project, entity, timeout):
        self.requested_run_id = run_id
        if mode != 'online' or run_id is None:
            return run_id

        try:
            api = wandb.Api(timeout=timeout)
            resolved_entity = entity or api.default_entity
            if not resolved_entity:
                raise RuntimeError('Could not determine the W&B entity for resume preflight')
            api.run(f'{resolved_entity}/{project}/{run_id}')
        except Exception as exc:
            if not self._is_missing_run_error(exc):
                raise RuntimeError(
                    f'Could not verify W&B run {run_id} before initialization: {exc}'
                ) from exc

            local_metadata = self._read_json(self.metadata_path)
            locally_known_run = local_metadata.get('id') == run_id
            if resume == 'must':
                raise ValueError(
                    f'wandb_resume=must requested missing W&B run {resolved_entity}/{project}/{run_id}'
                ) from exc
            if resume in {'allow', 'auto'} or locally_known_run:
                replacement_run_id = wandb.util.generate_id()
                self.replaced_run_id = run_id
                print(
                    f'W&B run {resolved_entity}/{project}/{run_id} is missing; '
                    f'starting a new run with id {replacement_run_id}'
                )
                return replacement_run_id
            return run_id

        if resume == 'never':
            raise ValueError(
                f'wandb_resume=never cannot use existing W&B run '
                f'{resolved_entity}/{project}/{run_id}'
            )
        return run_id

    def _write_run_metadata(self, state=None, exit_code=None, error=None):
        if self.run is None or self.metadata_path is None:
            return

        payload = self._read_json(self.metadata_path)
        payload.update({
            'entity': self.run.entity,
            'id': self.run.id,
            'mode': str(self.cfg.get('wandb_mode', 'online')),
            'name': self.run.name,
            'project': self.run.project,
            'resumed': bool(self.run.resumed),
            'url': self.run.url,
        })
        if self.requested_run_id is not None:
            payload['requested_run_id'] = self.requested_run_id
        if self.replaced_run_id is not None:
            payload['replaced_run_id'] = self.replaced_run_id
        if state is not None:
            payload['state'] = state
        if 'started_at' not in payload:
            payload.setdefault('started_at', datetime.now(timezone.utc).isoformat())
        if exit_code is not None:
            payload['exit_code'] = exit_code
        if state in {'finished', 'failed'}:
            payload['finished_at'] = datetime.now(timezone.utc).isoformat()
            payload.pop('finish_error', None)
            payload.pop('finish_failed_at', None)
        elif state == 'finish_failed':
            payload['finish_failed_at'] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            payload['finish_error'] = str(error)
        self._write_json(self.metadata_path, payload)

    def _log_source_state(self, repo_root):
        if self.cfg.get('wandb_log_code', True):
            self.run.log_code(root=repo_root)

        if not self.cfg.get('wandb_log_diff', True):
            return

        diff_path = os.path.join(self.run.dir, 'diff.patch')
        with open(diff_path, 'wb') as diff_file:
            result = subprocess.run(
                ['git', 'diff', 'HEAD', '--binary', '--no-ext-diff'],
                cwd=repo_root,
                stdout=diff_file,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            print(f'Could not capture git diff: {result.stderr.decode(errors="replace")}')
            return
        if os.path.getsize(diff_path) == 0:
            return

        import wandb

        diff_artifact = wandb.Artifact('source-diff', type='code', description='Uncommitted git diff')
        diff_artifact.add_file(diff_path)
        self.run.log_artifact(diff_artifact)

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        cfg = self.cfg
        resume = str(cfg.get('wandb_resume', 'auto')).lower()
        if resume not in {'allow', 'auto', 'must', 'never'}:
            raise ValueError(f'Unsupported wandb_resume value: {resume}')

        run_id = str(cfg.get('wandb_run_id', '')).strip() or None
        if resume == 'must' and run_id is None:
            raise ValueError('wandb_resume=must requires wandb_run_id')

        experiment_dir = os.path.abspath(os.path.join(config.get('train_dir', 'runs'), experiment_name))
        os.makedirs(experiment_dir, exist_ok=True)
        self.metadata_path = os.path.join(experiment_dir, 'wandb_run.json')

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        code_root = cfg.get('wandb_logcode_dir') or repo_root
        mode = str(cfg.get('wandb_mode', 'online')).lower()
        init_timeout = float(cfg.get('wandb_init_timeout', 120))
        run_id = self._preflight_run_id(
            wandb,
            run_id,
            resume,
            mode,
            cfg.wandb_project,
            cfg.wandb_entity or None,
            init_timeout,
        )
        run_config = cfg if isinstance(cfg, dict) else omegaconf_to_dict(cfg)

        print('Initializing WandB...')
        self.run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            group=cfg.wandb_group or None,
            job_type=cfg.get('wandb_job_type', 'train'),
            tags=list(cfg.wandb_tags),
            notes=cfg.get('wandb_notes', ''),
            config=run_config,
            dir=experiment_dir,
            sync_tensorboard=bool(cfg.get('wandb_sync_tensorboard', False)),
            id=run_id,
            name=cfg.get('wandb_name') or experiment_name,
            mode=mode,
            force=mode == 'online',
            resume=resume,
            settings=wandb.Settings(init_timeout=init_timeout),
        )
        print(f'WandB run id: {self.run.id}')
        print(f'WandB run URL: {self.run.url}')
        print(f'WandB local directory: {self.run.dir}')
        self._write_run_metadata(state='running')

        try:
            self._log_source_state(code_root)
        except Exception as exc:
            print(f'Could not log source state to WandB: {exc}')

    def after_init(self, algo):
        self.writer = algo.writer

    def wrap_summary_writer(self, writer):
        if self.run is None or self.cfg.get('wandb_sync_tensorboard', False):
            return writer
        return WandbSummaryWriter(writer, self.run)

    def finish(self, exit_code=0):
        if self.finished:
            return

        writer_error = None
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception as exc:
                writer_error = exc
        effective_exit_code = exit_code if writer_error is None else (exit_code or 1)
        if self.run is not None:
            self._write_run_metadata(state='finishing', exit_code=effective_exit_code)
            try:
                self.run.finish(exit_code=effective_exit_code)
            except Exception as exc:
                self._write_run_metadata(
                    state='finish_failed',
                    exit_code=effective_exit_code,
                    error=exc,
                )
                self.finished = True
                raise
            final_state = 'finished' if effective_exit_code == 0 else 'failed'
            self._write_run_metadata(
                state=final_state,
                exit_code=effective_exit_code,
                error=writer_error,
            )
        self.finished = True
        if writer_error is not None:
            raise RuntimeError(f'Could not flush W&B summary writer: {writer_error}') from writer_error
