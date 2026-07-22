import json
import os
import sys
import types

import pytest
from omegaconf import OmegaConf

from isaacgymenvs.pbt.launcher.experiment_args import parse_bool
from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver, WandbSummaryWriter
from rl_games.common.interval_summary_writer import IntervalSummaryWriter


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.histograms = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args, kwargs))

    def add_histogram(self, *args, **kwargs):
        self.histograms.append((args, kwargs))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


class FakeRun:
    def __init__(self, run_dir, run_id, finish_error=None):
        self.dir = run_dir
        self.entity = 'test-entity'
        self.id = run_id
        self.name = 'test-name'
        self.project = 'test-project'
        self.resumed = False
        self.url = f'https://wandb.invalid/runs/{run_id}'
        self.finished_with = None
        self.defined_metrics = []
        self.logged = []
        self.finish_error = finish_error

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def log_code(self, root):
        raise AssertionError(f'log_code should be disabled, received {root}')

    def log_artifact(self, artifact):
        raise AssertionError(f'log_artifact should be disabled, received {artifact}')

    def finish(self, exit_code=0):
        self.finished_with = exit_code
        if self.finish_error is not None:
            raise self.finish_error


def build_fake_wandb(tmp_path, existing_run_ids=(), finish_error=None, api_error=None):
    calls = []
    api_calls = []
    generated_ids = []

    class CommError(Exception):
        pass

    def init(**kwargs):
        calls.append(kwargs)
        run_dir = tmp_path / 'wandb-files'
        run_dir.mkdir(exist_ok=True)
        run_id = kwargs['id'] or f'generated-{len(calls)}'
        return FakeRun(str(run_dir), run_id, finish_error=finish_error)

    class Api:
        default_entity = 'test-entity'

        def __init__(self, timeout):
            self.timeout = timeout

        def run(self, path):
            api_calls.append(path)
            if api_error is not None:
                raise api_error
            if path.rsplit('/', 1)[-1] not in existing_run_ids:
                raise CommError(f'Could not find run <Run {path} (not found)>')
            return types.SimpleNamespace(id=path.rsplit('/', 1)[-1])

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def add_file(self, path):
            self.path = path

    class Histogram:
        def __init__(self, values):
            self.values = values

    class Util:
        @staticmethod
        def generate_id():
            run_id = f'replacement-{len(generated_ids) + 1}'
            generated_ids.append(run_id)
            return run_id

    return types.SimpleNamespace(
        init=init,
        Api=Api,
        Settings=Settings,
        Artifact=Artifact,
        Histogram=Histogram,
        util=Util,
        errors=types.SimpleNamespace(CommError=CommError),
    ), calls, api_calls


def wandb_cfg():
    return OmegaConf.create({
        'wandb_project': 'test-project',
        'wandb_entity': 'test-entity',
        'wandb_group': 'test-group',
        'wandb_job_type': 'train',
        'wandb_tags': ['test'],
        'wandb_notes': 'test run',
        'wandb_name': 'test-name',
        'wandb_mode': 'online',
        'wandb_resume': 'auto',
        'wandb_run_id': '',
        'wandb_log_code': False,
        'wandb_log_diff': False,
        'wandb_sync_tensorboard': False,
        'wandb_logcode_dir': '',
        'wandb_init_timeout': 17,
    })


def test_interval_writer_throttles_scalars_and_histograms():
    clock = FakeClock()
    writer = FakeSummaryWriter()
    interval_writer = IntervalSummaryWriter(
        writer,
        {
            'defer_summaries_sec': 5,
            'summaries_interval_sec_min': 10,
            'summaries_interval_sec_max': 10,
        },
        time_fn=clock,
    )

    interval_writer.add_scalar('loss', 1.0, 0)
    clock.now = 104
    interval_writer.add_scalar('loss', 1.0, 1)
    interval_writer.add_histogram('grads', [1.0], 1)
    assert writer.scalars == []
    assert writer.histograms == []

    clock.now = 105
    interval_writer.add_scalar('loss', 2.0, 2)
    interval_writer.add_histogram('grads', [2.0], 2)
    clock.now = 109
    interval_writer.add_scalar('loss', 3.0, 3)
    interval_writer.add_histogram('grads', [3.0], 3)
    clock.now = 115
    interval_writer.add_scalar('loss', 4.0, 4)
    interval_writer.add_histogram('grads', [4.0], 4)

    assert [call[0][1] for call in writer.scalars] == [2.0, 4.0]
    assert [call[0][1] for call in writer.histograms] == [[2.0], [4.0]]


def test_wandb_summary_writer_batches_metrics_by_axis(tmp_path, monkeypatch):
    fake_wandb, _, _ = build_fake_wandb(tmp_path)
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    run = FakeRun(str(tmp_path), 'direct-log')
    writer = FakeSummaryWriter()
    wandb_writer = WandbSummaryWriter(writer, run)

    wandb_writer.add_scalar('losses/a_loss', 1.0, 100)
    wandb_writer.add_scalar('successes/frame', 2.0, 100)
    wandb_writer.add_scalar('successes/iter', 2.0, 3)
    wandb_writer.add_scalar('successes/time', 2.0, 1.5)
    wandb_writer.add_scalar('losses/a_loss', 0.5, 200)
    wandb_writer.flush()

    global_step_payloads = [payload for payload in run.logged if 'global_step' in payload]
    iteration_payloads = [payload for payload in run.logged if 'iteration' in payload]
    time_payloads = [payload for payload in run.logged if 'training_time' in payload]
    assert global_step_payloads == [
        {'global_step': 100, 'losses/a_loss': 1.0, 'successes/frame': 2.0},
        {'global_step': 200, 'losses/a_loss': 0.5},
    ]
    assert iteration_payloads == [{'iteration': 3, 'successes/iter': 2.0}]
    assert time_payloads == [{'training_time': 1.5, 'successes/time': 2.0}]


def test_wandb_observer_uses_generated_id_and_records_lifecycle(tmp_path, monkeypatch):
    fake_wandb, calls, api_calls = build_fake_wandb(tmp_path)
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    observer = WandbAlgoObserver(wandb_cfg())

    observer.before_init('run', {'train_dir': str(tmp_path)}, 'experiment')

    assert calls[0]['id'] is None
    assert calls[0]['resume'] == 'auto'
    assert calls[0]['force'] is True
    assert calls[0]['sync_tensorboard'] is False
    assert calls[0]['dir'] == os.path.join(str(tmp_path), 'experiment')
    assert calls[0]['settings'].kwargs == {'init_timeout': 17.0}
    assert api_calls == []

    metadata_path = tmp_path / 'experiment' / 'wandb_run.json'
    initial_metadata = json.loads(metadata_path.read_text())
    assert initial_metadata['id'] == 'generated-1'
    assert initial_metadata['state'] == 'running'
    assert initial_metadata['started_at']

    writer = FakeSummaryWriter()
    observer.after_init(types.SimpleNamespace(writer=writer))
    observer.finish(exit_code=143)
    observer.finish(exit_code=0)

    final_metadata = json.loads(metadata_path.read_text())
    assert final_metadata['started_at'] == initial_metadata['started_at']
    assert final_metadata['exit_code'] == 143
    assert final_metadata['finished_at']
    assert final_metadata['state'] == 'failed'
    assert writer.flush_count == 1
    assert writer.close_count == 1
    assert observer.run.finished_with == 143


def test_wandb_must_resume_requires_explicit_id(tmp_path, monkeypatch):
    fake_wandb, calls, _ = build_fake_wandb(tmp_path)
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    cfg = wandb_cfg()
    cfg.wandb_resume = 'must'

    with pytest.raises(ValueError, match='requires wandb_run_id'):
        WandbAlgoObserver(cfg).before_init('run', {'train_dir': str(tmp_path)}, 'experiment')
    assert calls == []


def test_wandb_missing_resume_run_gets_fresh_id(tmp_path, monkeypatch):
    fake_wandb, calls, api_calls = build_fake_wandb(tmp_path)
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    cfg = wandb_cfg()
    cfg.wandb_run_id = 'deleted-run'
    cfg.wandb_resume = 'allow'

    observer = WandbAlgoObserver(cfg)
    observer.before_init('run', {'train_dir': str(tmp_path)}, 'experiment')

    assert api_calls == ['test-entity/test-project/deleted-run']
    assert calls[0]['id'] == 'replacement-1'
    metadata = json.loads((tmp_path / 'experiment' / 'wandb_run.json').read_text())
    assert metadata['requested_run_id'] == 'deleted-run'
    assert metadata['replaced_run_id'] == 'deleted-run'
    assert metadata['id'] == 'replacement-1'


def test_wandb_must_resume_rejects_missing_remote_run(tmp_path, monkeypatch):
    fake_wandb, calls, _ = build_fake_wandb(tmp_path)
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    cfg = wandb_cfg()
    cfg.wandb_run_id = 'missing-run'
    cfg.wandb_resume = 'must'

    with pytest.raises(ValueError, match='requested missing W&B run'):
        WandbAlgoObserver(cfg).before_init('run', {'train_dir': str(tmp_path)}, 'experiment')
    assert calls == []


def test_wandb_never_resume_rejects_existing_remote_run(tmp_path, monkeypatch):
    fake_wandb, calls, _ = build_fake_wandb(tmp_path, existing_run_ids={'existing-run'})
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    cfg = wandb_cfg()
    cfg.wandb_run_id = 'existing-run'
    cfg.wandb_resume = 'never'

    with pytest.raises(ValueError, match='cannot use existing W&B run'):
        WandbAlgoObserver(cfg).before_init('run', {'train_dir': str(tmp_path)}, 'experiment')
    assert calls == []


def test_wandb_preflight_fails_fast_on_api_error(tmp_path, monkeypatch):
    fake_wandb, calls, _ = build_fake_wandb(tmp_path, api_error=RuntimeError('network down'))
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    cfg = wandb_cfg()
    cfg.wandb_run_id = 'resume-run'

    with pytest.raises(RuntimeError, match='Could not verify W&B run'):
        WandbAlgoObserver(cfg).before_init('run', {'train_dir': str(tmp_path)}, 'experiment')
    assert calls == []


def test_wandb_finish_failure_is_recorded_after_failed_finish(tmp_path, monkeypatch):
    fake_wandb, _, _ = build_fake_wandb(tmp_path, finish_error=RuntimeError('backend failed'))
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    observer = WandbAlgoObserver(wandb_cfg())
    observer.before_init('run', {'train_dir': str(tmp_path)}, 'experiment')

    with pytest.raises(RuntimeError, match='backend failed'):
        observer.finish(exit_code=0)

    metadata = json.loads((tmp_path / 'experiment' / 'wandb_run.json').read_text())
    assert metadata['state'] == 'finish_failed'
    assert metadata['finish_error'] == 'backend failed'
    assert metadata['finish_failed_at']
    assert 'finished_at' not in metadata


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('true', True), ('1', True), ('yes', True), ('false', False), ('0', False), ('off', False)],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_parse_bool_rejects_invalid_value():
    with pytest.raises(ValueError):
        parse_bool('sometimes')
