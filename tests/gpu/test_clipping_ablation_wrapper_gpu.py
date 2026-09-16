"""CUDA equivalence checks for both branches of the continuation wrapper."""
from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch

from rl_games.common.a2c_common import A2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
from scripts import train_clipping_ablation


def make_agent(scope, epoch):
    model = torch.nn.Module()
    model.register_parameter('shared', torch.nn.Parameter(torch.tensor([1., -.5], device='cuda:0')))
    model.register_parameter('gene', torch.nn.Parameter(torch.tensor([2., -1.], device='cuda:0')))
    optimizer = torch.optim.Adam([model.shared], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.)
    scaler.scale(model.shared.square().sum() + model.gene.square().sum()).backward()
    return SimpleNamespace(model=model, optimizer=optimizer, scaler=scaler,
                           multi_gpu=False, truncate_grads=True, grad_norm=1.,
                           config={'gradient_clip_scope': scope}, epoch_num=epoch, frame=1000)


def test_wrapper_preserves_legacy_and_matches_owned_reference(tmp_path, monkeypatch):
    assert torch.cuda.is_available(), 'This algorithm check requires CUDA'
    originals = [(A2CBase, 'trancate_gradients_and_step'), (A2CBase, 'play_steps'),
                 (A2CBase, 'augment_batch_for_mixed_expl'), (A2CAgent, 'restore'),
                 (A2CAgent, 'save'), (WandbAlgoObserver, 'before_init'), (WandbAlgoObserver, 'finish')]
    # Register all process-local patches for pytest teardown, including install's writes.
    for owner, attribute in originals:
        monkeypatch.setattr(owner, attribute, getattr(owner, attribute))
    original_update = A2CBase.trancate_gradients_and_step
    lifecycle, record, _ = train_clipping_ablation.install(tmp_path, 1000, 2000, 'test-only')
    logged = []
    lifecycle['observer'] = SimpleNamespace(run=SimpleNamespace(log=logged.append))
    for epoch, scope in enumerate(('model', 'optimizer'), 1):
        actual, reference = make_agent(scope, epoch), make_agent(scope, epoch)
        gene_before = actual.model.gene.detach().clone()
        rng_before = torch.cuda.get_rng_state().clone()
        A2CBase.trancate_gradients_and_step(actual)
        assert torch.equal(torch.cuda.get_rng_state(), rng_before)
        if scope == 'model':
            original_update(reference)
        else:
            reference.scaler.unscale_(reference.optimizer)
            torch.nn.utils.clip_grad_norm_([reference.model.shared], 1.)
            reference.scaler.step(reference.optimizer)
            reference.scaler.update()
        assert torch.equal(actual.model.shared, reference.model.shared)
        assert torch.equal(actual.model.shared.grad, reference.model.shared.grad)
        assert torch.equal(actual.model.gene, gene_before)
        assert actual.scaler.state_dict() == reference.scaler.state_dict()
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.equal(actual.optimizer.state[actual.model.shared][key],
                               reference.optimizer.state[reference.model.shared][key])
    assert record['updates'] == 2
    assert record['amp_scale_drop_skip_proxy_count'] == 0
    assert len(record['first_update_metrics']) == 2
    assert logged


def test_cache_release_preserves_live_tensors_and_cuda_rng():
    assert torch.cuda.is_available(), 'Allocator validation requires CUDA'
    torch.cuda.empty_cache()
    live = torch.arange(1024, dtype=torch.float32, device='cuda:0')
    expected = live.clone()
    temporary = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
    del temporary
    rng_before = torch.cuda.get_rng_state().clone()
    row = train_clipping_ablation.release_unused_cuda_cache()
    assert torch.equal(live, expected)
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
    assert row['torch_allocated_before'] == row['torch_allocated_after']
    assert row['torch_reserved_before'] - row['torch_reserved_after'] >= 64 * 1024 * 1024
    assert row['cache_release_seconds'] >= 0


def test_completed_history_release_frees_storage_and_rejects_live_alias():
    import pytest
    assert torch.cuda.is_available(), 'History lifetime validation requires CUDA'
    batch = {'values': torch.arange(32, dtype=torch.float32, device='cuda:0')}
    expected = batch['values'].clone()
    extras = {'rnn_states': [torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda:0')]}
    rng_before = torch.cuda.get_rng_state().clone()
    row = train_clipping_ablation.release_completed_rollout_history(extras, batch)
    assert extras['rnn_states'] is None
    assert row['allocated_before'] - row['allocated_after'] == 64 * 1024 * 1024
    assert torch.equal(batch['values'], expected)
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
    extras = {'rnn_states': [torch.ones(32, device='cuda:0')]}
    aliased = {'rnn_states': [extras['rnn_states'][0][:16]]}
    with pytest.raises(RuntimeError, match='aliases'):
        train_clipping_ablation.release_completed_rollout_history(extras, aliased)
    assert extras['rnn_states'] is not None
