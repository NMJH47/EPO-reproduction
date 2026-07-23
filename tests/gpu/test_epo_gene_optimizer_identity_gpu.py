import copy
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.common.a2c_common import A2CBase
from rl_games.common.evolution_optimizer import (
    adam_row_moment_norm,
    merge_policy_gene_rows_,
    optimizer_contains_parameter,
    optimizer_orphan_parameter_count,
    update_offspring_adam_state_,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


class GeneModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.a2c_network = torch.nn.Module()
        self.a2c_network.extra_params = torch.nn.Parameter(
            torch.tensor(values, dtype=torch.float32, device="cuda:0")
        )


def initialized_adam():
    model = GeneModel(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    parameter = model.a2c_network.extra_params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    parameter.sum().backward()
    optimizer.step()
    parameter.grad = None

    state = optimizer.state[parameter]
    state['exp_avg'].copy_(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0], [11.0, 12.0]],
            device="cuda:0",
        )
    )
    state['exp_avg_sq'].copy_(
        torch.tensor(
            [[2.0, 4.0], [6.0, 8.0], [18.0, 20.0], [22.0, 24.0]],
            device="cuda:0",
        )
    )
    return model, parameter, optimizer


def test_original_parameter_rebind_is_detected_as_optimizer_orphan_on_cuda():
    model = GeneModel([[1.0], [2.0], [3.0]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    old_parameter = model.a2c_network.extra_params

    model.a2c_network.extra_params = torch.nn.Parameter(old_parameter.detach().clone())

    assert not optimizer_contains_parameter(optimizer, model.a2c_network.extra_params)
    assert optimizer_contains_parameter(optimizer, old_parameter)
    assert optimizer_orphan_parameter_count(optimizer, model) == 1


def test_network_crossover_preserves_parameter_identity_on_cuda():
    model = GeneModel([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]])
    parameter = model.a2c_network.extra_params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    network = SimpleNamespace(extra_params=parameter)
    parameter_id = id(parameter)

    child = A2CBuilder.Network.merge_extra_params(network, [0, 1], 2)

    assert id(network.extra_params) == parameter_id
    assert torch.equal(child, torch.tensor([2.0, 3.0], device="cuda:0"))
    assert torch.equal(parameter[2], child)
    assert optimizer_contains_parameter(optimizer, parameter)
    assert optimizer_orphan_parameter_count(optimizer, model) == 0


@pytest.mark.parametrize('mode', ['zero', 'parent_mean', 'retain'])
def test_offspring_adam_state_modes_are_row_local_on_cuda(mode):
    _, parameter, optimizer = initialized_adam()
    state = optimizer.state[parameter]
    exp_avg_before = state['exp_avg'].clone()
    exp_avg_sq_before = state['exp_avg_sq'].clone()
    step_before = copy.deepcopy(state['step'])
    norm_before = adam_row_moment_norm(optimizer, parameter, 2)

    reported_before, reported_after = update_offspring_adam_state_(
        optimizer,
        parameter,
        [0, 1],
        2,
        mode,
    )

    assert reported_before == pytest.approx(norm_before)
    assert torch.equal(state['exp_avg'][[0, 1, 3]], exp_avg_before[[0, 1, 3]])
    assert torch.equal(state['exp_avg_sq'][[0, 1, 3]], exp_avg_sq_before[[0, 1, 3]])
    if mode == 'zero':
        assert torch.count_nonzero(state['exp_avg'][2]).item() == 0
        assert torch.count_nonzero(state['exp_avg_sq'][2]).item() == 0
        assert reported_after == 0.0
    elif mode == 'parent_mean':
        assert torch.equal(state['exp_avg'][2], exp_avg_before[[0, 1]].mean(dim=0))
        assert torch.equal(state['exp_avg_sq'][2], exp_avg_sq_before[[0, 1]].mean(dim=0))
        assert reported_after == pytest.approx(adam_row_moment_norm(optimizer, parameter, 2))
    else:
        assert torch.equal(state['exp_avg'][2], exp_avg_before[2])
        assert torch.equal(state['exp_avg_sq'][2], exp_avg_sq_before[2])
        assert reported_after == pytest.approx(norm_before)

    if isinstance(step_before, torch.Tensor):
        assert torch.equal(state['step'], step_before)
    else:
        assert state['step'] == step_before


def test_crossover_gene_updates_through_real_amp_step_on_cuda():
    model = GeneModel([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]])
    parameter = model.a2c_network.extra_params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    merge_policy_gene_rows_(parameter, [0, 1], 2)
    update_offspring_adam_state_(optimizer, parameter, [0, 1], 2, 'zero')
    parameter_id = id(parameter)
    child_before_step = parameter[2].detach().clone()

    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    with torch.cuda.amp.autocast(enabled=True):
        loss = parameter[2].square().sum()
    scaler.scale(loss).backward()

    agent = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        multi_gpu=False,
        truncate_grads=True,
        grad_norm=0.25,
    )
    A2CBase.trancate_gradients_and_step(agent)

    assert id(model.a2c_network.extra_params) == parameter_id
    assert agent.last_optimizer_owns_gene == 1.0
    assert agent.last_orphan_parameter_count == 0
    assert 0.0 < agent.last_latent_grad_norm <= 0.250001
    assert agent.last_latent_update_norm > 0.0
    assert not torch.equal(parameter[2], child_before_step)
    assert optimizer_contains_parameter(optimizer, parameter)
    assert optimizer_orphan_parameter_count(optimizer, model) == 0
