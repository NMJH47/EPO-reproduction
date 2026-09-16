from types import SimpleNamespace

import isaacgym
import pytest
import torch

from rl_games.algos_torch.a2c_continuous import A2CAgent, optimizer_contains_parameter
from rl_games.algos_torch.network_builder import A2CBuilder


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


def test_crossover_preserves_gene_parameter_identity_and_gradient_anchor():
    parameter = torch.nn.Parameter(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]],
            device="cuda:0",
        )
    )
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    network = SimpleNamespace(extra_params=parameter)
    network.set_extra_params_trainable = lambda trainable: (
        A2CBuilder.Network.set_extra_params_trainable(network, trainable)
    )
    parameter.grad = torch.ones_like(parameter)
    parameter_id = id(parameter)

    A2CBuilder.Network.merge_extra_params(network, [0, 1], 2)

    assert id(network.extra_params) == parameter_id
    assert torch.equal(parameter[2], torch.tensor([2.0, 3.0], device="cuda:0"))
    assert optimizer_contains_parameter(optimizer, parameter)
    assert parameter.requires_grad
    assert parameter.grad is None


def test_legacy_equivalent_gene_is_outside_adam_but_inside_amp_clipping():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Parameter(torch.tensor(1.0, device="cuda:0"))
            self.a2c_network = torch.nn.Module()
            self.a2c_network.extra_params = torch.nn.Parameter(
                torch.tensor(2.0, device="cuda:0")
            )

    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    agent = SimpleNamespace(model=model, optimizer=optimizer)
    gene = model.a2c_network.extra_params
    gene_before = gene.detach().clone()

    assert A2CAgent.disable_gene_optimizer_updates(agent)
    assert not optimizer_contains_parameter(optimizer, gene)
    assert gene.requires_grad

    scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
    loss = model.shared.square() + gene.square()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    assert model.shared.grad.item() == pytest.approx(2.0)
    assert gene.grad.item() == pytest.approx(512.0)
    preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert preclip_norm.item() == pytest.approx((2.0 ** 2 + 512.0 ** 2) ** 0.5)

    scaler.step(optimizer)
    scaler.update()

    assert torch.equal(gene, gene_before)
    assert model.shared.item() < 1.0


def test_gene_optimizer_detachment_survives_optimizer_state_roundtrip():
    source_shared = torch.nn.Parameter(torch.tensor(1.0, device="cuda:0"))
    source_gene = torch.nn.Parameter(torch.tensor(2.0, device="cuda:0"))
    source_optimizer = torch.optim.Adam([source_shared, source_gene], lr=0.1)
    source_agent = SimpleNamespace(
        model=SimpleNamespace(
            a2c_network=SimpleNamespace(extra_params=source_gene)
        ),
        optimizer=source_optimizer,
    )

    assert A2CAgent.disable_gene_optimizer_updates(source_agent)
    optimizer_state = source_optimizer.state_dict()
    assert len(optimizer_state["param_groups"][0]["params"]) == 1

    target_shared = torch.nn.Parameter(torch.tensor(1.0, device="cuda:0"))
    target_gene = torch.nn.Parameter(torch.tensor(2.0, device="cuda:0"))
    target_optimizer = torch.optim.Adam([target_shared, target_gene], lr=0.1)
    target_agent = SimpleNamespace(
        model=SimpleNamespace(
            a2c_network=SimpleNamespace(extra_params=target_gene)
        ),
        optimizer=target_optimizer,
    )

    assert A2CAgent.disable_gene_optimizer_updates(target_agent)
    target_optimizer.load_state_dict(optimizer_state)

    assert not optimizer_contains_parameter(target_optimizer, target_gene)


def test_elite_clone_preserves_gene_parameter_identity():
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0], [2.0], [3.0]], device="cuda:0")
    )
    network = SimpleNamespace(extra_params=parameter)
    parameter_id = id(parameter)

    A2CBuilder.Network.eliminate_extra_params(network, 0, 2)

    assert id(network.extra_params) == parameter_id
    assert parameter[2].item() == 1.0
