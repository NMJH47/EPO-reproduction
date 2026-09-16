import isaacgym
import pytest
import torch
from omegaconf import OmegaConf

from isaacgymenvs.utils.rlgames_utils import EPOObserver
from rl_games.algos_torch.a2c_continuous import (
    off_policy_importance_diagnostics,
    off_policy_training_weights,
)
from rl_games.common.custom_utils import policy_block_means


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


def observer_config():
    return OmegaConf.create(
        {
            "task": {"env": {"numEnvs": 8}},
            "train": {"params": {"config": {"expl_coef_block_size": 2}}},
        }
    )


def test_baseline_policy_block_means_preserve_all_cuda_policy_blocks():
    expected = torch.arange(64, dtype=torch.float32, device="cuda:0")
    objectives = expected.repeat_interleave(320)

    assert torch.equal(policy_block_means(objectives, 64, 320), expected)


def test_baseline_rejects_master_only_objectives_as_full_population():
    objectives = torch.zeros(320, dtype=torch.float32, device="cuda:0")

    with pytest.raises(ValueError, match="expected 20480"):
        policy_block_means(objectives, 64, 320)


def test_baseline_observer_uses_full_population_before_metric_filtering():
    observer = EPOObserver(observer_config())
    objectives = torch.arange(4, dtype=torch.float32, device="cuda:0").repeat_interleave(2)

    observer.process_infos(
        {"true_objective": objectives, "successes": objectives.clone()},
        torch.arange(8, device="cuda:0").unsqueeze(-1),
        ignore_env_boundary=6,
    )
    observer._refresh_fitness_snapshot()

    assert observer.target_objective_known
    assert observer.curr_target_objective_value_per_block == [0.0, 1.0, 2.0, 3.0]
    assert observer.best_objective_curr_iteration_per_block == [0.0, 1.0, 2.0, 3.0]
    assert observer.direct_info["true_objective"].item() == 3.0


def test_baseline_observer_waits_for_every_policy_environment():
    observer = EPOObserver(observer_config())
    objectives = torch.arange(4, dtype=torch.float32, device="cuda:0").repeat_interleave(2)

    observer.process_infos(
        {"true_objective": objectives},
        torch.arange(7, device="cuda:0").unsqueeze(-1),
        ignore_env_boundary=6,
    )
    observer._refresh_fitness_snapshot()
    assert not observer.target_objective_known

    observer.process_infos(
        {"true_objective": objectives},
        torch.tensor([[7]], device="cuda:0"),
        ignore_env_boundary=6,
    )
    observer._refresh_fitness_snapshot()
    assert observer.target_objective_known
    assert observer.curr_target_objective_value_per_block == [0.0, 1.0, 2.0, 3.0]


def test_baseline_counts_only_newly_completed_cap_episodes():
    observer = EPOObserver(observer_config())
    successes = torch.tensor(
        [50, 1, 50, 2, 50, 3, 50, 4],
        dtype=torch.float32,
        device="cuda:0",
    )

    observer._process_completed_successes(
        {"current_episode_successes": successes},
        torch.tensor([[0], [3], [6]], device="cuda:0"),
    )

    assert observer.completed_episode_count.item() == 3
    assert observer.completed_at_cap_count.item() == 2
    assert torch.equal(
        observer.completed_episode_count_per_block,
        torch.tensor([1, 1, 0, 1], device="cuda:0"),
    )


def test_off_policy_diagnostics_are_detached_and_masked():
    old_neglogp = torch.tensor([0.0, 0.0, 2.0], device="cuda:0", requires_grad=True)
    new_neglogp = torch.tensor([0.0, 1.0, 0.0], device="cuda:0", requires_grad=True)
    mask = torch.tensor([False, True, True], device="cuda:0")

    diagnostics = off_policy_importance_diagnostics(old_neglogp, new_neglogp, mask, 0.1)

    assert diagnostics is not None
    assert all(not value.requires_grad for value in diagnostics.values())
    assert diagnostics['off_policy_ratio_clip_fraction'].item() == 1.0
    assert 0.0 < diagnostics['off_policy_ratio_ess'].item() <= 1.0


def test_confirmed_off_policy_baseline_gates_actor_and_disables_critic():
    old_neglogp = torch.tensor([0.0, 0.05, 1.0], device="cuda:0")
    new_neglogp = torch.zeros(3, device="cuda:0")
    off_policy_mask = torch.tensor([False, True, True], device="cuda:0")

    weights = off_policy_training_weights(
        old_neglogp,
        new_neglogp,
        off_policy_mask,
        clip=0.1,
        actor_coef=1.0,
        critic_coef=0.0,
        support_mode="actor_clip",
    )

    assert torch.equal(weights["actor"], torch.tensor([1.0, 1.0, 0.0], device="cuda:0"))
    assert torch.equal(weights["critic"], torch.tensor([1.0, 0.0, 0.0], device="cuda:0"))
    assert weights["weight_diagnostics"]["off_policy_actor_support_fraction"].item() == 0.5
