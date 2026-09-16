import isaacgym
import pytest
import torch
from omegaconf import OmegaConf

from isaacgymenvs.utils.rlgames_utils import EPOObserver
from rl_games.common.custom_utils import select_env_tail_from_info


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


def observer_config():
    return OmegaConf.create(
        {
            "task": {"env": {"numEnvs": 8, "maxConsecutiveSuccesses": 50}},
            "train": {"params": {"config": {"expl_coef_block_size": 2}}},
        }
    )


def test_env_tail_filter_uses_views_without_mutating_population():
    population = torch.arange(8, dtype=torch.float32, device="cuda:0")
    infos = {"successes": population, "scalars": {"tolerance": 0.01}}

    filtered = select_env_tail_from_info(infos, 6)

    assert torch.equal(filtered["successes"], population[6:])
    assert torch.equal(infos["successes"], population)
    assert filtered["scalars"]["tolerance"] == 0.01


def test_vectorized_fitness_keeps_stepwise_interval_max_on_cuda():
    observer = EPOObserver(observer_config())
    done_indices = torch.arange(8, device="cuda:0").unsqueeze(-1)
    first = torch.tensor(
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        device="cuda:0",
    )

    observer._process_true_objective_fitness(
        {"true_objective": first},
        done_indices,
    )
    second = first.clone()
    second[:2] = 5.0
    observer._process_true_objective_fitness(
        {"true_objective": second},
        torch.tensor([[0], [1]], device="cuda:0"),
    )
    observer._refresh_fitness_snapshot()

    assert observer.target_objective_known
    assert observer.curr_target_objective_value_per_block == pytest.approx([5.0, 2.0, 3.0, 4.0])
    assert observer.best_objective_curr_iteration_per_block == pytest.approx([5.0, 2.0, 3.0, 4.0])
    assert observer.last_target_objectives.is_cuda
    assert observer.best_target_objectives.is_cuda


def test_completed_stats_use_current_terminal_episode_not_cached_previous_episode():
    observer = EPOObserver(observer_config())
    cached_previous = torch.zeros(8, device="cuda:0")
    current_episode = torch.tensor(
        [50.0, 2.0, 3.0, 4.0, 50.0, 6.0, 7.0, 8.0],
        device="cuda:0",
    )

    observer.process_infos(
        {
            "successes": cached_previous,
            "current_episode_successes": current_episode,
            "true_objective": torch.zeros(8, device="cuda:0"),
        },
        torch.tensor([[0], [1], [4]], device="cuda:0"),
    )

    assert observer.completed_episode_count.item() == 3
    assert observer.completed_at_cap_count.item() == 2
    assert torch.equal(
        observer.completed_episode_count_per_block,
        torch.tensor([2, 0, 1, 0], device="cuda:0"),
    )
