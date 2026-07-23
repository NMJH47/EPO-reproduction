import isaacgym
import pytest
import torch
from omegaconf import OmegaConf

from isaacgymenvs.utils.rlgames_utils import EPOObserver
from rl_games.common.custom_utils import filter_leader, policy_block_means, rank_follower_policies


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


def test_policy_block_means_preserve_all_cuda_policy_blocks():
    num_policies = 64
    block_size = 320
    expected = torch.arange(num_policies, dtype=torch.float32, device="cuda:0")
    objectives = expected.repeat_interleave(block_size)

    actual = policy_block_means(objectives, num_policies, block_size)

    assert torch.equal(actual, expected)


def test_master_only_objectives_cannot_be_misread_as_full_population():
    master_objectives = torch.zeros(320, dtype=torch.float32, device="cuda:0")

    with pytest.raises(ValueError, match="expected 20480"):
        policy_block_means(master_objectives, 64, 320)


def test_follower_ranking_excludes_master_on_cuda():
    fitness = torch.tensor([1.0, 3.0, 2.0, 1000.0], device="cuda:0")

    ranked = rank_follower_policies(fitness, master_policy_idx=3)

    assert ranked == [1, 2, 0]


def test_leader_filter_maps_off_policy_samples_to_last_policy_block():
    num_policies = 4
    block_size = 2
    population_size = num_policies * block_size
    repeat_idxs = list(range(num_policies))
    policy_ids = torch.arange(num_policies, device="cuda:0").repeat_interleave(block_size)
    relabeled_ids = torch.cat(
        [torch.roll(policy_ids, block_size * repeat_idx) for repeat_idx in repeat_idxs]
    )

    filtered_ids = filter_leader(relabeled_ids, population_size, repeat_idxs, num_policies)

    expected_off_policy_ids = torch.full(
        ((num_policies - 1) * block_size,),
        num_policies - 1,
        device="cuda:0",
    )
    assert torch.equal(filtered_ids[:population_size], policy_ids)
    assert torch.equal(filtered_ids[population_size:], expected_off_policy_ids)


def test_observer_uses_full_population_for_fitness_and_master_for_metrics():
    cfg = OmegaConf.create(
        {
            "task": {"env": {"numEnvs": 8}},
            "train": {"params": {"config": {"expl_coef_block_size": 2}}},
        }
    )
    observer = EPOObserver(cfg)
    expected_block_fitness = [0.0, 1.0, 2.0, 3.0]
    true_objective = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        dtype=torch.float32,
        device="cuda:0",
    )
    done_indices = torch.arange(8, device="cuda:0").unsqueeze(-1)

    observer.process_infos(
        {"true_objective": true_objective, "successes": true_objective.clone()},
        done_indices,
        ignore_env_boundary=6,
    )

    assert observer.target_objective_known
    assert observer.master_policy_idx == 3
    assert observer.curr_target_objective_value_per_block == expected_block_fitness
    assert observer.best_objective_curr_iteration_per_block == expected_block_fitness
    assert observer.direct_info["true_objective"].item() == 3.0
    assert observer.direct_info["true_objective_per_block/block_0"] == 0.0
    assert observer.direct_info["true_objective_per_block/block_3"] == 3.0


def test_observer_waits_for_every_cuda_environment_to_finish_an_episode():
    cfg = OmegaConf.create(
        {
            "task": {"env": {"numEnvs": 8}},
            "train": {"params": {"config": {"expl_coef_block_size": 2}}},
        }
    )
    observer = EPOObserver(cfg)
    true_objective = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        dtype=torch.float32,
        device="cuda:0",
    )

    observer.process_infos(
        {"true_objective": true_objective},
        torch.arange(7, device="cuda:0").unsqueeze(-1),
        ignore_env_boundary=6,
    )

    assert not observer.target_objective_known
    assert observer.finished_agents == set(range(7))

    observer.process_infos(
        {"true_objective": true_objective},
        torch.tensor([[7]], device="cuda:0"),
        ignore_env_boundary=6,
    )

    assert observer.target_objective_known
    assert observer.curr_target_objective_value_per_block == [0.0, 1.0, 2.0, 3.0]


def test_after_steps_never_selects_or_replaces_master_policy():
    class RecordingWriter:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = (value, step)

    class RecordingNetwork:
        def __init__(self):
            self.extra_params = torch.arange(4, dtype=torch.float32, device="cuda:0")
            self.merge_calls = []

        def merge_extra_params(self, parents, replaced_policy):
            self.merge_calls.append((list(parents), replaced_policy))

    class FakeAlgo:
        def __init__(self):
            self.frame = 200_000_000
            self.device = torch.device("cuda:0")
            self.writer = RecordingWriter()
            self.model = type("FakeModel", (), {})()
            self.model.a2c_network = RecordingNetwork()

    cfg = OmegaConf.create(
        {
            "task": {"env": {"numEnvs": 8}},
            "train": {"params": {"config": {"expl_coef_block_size": 2}}},
        }
    )
    observer = EPOObserver(cfg)
    algo = FakeAlgo()
    observer.after_init(algo)
    observer.sapg_iteration = 9
    observer.target_objective_known = True
    observer.best_objective_curr_iteration_per_block = [10.0, 9.0, 8.0, 1000.0]

    observer.after_steps()

    assert algo.model.a2c_network.merge_calls == [([0, 1], 2)]
    assert algo.writer.scalars["evolution/master_policy_idx"] == (3, 200_000_000)
    assert algo.writer.scalars["evolution/parent_policy_0"] == (0, 200_000_000)
    assert algo.writer.scalars["evolution/parent_policy_1"] == (1, 200_000_000)
    assert algo.writer.scalars["evolution/replaced_policy"] == (2, 200_000_000)
