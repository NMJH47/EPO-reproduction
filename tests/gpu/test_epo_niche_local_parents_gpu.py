import isaacgym
import pytest
import torch

from rl_games.common.custom_utils import select_niche_local_parents


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


def test_local_parent_selection_rejects_distant_global_elites():
    objectives = torch.zeros(64, dtype=torch.float32, device="cuda:0")
    objectives[62] = 100.0
    objectives[61] = 90.0
    objectives[4] = 8.0
    objectives[6] = 7.0
    objectives[5] = -10.0

    parents = select_niche_local_parents(objectives.tolist(), 5, candidate_count=8)

    assert parents == [4, 6]
    assert 61 not in parents
    assert 62 not in parents


def test_local_parent_selection_handles_population_edge():
    objectives = torch.arange(64, dtype=torch.float32, device="cuda:0")

    parents = select_niche_local_parents(objectives.tolist(), 0, candidate_count=8)

    assert parents == [8, 7]
    assert 0 not in parents


def test_local_parent_selection_validates_candidate_count():
    objectives = torch.zeros(64, dtype=torch.float32, device="cuda:0")

    with pytest.raises(ValueError, match="at least two"):
        select_niche_local_parents(objectives.tolist(), 5, candidate_count=1)
