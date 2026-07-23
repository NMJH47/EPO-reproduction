## Problem

Describe exactly one problem and the evidence that it exists.

## Hypothesis

State why the proposed change should correct the problem. The paper and current code are both hypotheses, not authorities.

## Change

Describe the smallest implementation change being tested.

## Validation

- [ ] The branch starts from the latest verified `main`.
- [ ] Static CI and security checks pass.
- [ ] Local tests run in the `epo` Conda environment.
- [ ] Isaac Gym validation uses NVIDIA CUDA with no CPU fallback.
- [ ] The experiment uses the same seed, environment count, frame budget, and evaluation protocol as its baseline.
- [ ] W&B contains the complete run, configuration, required metrics, and terminal state.
- [ ] The result supports or rejects the hypothesis without combining unrelated fixes.

## Experiment Evidence

- Baseline commit:
- Candidate commit:
- W&B baseline:
- W&B candidate:
- Seeds:
- Primary metric:
- Result:

## Merge Decision

Explain why the evidence is sufficient to merge or why the PR should remain unmerged.
