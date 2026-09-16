"""Inert on import/CLI; root calls run(torch_runtime) in its CUDA child.

Small synthetic original critic-loss values-leaf test, no model or optimizer.
No RNG calls. All numeric construction/autograd/comparison is CUDA. CPU output
contains scalars only. A failure is retained by the root runner, not retried.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def run(t):
    source = Path(__file__).with_name('chain_probe.py')
    spec = importlib.util.spec_from_file_location('chain_math_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from rl_games.common import common_losses
    assert t.cuda.is_available()
    assert t.__version__ == '1.8.1', 'This coefficient contract is version-specific'
    rows = []
    backward_calls = 0
    with t.enable_grad():
        # Confirm actual torch.max dispatch at an exact tie directly.
        left = t.tensor([1.0], device='cuda:0', requires_grad=True)
        right = t.tensor([1.0], device='cuda:0', requires_grad=True)
        left_grad, right_grad = t.autograd.grad(t.max(left, right).sum(), (left, right))
        backward_calls += 1
        tie_record = {'left_grad': float(left_grad.item()), 'right_grad': float(right_grad.item())}
        rows.append({'case': 'actual_max_tie', **tie_record})
        assert tie_record == {'left_grad': 0.0, 'right_grad': 1.0}, 'Analytic tie convention does not match this runtime'
        for dtype in (t.float16, t.float32, t.float64):
            # Exact +/-0.125 boundaries avoid hiding boundary status under
            # decimal FP16 rounding. Include inside/outside, targets and ties.
            values_list = [0., .125, -.125, .25, -.25, .0625, -.0625,
                           .25, -.25, 1., -1., 0., .5, -.5]
            target_list = [0., 0., 0., 0., 0., .5, -.5,
                           .1875, -.1875, 2., -2., 1., 1., -1.]
            old = t.zeros(len(values_list), 1, device='cuda:0', dtype=t.float32 if dtype == t.float16 else dtype)
            value = t.tensor(values_list, device='cuda:0', dtype=dtype).reshape(-1, 1).requires_grad_()
            target = t.tensor(target_list, device='cuda:0', dtype=old.dtype).reshape(-1, 1)
            for clip_value in (True, False):
                original = common_losses.critic_loss(None, old, value, .125, target, clip_value)
                (actual,) = t.autograd.grad(original.sum(), value)
                backward_calls += 1
                with t.no_grad():
                    analytical = module.critic_arrays(t, old, value, target, .125, clip_value)
                    loss_equal = bool(t.equal(original, analytical['loss']))
                    derivative_equal = bool(t.equal(actual, analytical['derivative']))
                    rows.append({'dtype': str(dtype), 'clip_value': clip_value,
                                 'rows': len(values_list), 'loss_equal': loss_equal,
                                 'derivative_equal': derivative_equal,
                                 'max_abs_derivative_delta': float((actual-analytical['derivative']).abs().max().item()),
                                 'ties': int(analytical['tie'].sum().item()),
                                 'actual_values_leaf_derivatives': [float(v.item()) for v in actual.reshape(-1)]})
                    assert loss_equal and derivative_equal, 'Native critic reconstruction or values derivative differs'
            # Frozen production e_clip=.1, with representable neighboring
            # values. Includes ties created by target at the two-branch midpoint.
            with t.no_grad():
                c = t.tensor(.1, device='cuda:0', dtype=dtype)
                eps = t.tensor(t.finfo(dtype).eps, device='cuda:0', dtype=dtype)
                vec = t.stack((c, -c, c + eps, c - eps, -c + eps, -c - eps))
                v_old = t.zeros_like(vec, dtype=old.dtype).reshape(-1, 1)
                v_target = ((vec.to(old.dtype) + vec.to(old.dtype).clamp(-.1, .1)) * .5).reshape(-1, 1)
            v = vec.reshape(-1, 1).detach().requires_grad_()
            original = common_losses.critic_loss(None, v_old, v, .1, v_target, True)
            (actual,) = t.autograd.grad(original.sum(), v)
            backward_calls += 1
            with t.no_grad():
                analytical = module.critic_arrays(t, v_old, v, v_target, .1, True)
                okay = bool(t.equal(actual, analytical['derivative'])) and bool(t.equal(original, analytical['loss']))
                rows.append({'case': 'production_clip_neighbors', 'dtype': str(dtype), 'rows': 6,
                             'exact': okay, 'ties': int(analytical['tie'].sum().item()),
                             'max_abs_derivative_delta': float((actual-analytical['derivative']).abs().max().item())})
                assert okay, 'Production clip near-boundary derivative differs'
    return {'status': 'completed', 'gpu_validated': True,
            'scope': 'synthetic unweighted critic values leaf; not total model gradient or chain noninterference',
            'torch_version': t.__version__, 'chain_probe_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'synthetic_backward_calls': backward_calls, 'model_forwards': 0, 'optimizer_steps': 0,
            'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps({'status': 'dry_run', 'GPU_used': False,
                      'instruction': 'Root runner must call run(existing_torch_runtime) under its reviewed GPU ownership.'}))
