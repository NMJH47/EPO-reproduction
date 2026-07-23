import torch


OFFSPRING_ADAM_STATE_MODES = ('zero', 'parent_mean', 'retain')
_ADAM_ROW_STATE_KEYS = ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')


def validate_offspring_adam_state(mode):
    if mode not in OFFSPRING_ADAM_STATE_MODES:
        supported = ', '.join(OFFSPRING_ADAM_STATE_MODES)
        raise ValueError(f"offspring_adam_state must be one of: {supported}")
    return mode


def optimizer_contains_parameter(optimizer, parameter):
    return any(
        candidate is parameter
        for param_group in optimizer.param_groups
        for candidate in param_group['params']
    )


def optimizer_orphan_parameter_count(optimizer, model):
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    return sum(
        id(parameter) not in model_parameter_ids
        for param_group in optimizer.param_groups
        for parameter in param_group['params']
    )


def get_epo_gene_parameter(model):
    network = getattr(model, 'a2c_network', None)
    parameter = getattr(network, 'extra_params', None)
    if isinstance(parameter, torch.nn.Parameter):
        return parameter
    return None


def merge_policy_gene_rows_(parameter, parent_policies, replaced_policy):
    if not isinstance(parameter, torch.nn.Parameter):
        raise TypeError("EPO gene table must be an nn.Parameter")
    if parameter.ndim < 2:
        raise ValueError("EPO gene table must have a policy dimension and a feature dimension")
    if len(parent_policies) != 2:
        raise ValueError("EPO crossover requires exactly two parents")

    parent_policies = [int(policy_idx) for policy_idx in parent_policies]
    replaced_policy = int(replaced_policy)
    num_policies = parameter.shape[0]
    all_policies = parent_policies + [replaced_policy]
    if any(policy_idx < 0 or policy_idx >= num_policies for policy_idx in all_policies):
        raise IndexError("EPO crossover policy index is outside the gene table")
    if parent_policies[0] == parent_policies[1]:
        raise ValueError("EPO crossover parents must be distinct")
    if replaced_policy in parent_policies:
        raise ValueError("EPO offspring cannot replace one of its parents")

    with torch.no_grad():
        child = parameter[parent_policies].mean(dim=0)
        parameter[replaced_policy].copy_(child)
    return child.detach().clone()


def adam_row_moment_norm(optimizer, parameter, policy_idx):
    state = optimizer.state.get(parameter, {})
    squared_norm = parameter.new_zeros(())
    for key in _ADAM_ROW_STATE_KEYS:
        value = state.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f"Adam state {key} does not match the EPO gene table")
        squared_norm = squared_norm + value[int(policy_idx)].float().square().sum()
    return squared_norm.sqrt().item()


def update_offspring_adam_state_(optimizer, parameter, parent_policies, replaced_policy, mode):
    mode = validate_offspring_adam_state(mode)
    if not optimizer_contains_parameter(optimizer, parameter):
        raise RuntimeError("Adam does not own the current EPO gene Parameter")

    before = adam_row_moment_norm(optimizer, parameter, replaced_policy)
    state = optimizer.state.get(parameter, {})
    with torch.no_grad():
        for key in _ADAM_ROW_STATE_KEYS:
            value = state.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"Adam state {key} does not match the EPO gene table")
            if mode == 'zero':
                value[int(replaced_policy)].zero_()
            elif mode == 'parent_mean':
                parent_state = value[[int(policy_idx) for policy_idx in parent_policies]].mean(dim=0)
                value[int(replaced_policy)].copy_(parent_state)

    after = adam_row_moment_norm(optimizer, parameter, replaced_policy)
    return before, after
