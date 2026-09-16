import os
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch
import torch.distributed as dist 


def optimizer_contains_parameter(optimizer, parameter):
    return any(
        candidate is parameter
        for param_group in optimizer.param_groups
        for candidate in param_group['params']
    )


def remove_parameter_from_optimizer(optimizer, parameter):
    removed = False
    for param_group in optimizer.param_groups:
        retained = [
            candidate
            for candidate in param_group['params']
            if candidate is not parameter
        ]
        if len(retained) != len(param_group['params']):
            param_group['params'] = retained
            removed = True
    optimizer.state.pop(parameter, None)
    return removed


def off_policy_importance_diagnostics(old_neglogp, new_neglogp, off_policy_mask, clip):
    selected_log_ratio = torch.masked_select(
        old_neglogp - new_neglogp,
        off_policy_mask,
    ).detach()
    if selected_log_ratio.numel() == 0:
        return None

    finite_fraction = torch.isfinite(selected_log_ratio).float().mean()
    safe_log_ratio = torch.nan_to_num(
        selected_log_ratio,
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    ).clamp(-20.0, 20.0)
    weights = torch.exp(safe_log_ratio)
    normalized_ess = weights.sum().square() / (
        weights.numel() * weights.square().sum().clamp_min(1e-12)
    )
    clip_fraction = torch.logical_or(
        weights < 1.0 - clip,
        weights > 1.0 + clip,
    ).float().mean()
    return {
        'off_policy_ratio_ess': normalized_ess,
        'off_policy_ratio_clip_fraction': clip_fraction,
        'off_policy_log_ratio_abs_mean': safe_log_ratio.abs().mean(),
        'off_policy_log_ratio_finite_fraction': finite_fraction,
    }


def off_policy_training_weights(
    old_neglogp,
    new_neglogp,
    off_policy_mask,
    clip,
    actor_coef=1.0,
    critic_coef=1.0,
    support_mode='none',
):
    if actor_coef < 0.0:
        raise ValueError("off_policy_actor_coef must be non-negative")
    if critic_coef < 0.0:
        raise ValueError("off_policy_critic_coef must be non-negative")
    if support_mode not in {'none', 'actor_clip'}:
        raise ValueError(f"unsupported off-policy support mode: {support_mode}")

    actor_weights = torch.ones_like(new_neglogp).detach()
    critic_weights = torch.ones_like(new_neglogp).detach()
    result = {
        'actor': actor_weights,
        'critic': critic_weights,
        'diagnostics': None,
        'weight_diagnostics': {},
    }
    if off_policy_mask is None:
        return result

    flat_off_policy_mask = off_policy_mask.reshape(-1).bool()
    if flat_off_policy_mask.numel() != actor_weights.numel():
        raise ValueError("off_policy_mask must match the flattened PPO batch")
    if not flat_off_policy_mask.any():
        return result

    diagnostics = off_policy_importance_diagnostics(
        old_neglogp,
        new_neglogp,
        flat_off_policy_mask,
        clip,
    )
    selected_actor_weights = torch.full_like(
        torch.masked_select(actor_weights, flat_off_policy_mask),
        actor_coef,
    )
    if support_mode == 'actor_clip':
        selected_log_ratio = torch.masked_select(
            old_neglogp - new_neglogp,
            flat_off_policy_mask,
        ).detach()
        selected_ratio = torch.exp(
            torch.nan_to_num(
                selected_log_ratio,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(-20.0, 20.0)
        )
        selected_actor_weights *= torch.logical_and(
            selected_ratio >= 1.0 - clip,
            selected_ratio <= 1.0 + clip,
        ).float()

    actor_weights[flat_off_policy_mask] = selected_actor_weights
    critic_weights[flat_off_policy_mask] = critic_coef
    result.update({
        'diagnostics': diagnostics,
        'weight_diagnostics': {
            'off_policy_actor_effective_weight': selected_actor_weights.mean(),
            'off_policy_critic_effective_weight': critic_weights[flat_off_policy_mask].mean(),
            'off_policy_actor_support_fraction': (selected_actor_weights > 0.0).float().mean(),
        },
    })
    return result


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        
        if self.intr_reward_coef_embd is not None and not (self.expl_type.startswith('mixed_expl') and 'disjoint' in self.expl_type):
            input_shape = (self.obs_shape[0] + self.intr_reward_coef_embd.shape[1],)
        else:
            input_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : input_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'type' : 'simple' if 'learn_param' not in self.expl_type else 'extra_param',
        }
        
        if self.expl_type.startswith('mixed_expl'):
            build_config['coef_ids'] = self.intr_reward_coef_embd[::self.intr_coef_block_size,0]
            print( self.obs_shape)
            build_config['coef_id_idx'] = self.obs_shape[0]
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        log_off_policy_grads_env = os.getenv('LOG_OFF_POLICY_GRADS', '').strip().lower()
        self.log_off_policy_grads = bool(self.config.get('log_off_policy_grads', False)) or log_off_policy_grads_env in {
            '1', 'true', 'yes', 'on'
        }
        self.off_policy_actor_coef = float(self.config.get('off_policy_actor_coef', 1.0))
        self.off_policy_critic_coef = float(self.config.get('off_policy_critic_coef', 1.0))
        self.off_policy_support_mode = str(self.config.get('off_policy_support_mode', 'none'))
        if self.off_policy_actor_coef < 0.0:
            raise ValueError("off_policy_actor_coef must be non-negative")
        if self.off_policy_critic_coef < 0.0:
            raise ValueError("off_policy_critic_coef must be non-negative")
        if self.off_policy_support_mode not in {'none', 'actor_clip'}:
            raise ValueError(f"unsupported off-policy support mode: {self.off_policy_support_mode}")

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn, override_state=None):
        state = {self.global_rank: self.get_full_state_weights()}
        if override_state is not None:
            state = override_state
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint[self.global_rank] if self.global_rank in checkpoint else checkpoint, set_epoch=set_epoch)

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        network = self.model.a2c_network
        if hasattr(network, 'extra_params'):
            state['epo_gene_trainable'] = network.extra_params.requires_grad
            state['epo_gene_optimizer_active'] = optimizer_contains_parameter(
                self.optimizer,
                network.extra_params,
            )
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        network = self.model.a2c_network
        if (
            hasattr(network, 'extra_params')
            and not bool(weights.get('epo_gene_optimizer_active', True))
        ):
            self.disable_gene_optimizer_updates()
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if hasattr(network, 'extra_params'):
            network.set_extra_params_trainable(
                bool(weights.get('epo_gene_trainable', True))
            )

    def disable_gene_optimizer_updates(self):
        network = self.model.a2c_network
        if not hasattr(network, 'extra_params'):
            return False
        return remove_parameter_from_optimizer(
            self.optimizer,
            network.extra_params,
        )

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        off_policy_mask = input_dict.get('off_policy_mask')
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            training_weights = off_policy_training_weights(
                old_action_log_probs_batch,
                action_log_probs,
                off_policy_mask,
                curr_e_clip,
                actor_coef=self.off_policy_actor_coef,
                critic_coef=self.off_policy_critic_coef,
                support_mode=self.off_policy_support_mode,
            )
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)
            a_loss = a_loss * training_weights['actor']

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                c_loss = c_loss * training_weights['critic'].unsqueeze(1)
            else:
                c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)
            
            # ! AHHHHH
            if self.expl_type.startswith('mixed_expl') and self.config.get('expl_reward_type') == 'entropy':
                ec_candidates = self.intr_reward_coef[::self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
                ec_indices = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            elif self.expl_type.startswith('simple') and self.config.get('expl_reward_type') == 'entropy':
                entropy_coef = self.intr_reward_coef
            else:
                entropy_coef = self.entropy_coef
            
            collect_off_policy_diagnostics = (
                self.log_off_policy_grads and input_dict.get('off_policy_mask') is not None
            )
            if collect_off_policy_diagnostics:
                loss_arr = a_loss.unsqueeze(1) + 0.5 * c_loss * self.critic_coef - (entropy_coef*entropy).unsqueeze(1) + b_loss.unsqueeze(1) * self.bounds_loss_coef
                off_policy_loss = torch.masked_select(loss_arr, input_dict['off_policy_mask'].unsqueeze(1)).sum() / len(input_dict['off_policy_mask'])
                on_policy_loss = torch.masked_select(loss_arr, ~input_dict['off_policy_mask'].unsqueeze(1)).sum() / len(input_dict['off_policy_mask'])
                grads_off = self.get_grads(off_policy_loss)
                grads_on = self.get_grads(on_policy_loss)

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , (entropy_coef*entropy).unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy_loss, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0) 

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()

        extras = {'off_policy_diagnostics_valid': collect_off_policy_diagnostics}
        if off_policy_mask is not None:
            diagnostics = training_weights['diagnostics']
            if diagnostics is not None:
                extras.update({
                    name: value.item()
                    for name, value in diagnostics.items()
                })
            extras.update({
                name: value.item()
                for name, value in training_weights['weight_diagnostics'].items()
            })

            sample_kl = torch_ext.policy_kl(
                mu.detach(),
                sigma.detach(),
                old_mu_batch,
                old_sigma_batch,
                reduce=False,
            )
            valid_mask = torch.ones_like(sample_kl, dtype=torch.bool)
            if rnn_masks is not None:
                valid_mask &= rnn_masks.reshape(-1).bool()
            flat_off_policy_mask = off_policy_mask.reshape(-1).bool()
            on_policy_kl = sample_kl[valid_mask & ~flat_off_policy_mask]
            off_policy_kl = sample_kl[valid_mask & flat_off_policy_mask]
            if on_policy_kl.numel() > 0:
                extras['on_policy_behavior_kl'] = on_policy_kl.mean().item()
            if off_policy_kl.numel() > 0:
                extras['off_policy_behavior_kl'] = off_policy_kl.mean().item()
        if collect_off_policy_diagnostics:
            contrib_off = torch.masked_select(contrib, input_dict['off_policy_mask'])
            contrib_on = torch.masked_select(contrib, ~input_dict['off_policy_mask'])
            contrib_off = torch.nan_to_num(contrib_off.mean())
            contrib_on = torch.nan_to_num(contrib_on.mean())
        
            extras.update({
                "off_policy_contrib" : contrib_off.item(),
                "on_policy_contrib" : contrib_on.item(),
                "off_policy_grads" : grads_off.detach().cpu(),
                "on_policy_grads" : grads_on.detach().cpu(),
            })
        if self.expl_type.startswith('mixed_expl'):
            bl_ids = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
            bl_idxs = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [torch.nan_to_num(entropy[bl_idxs == i].detach().mean()).item() for i in range(self.num_actors // self.intr_coef_block_size)]
        self.train_result = (a_loss, c_loss, torch_ext.apply_masks([entropy.unsqueeze(1)], rnn_masks)[0][0], \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss, extras)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
    def get_grads(self, loss, retain_graph=True):
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None

        loss.backward(retain_graph=retain_graph)

        all_grads_list = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads_list.append(param.grad.view(-1))

        all_grads = torch.cat(all_grads_list)
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        
        return all_grads
