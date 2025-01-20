import torch
import torch.nn as nn

from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm
from ..utils.vcse import PBE,VCSE,TorchRunningMeanStd

torch.autograd.set_detect_anomaly(True)

class PPOTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.args = args
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm

        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

        # 是否使用gmm
        self.use_gmm = args.use_gmm
        self.use_inter_kl = args.use_inter_kl
        self.inter_kl_coef = args.inter_kl_coef

        #cma
        self.use_ppoloss=False

        # vcse config
        self.knn_k = 10
        self.use_state_reward=True
        self.use_vcse = True
        self.vcse = VCSE(self.knn_k,self.device)
        self.se = PBE(False, 0, self.knn_k, False, False, self.device)
        self.s_ent_stats = TorchRunningMeanStd(shape=[1], device=self.device)

    def compute_intr_reward_vcse(self, state, value):
        reward, _, _, _, _, _ = self.vcse(state,value)
        reward = reward.reshape(-1, 1)
        return reward

    def compute_intr_reward_se(self, state):
        reward = self.se(state)
        reward = reward.reshape(-1, 1)
        return reward
    
    #更新PPO网络
    def ppo_update(self, policy: PPOPolicy, sample):

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)

        #使用原本的PPO更新
        if self.use_ppoloss:
            # Obtain the loss function两种value，values是带熵的，extr_values是不带熵的
            values, action_log_probs, dist_entropy = policy.evaluate_actions(obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, masks_batch)
            extr_values, _ = policy.extr_critic(obs_batch, rnn_states_critic_batch, masks_batch)
            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
            policy_loss = -policy_loss.mean()

            extr_return_batch = returns_batch
            # 计算intrinsic_reward
            if self.use_state_reward:
                if self.use_vcse : 
                    intrinsic_reward = self.compute_intr_reward_vcse(rnn_states_actor_batch, extr_values)
                    self.s_ent_stats.update(intrinsic_reward)
                    #intrinsic_reward = intrinsic_reward / self.s_ent_stats.std
                else :
                    intrinsic_reward = self.compute_intr_reward_se(rnn_states_actor_batch)
                    self.s_ent_stats.update(intrinsic_reward)
                    intrinsic_reward = intrinsic_reward / self.s_ent_stats.mean
                returns_batch = returns_batch + intrinsic_reward

            if self.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            else:
                value_loss = 0.5 * (returns_batch - values).pow(2)
            value_loss = value_loss.mean()

            extr_value_loss = 0.5 * (extr_return_batch - extr_values).pow(2)
            extr_value_loss = extr_value_loss.mean()

            policy_entropy_loss = -dist_entropy.mean()

            loss = extr_value_loss * self.value_loss_coef + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef + policy_loss

            # Optimize the loss function
            policy.optimizer.zero_grad()
            loss.backward()
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
                critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
            else:
                actor_grad_norm = get_gard_norm(policy.actor.parameters())
                critic_grad_norm = get_gard_norm(policy.critic.parameters())
            policy.optimizer.step()

            return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm 

        #使用PPOCMA方法更新   
        else:
            #先更新协方差矩阵
            action_log_probs_var = policy.evaluate_actions_var(obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, masks_batch)
            # Detach gradients for policy mean and variance
            ratio_var = torch.exp(action_log_probs_var - old_action_log_probs_batch)
            surr1_var = ratio_var * advantages_batch
            surr2_var = torch.clamp(ratio_var, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            policyvar_loss = torch.sum(torch.min(surr1_var, surr2_var), dim=-1, keepdim=True)
            policyvar_loss = -policyvar_loss.mean()
            # Optimize the loss function
            policy.actor_optimizer.zero_grad()
            policyvar_loss.backward()
            policy.actor_optimizer.step()

            ##再更新Mean网络和价值网络
            action_log_probs_mean = policy.evaluate_actions_mean(obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, masks_batch)
            # Detach gradients for policy mean and variance
            ratio_mean = torch.exp(action_log_probs_mean - old_action_log_probs_batch)
            surr1_mean = ratio_mean * advantages_batch
            surr2_mean = torch.clamp(ratio_mean, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            policymean_loss = torch.sum(torch.min(surr1_mean, surr2_mean), dim=-1, keepdim=True)
            policymean_loss = -policymean_loss.mean()

            # Obtain the loss function两种value，values是带熵的，extr_values是不带熵的
            values, _, dist_entropy = policy.evaluate_actions(obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, masks_batch)
            extr_values, _ = policy.extr_critic(obs_batch, rnn_states_critic_batch, masks_batch)
            
            extr_return_batch = returns_batch
            # 计算intrinsic_reward
            if self.use_state_reward:
                if self.use_vcse : 
                    intrinsic_reward = self.compute_intr_reward_vcse(obs_batch, extr_values)
                    self.s_ent_stats.update(intrinsic_reward)
                    # intrinsic_reward = intrinsic_reward / self.s_ent_stats.mean
                else :
                    intrinsic_reward = self.compute_intr_reward_se(obs_batch)
                    self.s_ent_stats.update(intrinsic_reward)
                    intrinsic_reward = intrinsic_reward / self.s_ent_stats.mean
                returns_batch = returns_batch + intrinsic_reward
                
            if self.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            else:
                value_loss = 0.5 * (returns_batch - values).pow(2)
            value_loss = value_loss.mean()

            extr_value_loss = 0.5 * (extr_return_batch - extr_values).pow(2)
            extr_value_loss = extr_value_loss.mean()

            policy_entropy_loss = -dist_entropy.mean()

            loss = extr_value_loss * self.value_loss_coef + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef + policymean_loss

            # Optimize the loss function
            policy.optimizer.zero_grad()
            loss.backward()
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
                critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
            else:
                actor_grad_norm = get_gard_norm(policy.actor.parameters())
                critic_grad_norm = get_gard_norm(policy.critic.parameters())
            policy.optimizer.step()

            return policymean_loss, value_loss, policy_entropy_loss, ratio_mean, actor_grad_norm, critic_grad_norm 

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        # train_info['policySigmaLoss'] = 0
        # train_info['policyMeanLoss'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(self.args, buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                # raise NotImplementedError
                data_generator = ReplayBuffer.recurrent_generator(self.args, buffer, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()
                # train_info['policySigmaLoss'] += policySigmaLoss.item()
                # train_info['policyMeanLoss'] += policyMeanLoss.item()


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
