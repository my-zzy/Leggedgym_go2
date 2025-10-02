import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from modules.actor_critic import ActorCriticRMA
# from runner.rollout_storage import RolloutStorageWithCost
from onpolicy.store import RolloutStorageWithCost

class PPO:
    """
    Proximal Policy Optimization (PPO) with Cost Constraints
    Implements constrained RL for safe robot locomotion
    """
    
    def __init__(self, 
                 actor_critic,
                 depth_encoder=None,
                 depth_encoder_cfg=None, 
                 depth_actor=None,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 # Cost-related parameters
                 cost_value_loss_coef=0.1,
                 cost_viol_loss_coef=1.0,
                 k_value=None,
                 **kwargs):
        
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO hyperparameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Constrained PPO parameters
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.k_value = k_value if k_value is not None else torch.tensor([1.0], device=device)

        # Networks
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        
        # Estimator (placeholder - might be used for some models)
        self.estimator = None
        
        # Depth processing (if enabled)
        self.depth_encoder = depth_encoder
        self.depth_actor = depth_actor
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorageWithCost.Transition()

        # Storage will be initialized later
        self.storage = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, action_shape, cost_shape, cost_d_values):
        """Initialize rollout storage for collecting training data"""
        self.storage = RolloutStorageWithCost(
            num_envs, 
            num_transitions_per_env, 
            obs_shape, 
            privileged_obs_shape, 
            action_shape, 
            cost_shape,
            cost_d_values,
            self.device
        )

    def test_mode(self):
        """Set networks to evaluation mode"""
        self.actor_critic.eval()
        if self.depth_encoder is not None:
            self.depth_encoder.eval()

    def train_mode(self):
        """Set networks to training mode"""
        self.actor_critic.train()
        if self.depth_encoder is not None:
            self.depth_encoder.train()

    def act(self, observations, privileged_observations=None, depth_buffer=None):
        """
        Compute actions for the current observations
        Returns: actions, action log probabilities, values, cost values
        """
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Get policy outputs
        self.transition.actions = self.actor_critic.act(observations, privileged_observations=privileged_observations, hidden_states=self.transition.hidden_states)
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions)
        self.transition.action_mean = self.actor_critic.action_mean
        self.transition.action_sigma = self.actor_critic.action_std
        # Hidden states remain the same for non-recurrent networks
        
        # Get value estimates
        self.transition.values = self.actor_critic.evaluate(observations)
        
        # Get cost value estimates (for constrained training)
        if hasattr(self.actor_critic, 'evaluate_cost'):
            self.transition.cost_values = self.actor_critic.evaluate_cost(observations)
        else:
            self.transition.cost_values = torch.zeros_like(self.transition.values)
        
        self.transition.observations = observations
        self.transition.critic_observations = privileged_observations
        
        return self.transition.actions

    def process_env_step(self, rewards, costs, dones):
        """Process environment step results"""
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone() 
        self.transition.dones = dones.clone()
        
        # Store transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs=None):
        """Compute returns and advantages using GAE"""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs, last_critic_privileged_obs=None):
        """Compute cost returns and advantages using GAE"""
        if hasattr(self.actor_critic, 'evaluate_cost'):
            last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        else:
            last_cost_values = torch.zeros_like(self.actor_critic.evaluate(last_critic_obs).detach())
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam)

    def update_k_value(self, iteration):
        """Update k value for constrained RL (placeholder implementation)"""
        # Placeholder - implement actual k-value update logic if needed
        return self.k_value

    def set_imi_weight(self, weight):
        """Set imitation learning weight"""
        # Pass the weight to actor-critic if it supports imitation learning
        if hasattr(self.actor_critic, 'set_imi_weight'):
            self.actor_critic.set_imi_weight(weight)

    def update(self):
        """Update policy using PPO with cost constraints"""
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_surrogate_loss = 0
        mean_cost_surrogate_loss = 0
        mean_entropy_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch in generator:
            
            actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch, cost_value_batch = self.actor_critic.evaluate_actions(
                obs_batch, critic_obs_batch, actions_batch, hid_states_batch, masks_batch
            )

            # PPO policy loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_pred_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Cost value function loss
            cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            # Cost constraint loss (Lagrangian)
            # cost_advantages_batch has shape [batch_size, num_costs], ratio has shape [batch_size]
            # Need to expand ratio to match cost_advantages_batch dimensions
            ratio_expanded = ratio.unsqueeze(-1)  # [batch_size, 1]
            cost_surrogate_loss = (cost_advantages_batch * ratio_expanded).mean()

            # Entropy loss
            entropy_loss = entropy_batch.mean()

            # Total loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
            loss += self.cost_value_loss_coef * cost_value_loss
            loss += self.cost_viol_loss_coef * cost_surrogate_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Logging
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cost_surrogate_loss += cost_surrogate_loss.item()
            mean_entropy_loss += entropy_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_cost_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        
        # Placeholder for imitation loss (not implemented in this PPO)
        mean_imitation_loss = 0.0
        
        self.storage.clear()

        return mean_value_loss, mean_cost_value_loss, mean_cost_surrogate_loss, mean_surrogate_loss, mean_imitation_loss
