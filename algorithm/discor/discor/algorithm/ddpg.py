import os
import torch
import numpy as np
from torch.optim import Adam
from torch import nn

from .base import Algorithm
from discor.network import StateActionFunction, DeterministicPolicy
from discor.utils import disable_gradients, soft_update, update_params, assert_action

class DDPG(Algorithm):

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003,
                 policy_hidden_units=[256, 256], q_hidden_units=[256, 256],
                 target_update_coef=0.005, log_interval=10, seed=0,
                 exploration_noise=0.1): # Gaussian Noise std
        super().__init__(
            state_dim, action_dim, device, gamma, nstep, log_interval, seed)

        self.exploration_noise = exploration_noise

        # 1. Build Networks
        # Actor: Deterministic
        self._policy_net = DeterministicPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
            ).to(self._device)
        self._target_policy_net = DeterministicPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
            ).to(self._device).eval()

        # Critic: Q(s, a) (Single Q-network for standard DDPG)
        self._online_q_net = StateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device)
        self._target_q_net = StateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
            ).to(self._device).eval()

        # Copy parameters to target networks
        self._target_policy_net.load_state_dict(self._policy_net.state_dict())
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())

        # Disable gradients for targets
        disable_gradients(self._target_policy_net)
        disable_gradients(self._target_q_net)

        # Optimizers
        self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)
        
        self._target_update_coef = target_update_coef
        self._criterion = nn.MSELoss()

    def explore(self, state):
        # Action + Gaussian Noise
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self._policy_net(state)
            action = action.cpu().numpy()[0]

        # Add Noise
        noise = np.random.normal(0, self.exploration_noise, size=self._action_dim)
        action = np.clip(action + noise, -1.0, 1.0)
        
        assert_action(action)
        return action, None # Entropies are None for DDPG

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self._policy_net(state)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, None

    def update_target_networks(self):
        soft_update(self._target_q_net, self._online_q_net, self._target_update_coef)
        soft_update(self._target_policy_net, self._policy_net, self._target_update_coef)

    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        states, actions, rewards, next_states, dones = batch

        # --- Q-Function Update ---
        with torch.no_grad():
            # Target Policy Action
            next_actions = self._target_policy_net(next_states)
            # Target Q Value
            target_q_values = self._target_q_net(torch.cat([next_states, next_actions], dim=1))
            # Bellman Target
            y_target = rewards + (1.0 - dones) * self._discount * target_q_values

        # Current Q Value
        curr_q_values = self._online_q_net(torch.cat([states, actions], dim=1))
        
        q_loss = self._criterion(curr_q_values, y_target)
        update_params(self._q_optim, q_loss)

        # --- Policy Update ---
        # Maximize Q(s, pi(s)) -> Minimize -Q(s, pi(s))
        new_actions = self._policy_net(states)
        policy_loss = -self._online_q_net(torch.cat([states, new_actions], dim=1)).mean()
        update_params(self._policy_optim, policy_loss)

        # --- Logging ---
        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar('loss/Q', q_loss.detach().item(), self._learning_steps)
            writer.add_scalar('loss/policy', policy_loss.detach().item(), self._learning_steps)
            writer.add_scalar('stats/mean_Q', curr_q_values.mean().item(), self._learning_steps)

        return {
            "policy_loss": policy_loss.detach().item(),
            "q_loss": q_loss.detach().item()
        }

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_policy_net.save(os.path.join(save_dir, 'target_policy_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))

    def load_models(self, load_dir):
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'))
        self._online_q_net.load(os.path.join(load_dir, 'online_q_net.pth'))
        self._target_policy_net.load(os.path.join(load_dir, 'target_policy_net.pth'))
        self._target_q_net.load(os.path.join(load_dir, 'target_q_net.pth'))