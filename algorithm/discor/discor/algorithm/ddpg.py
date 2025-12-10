import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base import Algorithm
from discor.utils import disable_gradients, soft_update, update_params, assert_action

import logging
logger = logging.getLogger(__name__)


# placed nn modules here instead of network.py bc i am lazy
class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return torch.tanh(self.net(states))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


class StateActionFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

"""
    This implementation adds Gaussian Noise rather than OU noise.
"""
class DDPG(Algorithm):

    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 gamma=0.99,
                 nstep=1,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 noise_std=0.1,
                 policy_hidden_units=[256, 256],
                 q_hidden_units=[256, 256],
                 tau=0.001,
                 log_interval=10,
                 seed=0):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            gamma=gamma,
            nstep=nstep,
            log_interval=log_interval,
            seed=seed
        )

        # Actor
        self._policy_net = DeterministicPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
        ).to(self._device)

        self._policy_target = DeterministicPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
        ).to(self._device)

        # Critic
        self._q_net = StateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
        ).to(self._device)

        self._q_target = StateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
        ).to(self._device)

        # Copy parameters of the learning network to the target network.
        self._policy_target.load_state_dict(self._policy_net.state_dict())
        self._q_target.load_state_dict(self._q_net.state_dict())
        self._q_target.eval()
        self._policy_target.eval()

        disable_gradients(self._q_target)
        disable_gradients(self._policy_target)

        # Optimizers
        self._policy_optim = Adam(self._policy_net.parameters(), lr=actor_lr)
        self._q_optim = Adam(self._q_net.parameters(), lr=critic_lr)

        self._tau = tau
        self._noise_std = noise_std

        self.update_entropy = False


    def explore(self, state):
        state_t = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device
        )
        with torch.no_grad():
            action = self._policy_net(state_t)

        noise = torch.normal(
            mean=torch.zeros_like(action),
            std=self._noise_std
        )
        action = torch.clamp(action + noise, -1.0, 1.0)
        action_np = action.cpu().numpy()[0]
        assert_action(action_np)

        # Dummy entropy because driver main code uses SAC
        return action_np, torch.zeros(1, device=self._device)

    # This is determinstic
    def exploit(self, state):
        state_t = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device
        )
        with torch.no_grad():
            action = self._policy_net(state_t)
        action_np = action.cpu().numpy()[0]
        assert_action(action_np)
        return action_np, torch.zeros(1, device=self._device)


    def update_target_networks(self):
        soft_update(self._q_target, self._q_net, self._tau)
        soft_update(self._policy_target, self._policy_net, self._tau)


    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        states, actions, rewards, next_states, dones = batch

        # Critic update.
        with torch.no_grad():
            next_actions = self._policy_target(next_states)
            next_q = self._q_target(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self._discount * next_q

        curr_q = self._q_net(states, actions)
        q_loss = F.mse_loss(curr_q, target_q)

        update_params(self._q_optim, q_loss)


        policy_actions = self._policy_net(states)
        policy_loss = - self._q_net(states, policy_actions).mean()
        update_params(self._policy_optim, policy_loss)

        stats = None
        if self._learning_steps % self._log_interval == 0:
            q_loss_v = q_loss.detach().item()
            policy_loss_v = policy_loss.detach().item()
            mean_q = curr_q.detach().mean().item()

            writer.add_scalar('loss/Q', q_loss_v, self._learning_steps)
            writer.add_scalar('loss/policy', policy_loss_v, self._learning_steps)
            writer.add_scalar('stats/mean_Q', mean_q, self._learning_steps)

            stats = {
                "Q_loss": q_loss_v,
                "policy_loss": policy_loss_v,
                "mean_Q": mean_q
            }

        return stats


    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._policy_target.save(os.path.join(save_dir, 'policy_target_net.pth'))
        self._q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._q_target.save(os.path.join(save_dir, 'target_q_net.pth'))


    def load_models(self, load_dir):
        device = self._device
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'), device=device)
        self._policy_target.load(os.path.join(load_dir, 'policy_target_net.pth'), device=device)
        self._q_net.load(os.path.join(load_dir, 'online_q_net.pth'), device=device)
        self._q_target.load(os.path.join(load_dir, 'target_q_net.pth'), device=device)
