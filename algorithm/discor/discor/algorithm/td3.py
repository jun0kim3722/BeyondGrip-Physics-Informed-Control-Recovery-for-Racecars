import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base import Algorithm
from discor.utils import disable_gradients, soft_update, update_params, assert_action

import logging
logger = logging.getLogger(__name__)



class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=[256, 256, 256]):
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
        # Actor outputs [-1,1]
        return torch.tanh(self.net(states))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


class TwinQ(nn.Module):
    """Two critics in one module."""
    def __init__(self, state_dim, action_dim, hidden_units=[256, 256, 256]):
        super().__init__()
        # Q1
        layers = []
        input_dim = state_dim + action_dim
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers)

        # Q2
        layers = []
        input_dim = state_dim + action_dim
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*layers)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        sd = torch.load(path, map_location=device)
        self.load_state_dict(sd)



class TD3(Algorithm):

    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 gamma=0.99,
                 nstep=1,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 noise_std=0.2,          # exploration noise
                 policy_noise=0.2,       # target smoothing noise
                 noise_clip=0.5,         # clip target noise
                 policy_delay=2,         # delayed actor updates
                 policy_hidden_units=[256, 256, 256],
                 q_hidden_units=[256, 256, 256],
                 tau=0.005,
                 log_interval=10,
                 seed=0):

        super().__init__(state_dim, action_dim, device, gamma, nstep, log_interval, seed)

        # Actor network
        self.actor = DeterministicPolicy(state_dim, action_dim, policy_hidden_units).to(device)
        self.actor_target = DeterministicPolicy(state_dim, action_dim, policy_hidden_units).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Crtici network
        self.critic = TwinQ(state_dim, action_dim, q_hidden_units).to(device)
        self.critic_target = TwinQ(state_dim, action_dim, q_hidden_units).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        disable_gradients(self.actor_target)
        disable_gradients(self.critic_target)

        # Optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        # TD3 parameters
        self._tau = tau
        self._noise_std = noise_std
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_delay = policy_delay

        # SAC compatibility
        self.update_entropy = False

    
    def explore(self, state):
        s = torch.tensor(state[None], dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self.actor(s)

        initial_noise = 0.25
        final_noise   = 0.05
        decay_steps   = 15_000
        t = self._learning_steps

        sigma = final_noise + (initial_noise - final_noise) * max(0.0, 1.0 - t / decay_steps)

        # Gaussian exploration noise
        noise = torch.normal(
            mean=torch.zeros_like(action),
            std=sigma
        )

        action = torch.clamp(action + noise, -1, 1)

        return action.cpu().numpy()[0], None



    

    def exploit(self, state):
        """Pure deterministic actor"""
        s = torch.tensor(state[None, ...], dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self.actor(s)
        action_np = action.cpu().numpy()[0]
        assert_action(action_np)



        
        return self.explore(state)
        return action_np, torch.zeros(1, device=self._device)


    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            noise = torch.normal(0, self._policy_noise, size=actions.shape, device=self._device)
            noise = torch.clamp(noise, -self._noise_clip, self._noise_clip)

            next_actions = torch.clamp(self.actor_target(next_states) + noise, -1, 1)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            target_q = rewards + (1 - dones) * self._discount * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        update_params(self.critic_opt, critic_loss)

        stats = None
        if self._learning_steps % self._policy_delay == 0:

            policy_actions = self.actor(states)
            actor_loss = -self.critic.q1_only(states, policy_actions).mean()

            # w_gamma = 0.05
            # smooth_loss = torch.mean((policy_actions - actions)**2)
            # actor_loss += w_gamma* smooth_loss

            update_params(self.actor_opt, actor_loss)

            soft_update(self.critic_target, self.critic, self._tau)
            soft_update(self.actor_target, self.actor, self._tau)

            if self._learning_steps % self._log_interval == 0:
                writer.add_scalar("loss/critic", critic_loss.item(), self._learning_steps)
                writer.add_scalar("loss/actor", actor_loss.item(), self._learning_steps)
                stats = {
                    "critic_loss": critic_loss.item(),
                    "actor_loss": actor_loss.item(),
                }

        return stats

    def update_target_networks(self):
        """Soft-update target networks (required by Algorithm base class)."""
        soft_update(self.critic_target, self.critic, self._tau)
        soft_update(self.actor_target, self.actor, self._tau)
        return


    
    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.actor.save(os.path.join(save_dir, "actor.pth"))
        self.actor_target.save(os.path.join(save_dir, "actor_target.pth"))
        self.critic.save(os.path.join(save_dir, "critic.pth"))
        self.critic_target.save(os.path.join(save_dir, "critic_target.pth"))

    def load_models(self, load_dir):
        device = self._device
        self.actor.load(os.path.join(load_dir, "actor.pth"), device=device)
        self.actor_target.load(os.path.join(load_dir, "actor_target.pth"), device=device)
        self.critic.load(os.path.join(load_dir, "critic.pth"), device=device)
        self.critic_target.load(os.path.join(load_dir, "critic_target.pth"), device=device)
