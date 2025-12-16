import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

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
                 nstep=3,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 noise_std=0.2,          # exploration noise
                 policy_noise=0.1,       # target smoothing noise
                 noise_clip=0.3,         # clip target noise
                 policy_delay=4,         # delayed actor updates
                 policy_hidden_units=[256, 256, 256],
                 q_hidden_units=[256, 256, 256],
                 tau=0.005,
                 log_interval=10,
                 seed=0,
                 wandb_logger = None,
                 ):

        super().__init__(state_dim, action_dim, device, gamma, nstep, log_interval, seed)
        self._wandb = wandb_logger
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
        self._learning_steps = 0
        self.sigma = 0

    
    def explore(self, state):
        s = torch.tensor(state[None], dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self.actor(s)
        
        t = self._learning_steps
        if(t < 40_000):
            initial_noise = 0.4
            final_noise   = 0.1
            decay_steps   = 40_000
        else:
            initial_noise = 0.1
            final_noise   = 0.03
            decay_steps   = 20_000

        # use 0.1 to 0.03 noise from 40k to 80k ish steps

        sigma = final_noise + (initial_noise - final_noise) * max(0.0, 1.0 - t / decay_steps)
        self.sigma = sigma
        
        #self.sigma = 0.1

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



        
        #return self.explore(state)
        return action_np, torch.zeros(1, device=self._device)


    def update_online_networks(self, batch, writer):
        self._learning_steps += 1
        #states, actions, rewards, next_states, dones = batch
        states, actions, rewards, next_states, dones = batch


        with torch.no_grad():
            noise = torch.normal(0, self._policy_noise, size=actions.shape, device=self._device)
            noise = torch.clamp(noise, -self._noise_clip, self._noise_clip)

            next_actions = torch.clamp(self.actor_target(next_states) + noise, -1, 1)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            target_q = rewards + (1 - dones) * self._discount * q_next

        q1, q2 = self.critic(states, actions)
        #critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        # use Huber's loss instead of MSE to dampen outliers
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()
        # if self._learning_steps > 40_000:
        #     self._policy_delay = 4 # makes up for 2 steps per env

        #update_params(self.critic_opt, critic_loss)

        stats = None
        log_data = None
        if self._learning_steps % self._policy_delay == 0:

        ######################## actor only loss #############################
    #     obs_enabled_channels = [
    #     'speed',
    #     'gap',
    #     'LastFF',
    #     'RPM',
    #     'accelX',
    #     'accelY',
    #     'actualGear',
    #     #'angular_velocity_x',
    #     'angular_velocity_y',
    #     #'angular_velocity_z',
    #     'local_velocity_x',
    #     'local_velocity_y',
    #     #'local_velocity_z',
    #     'SlipAngle_fl',
    #     'SlipAngle_fr',
    #     'SlipAngle_rl',
    #     'SlipAngle_rr',
    # ]

    # obs_channels_info = {
    #     'speed': TOP_SPEED_MS,  # km/h
    #     'gap': 10.,
    #     'LastFF': 1.,
    #     'RPM': 10000.,
    #     'accelX': 5.,
    #     'accelY': 5.,

    #     'angular_velocity_x': np.pi,
    #     'angular_velocity_y': np.pi,
    #     'angular_velocity_z': np.pi,
    #     'local_velocity_x': TOP_SPEED_MS,
    #     'local_velocity_y': 20.,
    #     'local_velocity_z': 5.,
    #     'SlipAngle_fl': 25.,
    #     'SlipAngle_fr': 25.,
    #     'SlipAngle_rl': 25.,
    #     'SlipAngle_rr': 25.,
    #     'wheel_speed_rr': TOP_SPEED_MS / 3.6,
    #     'wheel_speed_rl': TOP_SPEED_MS / 3.6,
    #     'wheel_speed_fr': TOP_SPEED_MS / 3.6,
    #     'wheel_speed_fl': TOP_SPEED_MS / 3.6,
    #     'tyre_slip_ratio_fl': 1.,
    #     'tyre_slip_ratio_fr': 1.,
    #     'tyre_slip_ratio_rl': 1.,
    #     'tyre_slip_ratio_rr': 1.,
    #     'Dy_rr': 2.,    # lateral grip
    #     'Dy_rl': 2.,    # very noisy. Goes to zero when the wheel is spinning
    #     'Dy_fr': 2.,
    #     'Dy_fl': 2.,
    #     'LapCount': 1.,
    #     'LapDist': 1.,
    #     # commands feedback
    #     'steerAngle': 450,
    #     'accStatus': 1.,
    #     'brakeStatus': 1.,
    #     'actualGear': 8.,
    # }

            policy_actions = self.actor(states)
            L_actor_main = -self.critic.q1_only(states, policy_actions).mean()
            # previous action from replay
            delta_a = states[:, 47:50].detach()  # from obs

            L_smooth_raw = (delta_a ** 2).mean()


            target_ratio = 0.03

            gap_signed = states[:, 1] * 10.0
            v_lat = states[:, 9] * 20.0

            V_LAT_REF = 4.0
            v_lat_n = (v_lat / V_LAT_REF).clamp(-2.0, 2.0)

            L_vlat = -torch.tanh(-gap_signed * v_lat_n).mean()
            # normalize smoothness. The smoothness should always account for ~0.03 of the base actor loss.
            L_smooth = L_smooth_raw / (L_smooth_raw.detach() + 1e-6)

            actor_loss = (L_actor_main + target_ratio * L_actor_main.abs().detach() * L_smooth - 0.02 * L_vlat)


            update_params(self.actor_opt, actor_loss)

            soft_update(self.critic_target, self.critic, self._tau)
            soft_update(self.actor_target, self.actor, self._tau)

            if self._learning_steps % self._log_interval == 0:
                writer.add_scalar("loss/critic", critic_loss.item(), self._learning_steps)
                writer.add_scalar("loss/actor", actor_loss.item(), self._learning_steps)
                stats = {
                    "critic_loss": critic_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "learning_step": self._learning_steps
                }


                with torch.no_grad():
                    q1_vals, q2_vals = self.critic(states, actions)
                    td_error = torch.abs(target_q - q1_vals).mean().item()

                    mean_action = policy_actions.mean().item()
                    max_action  = policy_actions.abs().max().item()

                    done_min = dones.min().item()
                    done_max = dones.max().item()
                    done_mean = dones.float().mean().item()

                    target_q_mean = target_q.mean().item()
                    target_q_std  = target_q.std().item()

                    q_next_mean = q_next.mean().item()
                    q_next_std  = q_next.std().item()


                    dones_f = dones.float()
                    terminal_frac = dones_f.mean().item()

                    r_mean = rewards.mean().item()
                    r_min  = rewards.min().item()
                    r_max  = rewards.max().item()


                    tq_mean = target_q.mean().item()
                    tq_min  = target_q.min().item()
                    tq_max  = target_q.max().item()

                    q1_mean = q1_vals.mean().item()
                    q1_abs_mean = q1_vals.abs().mean().item()
                    q1_min = q1_vals.min().item()
                    q1_max = q1_vals.max().item()

                    term_mask = dones_f.squeeze(-1) > 0.5

                    if term_mask.any():
                        q1_term_mean = q1_vals[term_mask].mean().item()
                        r_term_mean  = rewards[term_mask].mean().item()
                        term_count   = term_mask.sum().item()
                    else:
                        q1_term_mean = 0.0
                        r_term_mean  = 0.0
                        term_count   = 0

                    td_error = torch.abs(target_q - q1_vals).mean().item()


                log_data = {
                    "actor/L_smooth_raw": L_smooth_raw.item(),
                    "actor/L_smooth_effective": (
                        target_ratio * L_actor_main.abs().item()
                    ),
                    "actor/smooth_ratio": (
                        (target_ratio * L_actor_main.abs().item()) /
                        (L_actor_main.abs().item() + 1e-6)
                    ),
                    "actor/L_actor_main": L_actor_main.item(),
                    "actor/L_actor_total": actor_loss.item(),
                    "actor/L_vlat": L_vlat.item(),
                    "actor/vlat_contrib": (0.02 * L_vlat).item(),

                    "critic/critic_loss": critic_loss.item(),
                    "actor/actor_loss": actor_loss.item(),
                    "q_values/q1_mean": q1_vals.mean().item(),
                    "q_values/q2_mean": q2_vals.mean().item(),
                    "q_values/td_error": td_error,


                    "debug/done_min": done_min,
                    "debug/done_max": done_max,
                    "debug/done_mean": done_mean,
                    "debug/target_q_mean": target_q_mean,
                    "debug/target_q_std": target_q_std,
                    "debug/q_next_mean": q_next_mean,
                    "debug/q_next_std": q_next_std,

                    "actor/mean_action": mean_action,
                    "actor/max_action": max_action,
                    "exploration/sigma": self.sigma,
                    "steps/learning_step": self._learning_steps,


                    "log/terminal_frac": terminal_frac,
                    "log/terminal_count": term_count,

                    "log/reward_mean": r_mean,
                    "log/reward_min": r_min,
                    "log/reward_max": r_max,

                    "log/target_q_mean": tq_mean,
                    "log/target_q_min": tq_min,
                    "log/target_q_max": tq_max,

                    "log/q1_mean": q1_mean,
                    "log/q1_abs_mean": q1_abs_mean,
                    "log/q1_min": q1_min,
                    "log/q1_max": q1_max,

                    "log/q1_terminal_mean": q1_term_mean,
                    "log/reward_terminal_mean": r_term_mean,

                    "log/td_error": td_error,

                }

                #self._wandb.log(log_data)

        return log_data

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
