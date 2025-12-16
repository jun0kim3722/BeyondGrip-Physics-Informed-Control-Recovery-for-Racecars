# python train_recovery.py
import os
import sys
import argparse
import logging
import numpy as np
import torch
from datetime import datetime
from omegaconf import OmegaConf
from collections import deque


sys.path.append(os.path.abspath('./assetto_corsa_gym'))
sys.path.append(os.path.abspath('./algorithm/discor'))

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from assetto_corsa_gym.AssettoCorsaEnv.recovery_ac_env import RecoveryAssettoEnv
from assetto_corsa_gym.AssettoCorsaEnv.recovery_ac_env import RecoveryAssettoEnv
from algorithm.discor.discor.replay_buffer import EliteReplayBuffer


from assetto_corsa_gym.AssettoCorsaEnv.recovery_phys_ac_env import PhysicsRecoveryEnv

import common.logging_config as logging_config
from common.logger import Logger  # WandB Logger

from discor.agent import Agent
from discor.algorithm.sac import SAC  # SAC
from discor.algorithm.td3 import TD3  # SAC



# from discor.algorithm.ddpg import DDPG  # DDPG

try:
    from mpc_controller import MPC_controller
except ImportError:
    MPC_controller = None

logger = logging.getLogger("RecoveryTrainer")


def trunc_norm(mu, sigma, low, high):
    while True:
        x = mu + sigma * np.random.randn()
        if low <= x <= high:
            return x

class Destabilizer:
    def __init__(self, feint_duration=10, counter_duration=20):
        self.step_counter = 0
        self.feint_duration = feint_duration
        self.counter_duration = counter_duration

        self.direction = 1.0
        self.intensity_factor = 1.0
        self.counter_weight = 3.5

    def reset(self, current_speed_kmh, base_mag, random_steer):
        self.step_counter = 0
       
        if random_steer:
            self.direction = 1.0 if np.random.random() > 0.5 else -1.0
        else:
            self.direction = -1.0


        random_noise = np.random.normal(0, 0.05)

        low_intensity = 0.3
        high_intensity = 1.0
        raw_intensity = trunc_norm(base_mag, 0.1, low_intensity, high_intensity)
        #raw_intensity = np.clip(base_mag + random_noise, 0.3, 1.0)
       
        speed_ratio = 80.0 / max(current_speed_kmh, 10.0)
       
        self.intensity_factor = np.clip(raw_intensity * speed_ratio, 0.2, 1.0)

    def get_action(self):
        self.step_counter += 1
        steer = 0.0
        gas = 0.0
        brake = 0.0
        is_finished = False

        # 1. Feint Motion
        if self.step_counter <= self.feint_duration:
            steer = self.direction * self.intensity_factor * 0.4
            gas = 0.0
           
        # 2. Reverse Counter
        elif self.step_counter <= (self.feint_duration + self.counter_duration):
            # Opposite side * weight
            steer = self.direction * (-1.0) * self.intensity_factor * self.counter_weight
           
            steer = np.clip(steer, -1.0, 1.0)
            gas = 1.0
           
        # 3. Finalize
        else:
            is_finished = True
            steer = 0.0
       
        return np.array([steer, gas, brake]), is_finished

# it should be noted that the environment already checks that you are within 3m from the reference line and on the track otherwise it isn't a succesfull recovery
def check_success_criteria(info, state, target_speed_kmh):
    # Slip recovery decision from environment
    slip_recovered = info.get('slip_recovered', False)
    #print(f"Slip Recovered: {slip_recovered}")
   
    gap = abs(state.get('gap', 100.0))
    #is_on_track = gap < 3.0  # 2m

    current_speed = state.get('speed', 0) * 3.6
    is_speed_ok = current_speed > 30.0  # Above 30 km/h
  
    is_success = slip_recovered and is_speed_ok
   
    return is_success, {"gap": gap, "speed": current_speed}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", type=str)
    parser.add_argument("--randomize_speed", action="store_true", help="Enable target speed randomization", default=False)
    parser.add_argument("--randomize_steer", action="store_true", help="Enable destabilize steer randomization", default=False)
    parser.add_argument("--base_target_speed", type=float, default=80.0)
    parser.add_argument("--base_steer_mag", type=float, default=0.8)
    parser.add_argument("--num_episodes", type=int, default=10000)
    #parser.add_argument("--load_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo", type=str, default="td3")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # Set a path like this to load a path. If you uncomment this line it WILL be used.
    #args.load_path = r"C:\Users\22ave\Desktop\BeyondGrip-Physics-Informed-Control-Recovery-for-Racecars\outputs_recovery\monza\20251215_151914\checkpoint_32530"
   
    # --- Config Setup ---
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join("outputs_recovery", f"{config.AssettoCorsa.track}", timestamp)
    os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir
   
    # --- Logging Setup ---
    logging_config.create_logging(level=logging.INFO, file_name=os.path.join(work_dir, "train_recovery.log"))
    logger.info("=== Starting Recovery Training ===")
   
    # --- WandB Setup ---
    config.exp_name = f'Recovery-{config.AssettoCorsa.track}'
    config.action_dim = 3  # steer, gas, brake
    config.steps = config.Agent.num_steps
   
    wandb_logger = None
    if not config.get("disable_wandb", False):
        try:
            wandb_logger = Logger(config.copy())
            logger.info("WandB initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to init WandB: {e}")

    # --- Environment Setup ---
    env = assettoCorsa.make_ac_env(
        cfg=config,
        work_dir=work_dir,
        env_class=RecoveryAssettoEnv,
        env_kwargs=dict(
            slip_threshold=5,
            recovery_time=1.5
        )
    )
   
    # --- Agent Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # SAC parameters from config
    sac_params = config.get("SAC", {})
   
    try:
        policy_hidden = OmegaConf.to_container(sac_params.get("policy_hidden_units", [256, 256]), resolve=True)
    except:
        policy_hidden = list(sac_params.get("policy_hidden_units", [256, 256]))
   
    try:
        q_hidden = OmegaConf.to_container(sac_params.get("q_hidden_units", [256, 256]), resolve=True)
    except:
        q_hidden = list(sac_params.get("q_hidden_units", [256, 256]))

    if not isinstance(policy_hidden, list):
        policy_hidden = list(policy_hidden)
    if not isinstance(q_hidden, list):
        q_hidden = list(q_hidden)

    algo = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        seed=args.seed,
        policy_hidden_units=policy_hidden,
        q_hidden_units=q_hidden
    )

    # override and choose algo based on CLI
    if args.algo == 'td3':
        algo = TD3(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            wandb_logger=wandb_logger,
            **OmegaConf.to_container(config.TD3))
    elif args.algo == 'sac':
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC))
   
    agent_config = dict(config.Agent)
    agent_config['update_interval'] = agent_config.get('update_interval', 1)
    agent_config['start_steps'] = agent_config.get('start_steps', 2000)
   
    agent = Agent(
        env=env,
        test_env=env,
        algo=algo,
        log_dir=work_dir,
        device=device,
        seed=args.seed,
        **agent_config,
        wandb_logger=wandb_logger,

    )

    if args.load_path is not None:
        load_buffer = True
        agent.load(args.load_path, load_buffer=load_buffer)

    destabilizer = Destabilizer(feint_duration=6, counter_duration=12)


    logger.info(">>> Recovery Training Loop Started")
    logger.info("=" * 60)
    logger.info(f"Training Configuration:")
    logger.info(f"  Base Target Speed: {args.base_target_speed} km/h")
    logger.info(f"  Base Steer Magnitude: {args.base_steer_mag}")
    logger.info(f"  Randomize Speed: {args.randomize_speed}")
    logger.info(f"  Randomize Steer: {args.randomize_steer}")
    logger.info(f"  Total Episodes: {args.num_episodes}")
    logger.info("=" * 60)
   
    total_steps = 0
    SUCCESS_WINDOW = 300 # moving average window
    episode_success_buffer = deque(maxlen=SUCCESS_WINDOW)
   



    print(f"nstep  {agent._algo.nstep}")

    # Leave this as false, I was not able to get meaningful results with True, this is why we are doing heirchal RL I guess.
    use_relative_actions = False

    bin_idx = 0
    for episode in range(args.num_episodes):
       
        target_speed = args.base_target_speed
        if args.randomize_speed:
            bins = [(60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)]
            low, high = bins[bin_idx]
            target_speed = np.random.uniform(low, high)
            bin_idx = (bin_idx + 1) % len(bins)
        target_speed = 80
        steer_mag = args.base_steer_mag
        if args.randomize_steer:
            steer_mag += np.random.normal(0, 0.05)
            steer_mag = np.clip(steer_mag, 0.5, 1.0)

        # MPC Controller Re-initialization
        if MPC_controller:
            mpc = MPC_controller(config, env.norm_steer_at_max, target_speed)
        else:
            class MockMPC:
                def get_action(self, obs): return np.array([0.0, 1.0, 0.0])
            mpc = MockMPC()

        # Environment & Destabilizer Reset
        try:
            env.restart_episode()
        except AttributeError:
            pass
           
        obs = env.reset()
        # reset randomizes the intensity already
        destabilizer.reset(current_speed_kmh=(env.state['speed'] * 3.6), base_mag=steer_mag, random_steer=args.randomize_steer)

        destabilizer.counter_weight = 3.5
        destabilizer.counter_weight += np.random.normal(0, 2)
        destabilizer.counter_weight = trunc_norm(2.5, 0.3, 1.5, 3.5)

        destabilizer.feint_duration = trunc_norm(6, 1, 4, 8)

        destabilizer.counter_duration = trunc_norm(14, 6, 8, 25)
        # Episode Loop Variables
        done = False
        phase = "APPROACH"
       
        ep_reward = 0
        ep_steps = 0
        is_success = False
        recovery_steps = 0
       
        # Track reward components for logging
        phase_rewards = {"APPROACH": 0.0, "DESTABILIZE": 0.0, "RECOVERY": 0.0}
       
        logger.info(f"EP {episode} | Target Speed: {target_speed:.1f} km/h")

        while not done:
            time_info={"episode": episode,
                        "recovery_steps": recovery_steps,
                        "global_step": total_steps,
                        "step": total_steps,
            }
            current_speed_kmh = env.state['speed'] * 3.6
            # By Default Do no allow episode termination due to slip recovery
            env.slip_recovery_mode = False # this line is not really necessary because the flag is to False on env.reset()
            #env.use_relative_actions = True
            env.use_relative_actions = True


            if phase == "APPROACH":
                action = mpc.get_action(env.state) # MPC
               
                if abs(current_speed_kmh - target_speed) < 5.0:
                    phase = "DESTABILIZE"
                    destabilizer.reset(current_speed_kmh=current_speed_kmh, base_mag=steer_mag, random_steer=args.randomize_steer)

            elif phase == "DESTABILIZE":
                action, is_finished = destabilizer.get_action()
                if is_finished:
                    phase = "RECOVERY"

            elif phase == "RECOVERY":
                env.slip_recovery_mode = True

                env.use_relative_actions = use_relative_actions
                if total_steps < agent._start_steps:
                    action = env.action_space.sample()
                else:
                    action, _ = agent._algo.explore(obs)


                    # Need to explore more. (if using relative actions)
                    # if total_steps < 10000:
                    #     action += np.random.normal(0, 0.25, size=action.shape)
                    #     action = np.clip(action, -1, 1)
           
            next_obs, reward, done, info = env.step(action)
           
            if phase != "RECOVERY":
                current_speed = env.state['speed'] * 3.6
               
                if done and current_speed < 3.6:  # Below ~1 m/s
                    logger.warning(f"Episode stopped in {phase} phase: Speed too low ({current_speed:.1f} km/h)")
                    break
               
                # Check if completely off-track (max_gap exceeded)
                if done and info.get('terminated', False):
                    gap = abs(env.state.get('gap', 0))
                    if gap > 10.0:
                        logger.warning(f"Episode stopped in {phase} phase: Off-track (gap: {gap:.2f}m)")
                        break


                done = False
                reward = 0.0
            else:
                # RECOVERY phase: check success and log slip_recovered status

                stats = {
                    "debug/done_from_env": done,
                    "telemetry/speed_kmh": env.state["speed"] * 3.6,
                    "telemetry/gap": env.state["gap"],

                    "telemetry/max_rear_slip_angle_deg": max(
                        abs(env.state["SlipAngle_rl"]),
                        abs(env.state["SlipAngle_rr"]),
                    ),

                    "telemetry/min_rear_Dy": min(
                        env.state["Dy_rl"],
                        env.state["Dy_rr"],
                    ),

                    "telemetry/max_rear_slip_ratio": max(
                        abs(env.state["tyre_slip_ratio_rl"]),
                        abs(env.state["tyre_slip_ratio_rr"]),
                    ),

                    "telemetry/steerAngle_deg": env.state["steerAngle"],
                    "telemetry/steerAction": env.state["steerAngle"] / 360,
                    "telemetry/throttle": env.state["accStatus"],
                    "telemetry/brake": env.state["brakeStatus"],

                    "telemetry/accel_x": env.state["accelX"],
                    "telemetry/accel_y": env.state["accelY"],
                    "telemetry/yaw_rate": env.state["angular_velocity_z"],


                    "telemetry/wheel_speed_rear_diff": abs(
                        env.state["wheel_speed_rl"] - env.state["wheel_speed_rr"]
                    ),
                    "telemetry/gear": env.state["actualGear"],
                    "telemetry/vel_x": env.state["local_velocity_x"],
                    "telemetry/vel_y": env.state["local_velocity_y"],

                    "per_step_reward": reward,
                }
                stats |= time_info
                wandb_logger.log(stats)


                slip_recovered = info.get('slip_recovered', False)
                is_success, metrics = check_success_criteria(info, env.state, target_speed)

                if recovery_steps % 12 == 0:
                    logger.info(f"Step {ep_steps} | Slip Angle: {env.state['SlipAngle_rl']} Phase: RECOVERY | SlipRecovered: {slip_recovered} | Gap: {metrics['gap']:.2f}m | Speed: {metrics['speed']:.1f}km/h")
               
                # Augmentation of terminal rewards is not necessary this is handled by the env and automatically routes the correct rewards
                # only success if the car is <3m from referrence line.
                if is_success:
                    #reward += 10.0
                    done = True
                    logger.info(f"SUCCESS! Gap: {metrics['gap']:.2f}, Speed: {metrics['speed']:.1f}")


                if env.state['done'] and not is_success:
                    done = True
                    logger.info(f"FAILED. Normal Termination.")

                    #reward -= 5.0
                
                timeout = (ep_steps >= env._max_episode_steps)
                terminated = done and not env.state["slip_recovered"] and not timeout

                agent._replay_buffer.append(obs, action, reward, next_obs, terminated, done)


               
                if total_steps >= agent._start_steps:
                    if total_steps < 30_000:
                        updates = 1
                    # # elif total_steps < 35_000:
                    # #     updates = 2
                    # else:
                    #     updates = 3
                    updates = 2
                    for _ in range(updates):
                        stats = agent.update_model()
                        if stats == None:
                            continue
                        stats = stats | time_info
                        
                        wandb_logger.log(stats)
               
                ep_reward += reward
                phase_rewards["RECOVERY"] += reward
                total_steps += 1
                recovery_steps += 1

            obs = next_obs
            ep_steps += 1
           
            if ep_steps > 500:
                # timeout should get a terminal reward of failure
                info["slip_recovered"] = False
                env.state["done"] = 1
                reward += env.terminal_reward(env.state, info)
                done = True
                logger.info(f"FAILED. Timeout.")



        # --- Episode End Logging ---
        logger.info(f"Ep {episode} End | Phase: {phase} | Reward: {ep_reward:.2f} | Steps: {ep_steps} | Recovery Steps: {recovery_steps} | Success: {is_success}")
       
        episode_success_buffer.append(is_success)
        if len(episode_success_buffer) > 0:
            rolling_success = sum(episode_success_buffer)/len(episode_success_buffer)
        else:
            rolling_success = 0
        
        # Don't worry about this, I was testing using mor advanced replay buffers
        # if info.get("slip_recovered", False):
        #     for (s, a, r, ns, d) in episode_transitions:
        #         agent._replay_buffer.append(s, a, r, ns, d, is_elite=True)


        # WandB logging
        if wandb_logger:
            log_dict = {
                "success/rate": rolling_success,
                "success/window_size": len(episode_success_buffer),

                "episode": episode,
                "episode_reward": ep_reward,
                "episode_steps": ep_steps,
                "recovery_steps": recovery_steps,
                "global_step": total_steps,
                "step": total_steps,
               
                "phase/final_phase": phase,
                "phase/approach_reward": phase_rewards["APPROACH"],
                "phase/destabilize_reward": phase_rewards["DESTABILIZE"],
                "phase/recovery_reward": phase_rewards["RECOVERY"],
               
                "success/is_success": float(is_success),
                "success/ended_in_recovery": int(phase == "RECOVERY"),
               
                "config/target_speed": target_speed,
                "config/bin_idx": bin_idx,
                "config/steer_magnitude": steer_mag,

                "config/feint_duration": destabilizer.feint_duration,
                "config/counter_duration" : destabilizer.counter_duration,
                "config/intensity_factor":destabilizer.intensity_factor,
                "config/counter_weight":destabilizer.counter_weight,
            }
           
            if phase == "RECOVERY":
                slip_recovered = info.get('slip_recovered', False)
                log_dict.update({
                    "final/slip_recovered": float(slip_recovered),
                    "final/gap": abs(env.state.get('gap', 100.0)),
                    "final/speed_kmh": env.state.get('speed', 0) * 3.6,
                })
           
            wandb_logger.log(log_dict)
           
        if episode % 10 == 0:
            agent.save(os.path.join(work_dir, "model_latest"))
           
        if config.Agent.get('checkpoint_freq', 0) > 0 and total_steps % config.Agent.checkpoint_freq < recovery_steps:
            checkpoint_path = os.path.join(work_dir, f"checkpoint_{total_steps}")
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved at step {total_steps}: {checkpoint_path}")

    agent.save(os.path.join(work_dir, "model_final"))
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()