# python train_recovery.py
import os
import sys
import argparse
import logging
import numpy as np
import torch
from datetime import datetime
from omegaconf import OmegaConf

sys.path.append(os.path.abspath('./assetto_corsa_gym'))
sys.path.append(os.path.abspath('./algorithm/discor'))

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from assetto_corsa_gym.AssettoCorsaEnv.recovery_ac_env import RecoveryAssettoEnv
# from assetto_corsa_gym.AssettoCorsaEnv.recovery_phys_ac_env import PhysicsRecoveryEnv

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
    
    def safe_reset(self):
        self.step_counter = 0

    def reset(self, current_speed_kmh, base_mag, random_steer):
        self.step_counter = 0
       
        # 1. Random Direction
        if random_steer:
            self.direction = 1.0 if np.random.random() > 0.5 else -1.0
        else:
            self.direction = -1.0


        # 2. Random Intensity
        random_noise = np.random.normal(0, 0.05)

        low_intensity = 0.3
        high_intensity = 1.0
        raw_intensity = trunc_norm(base_mag, 0.1, low_intensity, high_intensity)
        #raw_intensity = np.clip(base_mag + random_noise, 0.3, 1.0)
       
        # 3. Speed Scaling
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
   
    # is_success = slip_recovered and is_on_track and is_speed_ok
    #is_success = slip_recovered and is_speed_ok and is_on_track
    is_success = slip_recovered and is_speed_ok
   
    return is_success, {"gap": gap, "speed": current_speed}



class Evaluator():
    def __init__(self, config, env, agent, destabilizer, wandb_logger):
        self.config = config
        self.env = env
        self.agent = agent
        self.destabilizer = destabilizer
        self.logger = wandb_logger
        self.episode = 0
        self.direction = -1
        self.obs_idx = {
            name: i for i, name in enumerate(self.env.obs_enabled_channels)
        }
        self.yaw_rate_idx = self.obs_idx['angular_velocity_y']
    

    def real_evaluate(self, config, env, agent, episode, destabilizer, wandb_logger,
                      slip_param, test_class):
        
        rng = np.random.default_rng(seed=episode)
        feint_duration = slip_param["feint_duration"]
        counter_duration = slip_param["counter_duration"]
        direction = self.direction
        intensity_factor = slip_param["intensity_factor"]
        counter_weight = slip_param["counter_weight"]
        target_speed = slip_param["target_speed"]

        destabilizer.safe_reset()

        destabilizer.direction = direction
        destabilizer.feint_duration = feint_duration
        destabilizer.counter_duration = counter_duration
        destabilizer.intensity_factor = intensity_factor
        destabilizer.counter_weight = counter_weight

        if MPC_controller:
            mpc = MPC_controller(config, env.norm_steer_at_max, target_speed)
        else:
            class MockMPC:
                def get_action(self, obs): return np.array([0.0, 1.0, 0.0])
            mpc = MockMPC()

        env.restart_episode()
        obs = env.reset()

        # Episode Loop Variables
        done = False
        phase = "APPROACH"
        
        ep_reward = 0
        ep_steps = 0
        is_success = False
        recovery_steps = 0


        yaw_rate_scale = 0
        rng = np.random.default_rng(seed=episode)
        yaw_rate_scale = rng.uniform(0.9, 1.1)
        
        while not done:
            alpha = 0
            time_info={"episode": episode,
                        "recovery_steps": recovery_steps,
                        "step": recovery_steps,
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
                    destabilizer.reset(current_speed_kmh=current_speed_kmh, base_mag=0.8)

            elif phase == "DESTABILIZE":
                action, is_finished = destabilizer.get_action()
                if is_finished:
                    phase = "RECOVERY"

            elif phase == "RECOVERY":
                env.slip_recovery_mode = True
                env.use_relative_actions = False

                obs_used = obs.copy()
                if test_class in ["ROBUST_OBS_YAW", "ROBUST_COMBINED"]:
                    obs_used[self.yaw_rate_idx] *= yaw_rate_scale

                action, entropy = self.agent._algo.exploit(obs_used)

                alpha = 0.7
                if test_class in ["ROBUST_ACTION_CLIP", "ROBUST_COMBINED"]:
                    alpha
                    action[0] = np.clip(action[0], -alpha, alpha)

            next_obs, reward, done, info = env.step(action)
            
            if phase != "RECOVERY":
                current_speed = env.state['speed'] * 3.6
                
                if done and current_speed < 3.6:  # Below ~1 m/s
                    break
                
                # Check if completely off-track (max_gap exceeded)
                if done and info.get('terminated', False):
                    gap = abs(env.state.get('gap', 0))
                    if gap > 10.0:
                        break

                done = False
                reward = 0.0
            else:
                stats = {
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
                    "telemetry/throttle": env.state["accStatus"],
                    "telemetry/brake": env.state["brakeStatus"],

                    "telemetry/accel_x": env.state["accelX"],
                    "telemetry/accel_y": env.state["accelY"],
                    "telemetry/yaw_rate": env.state["angular_velocity_z"],


                    "telemetry/wheel_speed_rear_diff": abs(
                        env.state["wheel_speed_rl"] - env.state["wheel_speed_rr"]
                    ),
                    "telemetry/gear": env.state["actualGear"],
                    "per_step_reward": reward,
                }
                stats |= time_info
                wandb_logger.log(stats)
                slip_recovered = info.get('slip_recovered', False)
                is_success, metrics = check_success_criteria(info, env.state, target_speed)
                
                if is_success:
                    done = True

                if env.state['done'] and not is_success:
                    done = True
                
                ep_reward += reward
                recovery_steps += 1

            obs = next_obs
            ep_steps += 1
            
            if ep_steps > 500:
                info["slip_recovered"] = False
                env.state["done"] = 1
                reward += env.terminal_reward(env.state, info)
                done = True

        # WandB logging
        speed_bins = [(60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)]

        bin_idx = -1
        for i, (low, high) in enumerate(speed_bins):
            if low <= target_speed < high:
                bin_idx = i


        ood_test_classes = {
            "OOD_HI_SPEED",
            "OOD_LO_SPEED",
            "OOD_INTENSITY",
            "OOD_COUNTER",
        }
        is_ood = int(test_class in ood_test_classes)

        result = {
            "test/episode": episode,
            "test/target_speed_kmh": target_speed,
            "test/success": float(is_success),
            "test/recovery_steps": recovery_steps,
            "test/recovery_time_sec": float(recovery_steps / 25),
            "test/final_speed_kmh": env.state["speed"] * 3.6,
            "test/final_gap": abs(env.state.get("gap", 100.0)),
            "test/feint_duration": destabilizer.feint_duration,
            "test/counter_duration": destabilizer.counter_duration,
            "test/intensity_factor": destabilizer.intensity_factor,
            "test/counter_weight": destabilizer.counter_weight,
            "success/is_success": float(is_success),
            "success/ended_in_recovery": int(phase == "RECOVERY"),
            "test/type": test_class,
            "test/yaw_rate_scale": yaw_rate_scale,
            "test/alpha": alpha,
            "test/bin_idx": bin_idx,
            "test/is_ood": is_ood,
        }
        result |= time_info
        wandb_logger.log(result)
        self.episode += 1
    
    def evaluate(self, slip_param, test_class):
        self.real_evaluate(config = self.config,
                           env = self.env,
                           agent = self.agent,
                           episode = self.episode,
                           destabilizer = self.destabilizer,
                           wandb_logger = self.logger,
                           slip_param = slip_param,
                           test_class = test_class,
        )
    
    def test(self):
        slip_param_gen = SlipParamGenerator()
        
        # In Distribution Tests
        tests_per_bin = 50
        for i in range(tests_per_bin * 8):
            slip_param = slip_param_gen.generate()
            self.evaluate(slip_param, test_class = 'ID')
        

        ood_tests = 18
        # OOD SPEED HIGH
        for i in range(ood_tests):
            slip_param = slip_param_gen.generate()
            slip_param["target_speed"] = 110
            self.evaluate(slip_param, test_class='OOD_HI_SPEED')

        # # OOD SPEED LOW
        # for i in range(ood_tests):
        #     slip_param = slip_param_gen.generate()
        #     slip_param["target_speed"] = 50
        #     self.evaluate(slip_param, test_class='OOD_LO_SPEED')
        
        #OOD INTENSITY
        for i in range(ood_tests//3):
            for intensity_factor in [1.1, 1.3, 1.5]:
                slip_param = slip_param_gen.generate()
                slip_param["intensity_factor"] = intensity_factor
                self.evaluate(slip_param, test_class="OOD_INTENSITY")

        #OOD COUNTER
        for i in range(ood_tests//3):
            for counter_weight in [3.6, 3.8, 4.0]:
                slip_param = slip_param_gen.generate()
                slip_param["counter_weight"] = counter_weight
                self.evaluate(slip_param, test_class='OOD_COUNTER')
        

        #ID PERTURB
        for i in range(ood_tests//2):
            slip_param = slip_param_gen.generate()

            slip_param["target_speed"] = 80
            slip_param["feint_duration"] = 6
            slip_param["counter_duration"] = 16
            slip_param["counter_weight"] = 2.5
            slip_param["intensity_factor"] = 1.0

            self.evaluate(
                slip_param,
                test_class="ROBUST_OBS_YAW",
            )

        for i in range(ood_tests//2):
            slip_param = slip_param_gen.generate()

            slip_param["target_speed"] = 80
            slip_param["feint_duration"] = 6
            slip_param["counter_duration"] = 16
            slip_param["counter_weight"] = 2.5
            slip_param["intensity_factor"] = 1.0

            self.evaluate(
                slip_param,
                test_class="ROBUST_ACTION_CLIP",
            )
        
        for i in range(ood_tests//2):
            slip_param = slip_param_gen.generate()

            slip_param["target_speed"] = 80
            slip_param["feint_duration"] = 6
            slip_param["counter_duration"] = 16
            slip_param["counter_weight"] = 2.5
            slip_param["intensity_factor"] = 1.0

            self.evaluate(
                slip_param,
                test_class="ROBUST_COMBINED",
            )

        



class SlipParamGenerator:
    def __init__(self, seed = 100):
        self.rng = np.random.RandomState(seed)
        self.speed_bins = [(60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)]
        self.idx = 0

    def trunc_norm(self, mu, sigma, low, high):
        while True:
            x = mu + sigma * self.rng.randn()
            if low <= x <= high:
                return x

    def generate(self):
        low, high = self.speed_bins[self.idx]
        self.idx = (self.idx + 1) % len(self.speed_bins)
        target_speed = (low + high) / 2.0

        speed_ratio = 80.0 / max(target_speed, 10.0)
        raw_intensity = trunc_norm(0.8, 0.1, 0.3, 1.0)
        
        intensity_factor = np.clip(raw_intensity * speed_ratio, 0.2, 1.0)

        return {
            "target_speed": target_speed,

            "feint_duration": int(
                self.trunc_norm(6, 1, 4, 8)
            ),
            "counter_duration": int(
                self.trunc_norm(14, 4, 8, 25)
            ),
            "counter_weight": float(
                self.trunc_norm(2.5, 0.3, 1.5, 3.5)
            ),
            "intensity_factor": float(
                intensity_factor
            ),
        }



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", type=str)
    parser.add_argument("--algo", type=str, default="td3")
   
    return parser.parse_args()

def main():
    args = parse_args()
   
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

    destabilizer = Destabilizer()
   

        

if __name__ == "__main__":
    main()