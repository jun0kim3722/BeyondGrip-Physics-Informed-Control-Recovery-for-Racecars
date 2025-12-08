# python train_ddpg.py --target_speed 60 --flick_intensity 0.4 \
#     AssettoCorsa.track=monza AssettoCorsa.car=bmw_z4_gt3 \
#     Agent.num_steps=100000
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np
from omegaconf import OmegaConf
import torch
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Add paths
sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])

# Custom module imports
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.algorithm import SAC, DisCor
from discor.agent import Agent
import common.misc as misc
import common.logging_config as logging_config
from mpc_controller import MPC_controller

# Set Logger 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScenarioRunner")

TARGET_SPEED = 120.0 # km/h

# ==========================================
# 1. Scenario Supervisor (State Machine)
# ==========================================
class DrivingState(Enum):
    APPROACH = 0      # Normal Driving (Pre-trained Agent)
    DESTABILIZE = 1   # Induce Slip (Heuristic)
    RECOVERY = 2      # Control Recovery (Custom Controller)

class ScenarioSupervisor:
    def __init__(self, normal_agent, env, mpc, target_speed_kmh=100.0, 
                 flick_intensity=1.0):
        """
        Args:
            normal_agent: Agent for normal driving (APPROACH phase)
            env: Environment
            mpc: MPC controller
            target_speed_kmh: Target speed in km/h
            flick_intensity: Flick intensity 0.0-1.0 (controls slip severity)
        """
        self.state = DrivingState.APPROACH
        self.normal_agent = normal_agent
        self.env = env  # Direct access to environment for telemetry
        self.mpc = mpc
        
        # Training parameters
        self.target_speed_kmh = target_speed_kmh
        self.target_speed_ms = target_speed_kmh / 3.6
        self.flick_intensity = flick_intensity
        self.slip_threshold_rad = 0.20
        self.feint_duration = 7
        self.max_destabilize_steps = 20

        self.min_flick_duration = 7  # Minimum steps to maintain flick before checking slip
        self.slip_confirm_count = 2  # Consecutive slip detections needed
        
        # Set Destabilize
        self.destabilize_step_count = 0
        self.slip_detected = False
        self.consecutive_slip_count = 0  # Counter for consecutive slip detections
        
        self.vehicle_state = None
        
        # Vehicle state history for logging and slip analysis
        self.state_history = []
        self.step_counter = 0

    def get_action(self, obs, info):
        # Get vehicle telemetry directly from env.state (updated via UDP)
        self.vehicle_state = self.env.state if hasattr(self.env, 'state') else {}
        
        # Record state history with timestamp and state label
        self._record_state()

        # Get current speed from vehicle state
        current_speed = self.vehicle_state.get('speed', info.get('speed', 0.0))

        # [Step 1] Approach: Accelerate to Target Speed
        if self.state == DrivingState.APPROACH:
            # Change to Destabilize upon reaching target speed
            if current_speed >= self.target_speed_ms:
                cur_speed_kmh = current_speed * 3.6
                logger.info(f">>> [TRANSITION] Target Speed ({cur_speed_kmh:.1f} km/h) Reached! -> Switch to DESTABILIZE")
                self.state = DrivingState.DESTABILIZE
                return self._get_destabilize_action()

            # Use classic control
            # action, _ = self.normal_agent._algo.exploit(obs)
            action = self.mpc.get_action(self.vehicle_state)
            print(action)
            return action

        # [Step 2] Destabilize: Induce Slip
        elif self.state == DrivingState.DESTABILIZE:
            
            min_steps_before_check = self.feint_duration + self.min_flick_duration
            
            # Only check slip after feint + min flick duration
            if self.destabilize_step_count > min_steps_before_check:
                if self._check_slip_condition(self.vehicle_state):
                    self.consecutive_slip_count += 1
                    if self.consecutive_slip_count >= self.slip_confirm_count:
                        slip_rl = self.vehicle_state.get('SlipAngle_rl', 0.0)
                        slip_rr = self.vehicle_state.get('SlipAngle_rr', 0.0)
                        logger.info(f">>> [TRANSITION] Sustained Slip Detected ({self.consecutive_slip_count} consecutive)! "
                                   f"RL: {slip_rl:.3f} rad, RR: {slip_rr:.3f} rad -> Switch to RECOVERY")
                        self.state = DrivingState.RECOVERY
                        self.slip_detected = True
                        return self._get_recovery_action(obs)
                else:
                    # Reset counter if slip not detected
                    if self.consecutive_slip_count > 0:
                        logger.debug(f"Slip counter reset: {self.consecutive_slip_count} -> 0")
                    self.consecutive_slip_count = 0
            
            # Safety fallback: transition after max steps even if slip not confirmed
            if self.destabilize_step_count >= self.max_destabilize_steps:
                logger.info(f">>> [TRANSITION] Max Destabilize Steps Reached "
                           f"(consecutive slip count: {self.consecutive_slip_count}) -> Switch to RECOVERY")
                self.state = DrivingState.RECOVERY
                return self._get_recovery_action(obs)
            
            return self._get_destabilize_action()

        # [Step 3] Recovery: Control Recovery
        elif self.state == DrivingState.RECOVERY:
            return self._get_recovery_action(obs)

        return np.zeros(3)

    def _get_destabilize_action(self):
        self.destabilize_step_count += 1
        
        self.vehicle_state = self.env.state if hasattr(self.env, 'state') else {}
        
        # Action Space: [Steer, Gas, Brake]
        # Steer: -1.0 (Left) ~ 1.0 (Right)
        # Gas: 0.0 ~ 1.0
        # Brake: 0.0 ~ 1.0

        # Phase 1: Feint (NO slip detection during this phase!)
        if self.destabilize_step_count <= self.feint_duration:
            steer = +0.5 * self.flick_intensity  # Move right (scaled by intensity)
            gas = 0.0
            brake = 0.0
            
            if self.destabilize_step_count == 1:
                logger.info(">>> [DESTABILIZE] Phase 1: Feint (Turn Right, Lift-off)")
                logger.info(f"    Feint intensity: {steer:.2f}")
                logger.info(f"    Slip detection will start after step {self.feint_duration}")
                
            return np.array([steer, gas, brake])
            
        # Phase 2: Initiation (slip detection is ACTIVE now)
        else:
            steer = -1.0 * self.flick_intensity  # Scale by curriculum level
            gas = 1.0 * self.flick_intensity     # Gas also scaled
            brake = 0.0
            
            if self.destabilize_step_count == self.feint_duration + 1:
                logger.info(">>> [DESTABILIZE] Phase 2: Flick & Power (Turn Left + Full Gas)")
                logger.info(f"    Flick intensity: {self.flick_intensity:.2f} (Steer: {steer:.2f}, Gas: {gas:.2f})")
                logger.info(f"    Slip detection will start after step {self.feint_duration + self.min_flick_duration}")
            
            # Log telemetry periodically during flick phase
            slip_rl = self.vehicle_state.get('SlipAngle_rl', 0.0)
            slip_rr = self.vehicle_state.get('SlipAngle_rr', 0.0)
            if (self.destabilize_step_count - self.feint_duration) % 3 == 0:
                logger.info(f"[DESTABILIZE Step {self.destabilize_step_count}] "
                           f"Speed: {self.vehicle_state.get('speed', 0)*3.6:.1f} km/h, "
                           f"SlipAngle RL: {slip_rl:.3f} rad, RR: {slip_rr:.3f} rad "
                           f"(threshold: {self.slip_threshold_rad:.3f}, consecutive: {self.consecutive_slip_count})")
                
            return np.array([steer, gas, brake])

    def _get_recovery_action(self, obs):
        self.vehicle_state = self.env.state if hasattr(self.env, 'state') else {}
        
        action, _ = self.normal_agent._algo.exploit(obs)
        
        # Log recovery telemetry periodically
        if hasattr(self, 'recovery_step_count'):
            self.recovery_step_count += 1
        else:
            self.recovery_step_count = 1
            logger.info(">>> [RECOVERY] Using pre-trained agent for recovery (temporary)")
        
        if self.recovery_step_count % 5 == 0:
            slip_rl = self.vehicle_state.get('SlipAngle_rl', 0.0)
            slip_rr = self.vehicle_state.get('SlipAngle_rr', 0.0)
            logger.info(f"[RECOVERY Step {self.recovery_step_count}] "
                       f"Speed: {self.vehicle_state.get('speed', 0)*3.6:.1f} km/h, "
                       f"SlipAngle RL: {slip_rl:.3f} rad, RR: {slip_rr:.3f} rad")
        
        return action
        # return np.array([0.0, 0.0, 0.0])
    
    def _record_state(self):
        if self.vehicle_state:
            self.step_counter += 1
            state_snapshot = {
                'step': self.step_counter,
                'scenario_state': self.state.name,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3],
            }
            # Add all vehicle state data
            state_snapshot.update(self.vehicle_state)
            self.state_history.append(state_snapshot)
    
    def save_state_history(self, filepath):
        if self.state_history:
            df = pd.DataFrame(self.state_history)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved vehicle state history ({len(self.state_history)} samples) to {filepath}")
            return filepath
        else:
            logger.warning("No state history to save")
            return None
    
    def _check_slip_condition(self, vehicle_state):
        # Get rear wheel slip angles (unit: radians)
        slip_rl = vehicle_state.get('SlipAngle_rl', 0.0)
        slip_rr = vehicle_state.get('SlipAngle_rr', 0.0)
        
        # Check if either rear wheel exceeds threshold
        is_slipping = (abs(slip_rl) > self.slip_threshold_rad) or (abs(slip_rr) > self.slip_threshold_rad)
        
        # Optional: Additional check using normalized slip
        # nd_slip = vehicle_state.get('NdSlip', [0, 0, 0, 0])
        # if len(nd_slip) == 4:
        #     is_slipping = is_slipping or (nd_slip[2] > 0.15) or (nd_slip[3] > 0.15)
        
        return is_slipping

# ==========================================
# 2. Main Logic
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Assetto Corsa Scenario Runner")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to config file")
    parser.add_argument("--load_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm type (sac/discor)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Override config values")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)

    # Set working directory (for logging)
    work_dir = "outputs_scenario" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S') + os.sep
    os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir
    
    # Set up logging
    logging_config.create_logging(level=logging.INFO, file_name=work_dir + "scenario.log")
    
    logger.info("=== Starting Scenario Runner ===")
    logger.info(f"Model Load Path: {args.load_path}")

    # Create Environment
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
    env.set_eval_mode() # Set evaluation mode

    # Initialize and load model (Agent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize algorithm
    if args.algo == 'discor':
        algo = DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC), **OmegaConf.to_container(config.DisCor))
    else:
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC))

    # Initialize Agent
    agent = Agent(env=env, test_env=env, algo=algo, log_dir=work_dir,
                  device=device, seed=config.seed, **config.Agent)

    # Load checkpoint
    agent.load(args.load_path, load_buffer=False)
    logger.info("Checkpoints loaded successfully.")

    mpc = MPC_controller(config, env.norm_steer_at_max, TARGET_SPEED)

    # Initialize Supervisor
    supervisor = ScenarioSupervisor(normal_agent=agent, 
                                   env=env,
                                   mpc=mpc,
                                   target_speed_kmh=TARGET_SPEED)

    # 5. Run simulation loop
    obs = env.reset()
    done = False

    # Initialize info before the first step
    info = {'speed': 0.0}

    logger.info(">>> Simulation Loop Started")

    try:
        while not done:
            # (1) Get action from Supervisor (uses env.state for telemetry)
            action = supervisor.get_action(obs, info)

            # (2) Step simulation (UDP communication happens here, env.state gets updated)
            next_obs, reward, done, info = env.step(action)

            # (3) Update observation
            obs = next_obs

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        # Save vehicle state history to CSV
        state_csv_path = os.path.join(work_dir, 'vehicle_state_history.csv')
        supervisor.save_state_history(state_csv_path)
        
        env.close()
        logger.info("Simulation finished.")

if __name__ == "__main__":
    main()