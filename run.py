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

# Add paths
sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])

# Custom module imports
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.algorithm import SAC, DisCor
from discor.agent import Agent
import common.misc as misc
import common.logging_config as logging_config

# Set Logger 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScenarioRunner")

# ==========================================
# 1. Scenario Supervisor (State Machine)
# ==========================================
class DrivingState(Enum):
    APPROACH = 0      # Normal Driving (Pre-trained Agent)
    DESTABILIZE = 1   # Induce Slip (Heuristic)
    RECOVERY = 2      # Control Recovery (Custom Controller)

class ScenarioSupervisor:
    def __init__(self, normal_agent, env, target_speed_kmh=100.0):
        self.state = DrivingState.APPROACH
        self.normal_agent = normal_agent
        self.env = env  # Direct access to environment for telemetry
        self.target_speed_ms = target_speed_kmh / 3.6 # Simulation uses m/s

        # Set Destabilize
        self.destabilize_step_count = 0
        self.max_destabilize_steps = 20
        
        self.vehicle_state = None
        
        # Vehicle state history for logging and slip analysis
        self.state_history = []
        self.step_counter = 0

    def get_action(self, obs, info):
        """
        Decide action based on current state and observation.
        Transition states as needed.
        """
        # Get vehicle telemetry directly from env.state (updated via UDP)
        self.vehicle_state = self.env.state if hasattr(self.env, 'state') else {}
        
        # Record state history with timestamp and state label
        self._record_state()
        
        # # print vehicle current status
        # logger.info(f"SPEED: {vehicle_state.get('speed'):.2f} m/s")

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

            # Use existing Agent's policy
            action, _ = self.normal_agent._algo.exploit(obs)
            return action

        # [Step 2] Destabilize: Induce Slip
        elif self.state == DrivingState.DESTABILIZE:
            # Change to Recovery state after timesteps
            # TODO
            # Add Slip conditions such as <abs(info['slip_angle']) > threshold>
            if self.destabilize_step_count >= self.max_destabilize_steps:
                logger.info(">>> [TRANSITION] Destabilize Done! -> Switch to RECOVERY")
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
        
        # [Scandinavian Flick / Pendulum Turn Strategy]
        # Phase 1: Feint (move right to shift weight)
        # Phase 2: Initiation (move left while applying full throttle to induce rear slip)

        feint_duration = 7  # faint duration
        
        # Phase 1: Feint
        if self.destabilize_step_count <= feint_duration:
            steer = +0.3 # Move right
            gas = 0.0
            brake = 0.0
            
            if self.destabilize_step_count == 1:
                logger.info(">>> [DESTABILIZE] Phase 1: Feint (Turn Right, Lift-off)")
                
            return np.array([steer, gas, brake])
            
        # Phase 2: Initiation
        else:
            steer = -0.5   # Left full lock
            gas = 1.0     # Full gas (induce rear slip)
            brake = 0.0
            
            if self.destabilize_step_count == feint_duration + 1:
                logger.info(">>> [DESTABILIZE] Phase 2: Flick & Power (Turn Left + Full Gas)")
            
            # Log telemetry periodically during flick phase
            if (self.destabilize_step_count - feint_duration) % 5 == 0:
                logger.info(f"[DESTABILIZE {self.destabilize_step_count}/{self.max_destabilize_steps}] "
                           f"Speed: {self.vehicle_state.get('speed', 0)*3.6:.1f} km/h, "
                           f"Slip: {self.vehicle_state.get('slip_angle', 0):.2f}°, "
                           f"Steer: {self.vehicle_state.get('steerAngle', 0):.2f}")
                
            return np.array([steer, gas, brake])

    def _get_recovery_action(self, obs):
        # Get vehicle telemetry directly from env
        self.vehicle_state = self.env.state if hasattr(self.env, 'state') else {}
        
        # [Temporary] Use pre-trained agent for recovery to test the system
        # This allows us to see how the trained agent handles slip recovery
        # Later, we'll replace this with our custom slip recovery controller
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
        
        # return action
        return np.array([0.0, 0.0, 0.0])
        
        # TODO: Replace with custom slip recovery controller
        # Custom controller logic will go here:
        # speed = self.vehicle_state.get('speed', 0.0)
        # slip_angle = self.vehicle_state.get('slip_angle', 0.0)
        # steer_angle = self.vehicle_state.get('steerAngle', 0.0)
        # yaw_rate = self.vehicle_state.get('yaw_rate', 0.0)
        # return np.array([custom_steer, custom_gas, custom_brake])
    
    def _record_state(self):
        """Record current vehicle state for slip analysis"""
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
        """Save vehicle state history to CSV for slip analysis"""
        if self.state_history:
            df = pd.DataFrame(self.state_history)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved vehicle state history ({len(self.state_history)} samples) to {filepath}")
            return filepath
        else:
            logger.warning("No state history to save")
            return None

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

    # Initialize Supervisor
    supervisor = ScenarioSupervisor(normal_agent=agent, 
                                   env=env,
                                   target_speed_kmh=100.0)

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