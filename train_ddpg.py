# python train_ddpg.py --target_speed 60 --flick_intensity 0.4 \
#     AssettoCorsa.track=monza AssettoCorsa.car=bmw_z4_gt3 \
#     Agent.num_steps=100000
import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf
import torch
import math

# ----------------------------------------------------------------
# 1. Path & Import Setup
# ----------------------------------------------------------------
sys.path.append(os.path.abspath('./assetto_corsa_gym'))
sys.path.append(os.path.abspath('./algorithm/discor'))

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
import common.logging_config as logging_config
from common.logger import Logger  # WandB Logger

try:
    from discor.agent import Agent
    from discor.algorithm.ddpg import DDPG  
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

from mpc_controller import MPC_controller
from test_slip_w_mpc import DrivingState, ScenarioSupervisor

logger = logging.getLogger("TrainDDPG")

# ----------------------------------------------------------------
# Dense Recovery Reward Function (Hybrid Approach)
# ----------------------------------------------------------------
def calculate_dense_recovery_reward(state, info, action, prev_action=None, 
                                     target_speed_kmh=100.0, verbose=False):
    """
    Dense reward for slip recovery (every step).
    
    Components:
    1. Survival Bonus: +0.1 (living is good, prevents suicide agent)
    2. Slip Suppression: Reduce rear wheel slip angle
    3. Speed Maintenance: Keep moving (prevent full brake)
    4. Counter-Steer Alignment: Steer opposite to slip direction
    5. Action Smoothness: Prevent jerky control
    6. Out-of-Track Penalty: Heavy penalty for leaving track
    
    Args:
        state: Current vehicle state dict
        info: Info dict with slip angles, speed, etc.
        action: Current action [steer, gas, brake]
        prev_action: Previous action for smoothness calculation
        target_speed_kmh: Target speed in km/h
        verbose: Log detailed breakdown
    
    Returns:
        total_reward: Scalar reward
        components: Dict with breakdown for logging
    """
    
    # Weight configuration
    w_survival = 1.0
    w_slip = 2.0        # Highest priority: suppress slip
    w_speed = 0.5       # Prevent stopping completely
    w_align = 1.0       # Reward correct counter-steer direction
    w_action = 0.3      # Smoothness
    
    # ============================================================
    # Component 1: Survival Bonus
    # ============================================================
    r_survival = 0.1
    
    # ============================================================
    # Component 2: Slip Suppression (CRITICAL)
    # ============================================================
    slip_rl = abs(info.get('SlipAngle_rl', 0.0))
    slip_rr = abs(info.get('SlipAngle_rr', 0.0))
    max_slip = max(slip_rl, slip_rr)
    
    # Normalize: 0.3 rad (~17 deg) is severe slip
    slip_threshold = 0.3
    r_slip = 1.0 - min(max_slip / slip_threshold, 1.0)
    
    # Bonus for getting below safe threshold
    safe_slip_threshold = 0.10  # ~5.7 degrees
    if max_slip < safe_slip_threshold:
        r_slip += 0.5  # Extra bonus for recovery
    
    # ============================================================
    # Component 3: Speed Maintenance
    # ============================================================
    speed_kmh = info.get('speed', 0.0) * 3.6
    target_speed_ms = target_speed_kmh / 3.6
    
    # Normalize speed (allow some reduction during recovery)
    speed_ratio = min(speed_kmh / target_speed_kmh, 1.0)
    
    # Penalty if too slow (prevents full brake strategy)
    if speed_kmh < target_speed_kmh * 0.3:  # Below 30% of target
        r_speed = -0.5
    else:
        r_speed = speed_ratio
    
    # ============================================================
    # Component 4: Counter-Steer Alignment
    # ============================================================
    # Get vehicle body slip (lateral velocity / longitudinal velocity)
    local_vel_x = state.get('local_velocity_x', 0.0)  # Lateral
    local_vel_y = state.get('local_velocity_y', 0.0)  # Longitudinal
    
    # Body slip angle (different from wheel slip)
    body_slip = np.arctan2(local_vel_x, local_vel_y + 1e-6)
    
    # Steering direction (normalized -1 to 1)
    steer = action[0]
    
    # Correct counter-steer: if sliding right (+slip), steer left (-steer)
    # This gives positive reward when steering opposite to slip
    alignment = -np.sign(body_slip) * steer
    r_align = max(alignment, 0.0)  # Only reward correct direction
    
    # ============================================================
    # Component 5: Action Smoothness
    # ============================================================
    if prev_action is not None:
        action_diff = np.linalg.norm(action - prev_action)
        r_action = -action_diff  # Penalty for large changes
    else:
        r_action = 0.0
    
    # ============================================================
    # Component 6: Out-of-Track Penalty
    # ============================================================
    out_of_track = state.get('out_of_track', False)
    r_oot = -2.0 if out_of_track else 0.0
    
    # ============================================================
    # Total Reward
    # ============================================================
    total_reward = (
        w_survival * r_survival +
        w_slip * r_slip +
        w_speed * r_speed +
        w_align * r_align +
        w_action * r_action +
        r_oot
    )
    
    # Component breakdown for logging
    components = {
        'r_survival': r_survival,
        'r_slip': r_slip,
        'r_speed': r_speed,
        'r_align': r_align,
        'r_action': r_action,
        'r_oot': r_oot,
        'total': total_reward,
        'max_slip': max_slip,
        'speed_kmh': speed_kmh,
        'body_slip': body_slip,
    }
    
    if verbose:
        logger.info(f"Dense Reward: {total_reward:.3f} | "
                   f"Slip: {r_slip:.2f} (max={max_slip:.3f}rad) | "
                   f"Speed: {r_speed:.2f} ({speed_kmh:.1f}km/h) | "
                   f"Align: {r_align:.2f} | Survival: {r_survival:.2f}")
    
    return total_reward, components


# ----------------------------------------------------------------
# Softmin Terminal Reward Function
# ----------------------------------------------------------------
def calculate_softmin_reward(state, info, track_info, beta=2.0):
    current_v = info.get('speed', 0.0) 
    current_heading = info.get('heading', 0.0)

    curvature = track_info.get('curvature', 0.0)
    ref_heading = track_info.get('ref_heading', 0.0)

    friction_mu = 1.0
    gravity = 9.81
    
    if abs(curvature) < 1e-4:
        v_max = 300.0 / 3.6 
    else:
        v_max = np.sqrt((friction_mu * gravity) / (abs(curvature) + 1e-8))

    lambda_speed = 1.0
    r1 = lambda_speed * current_v
    
    k_safety = 5.0
    diff_v = max(0, current_v - v_max)
    r2 = -k_safety * (diff_v ** 2)
    
    w_heading = 10.0
    heading_error = current_heading - ref_heading
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    r3 = -w_heading * (heading_error ** 2)
    
    terms = np.array([-beta * r1, -beta * r2, -beta * r3])
    max_val = np.max(terms)
    sum_exp = np.sum(np.exp(terms - max_val))
    softmin_reward = - (1.0 / beta) * (np.log(sum_exp) + max_val)
    
    return softmin_reward


# ----------------------------------------------------------------
# 3. Main Training Script
# ----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", type=str)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--target_speed", type=float, default=60.0,
                       help="Target speed in km/h (default: 60)")
    parser.add_argument("--flick_intensity", type=float, default=0.4,
                       help="Flick intensity 0.0-1.0 (default: 0.4)")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Config Setup ---
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join("outputs_ddpg_slip", f"{config.AssettoCorsa.track}", timestamp)
    os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir
    
    logging_config.create_logging(level=logging.INFO, file_name=os.path.join(work_dir, "train_ddpg.log"))
    logger.info("=== Starting DDPG Slip Recovery Training ===")

    # --- Environment Setup ---
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
    raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    has_track_info = False
    if hasattr(raw_env, 'track') and hasattr(raw_env.track, 'curvature'):
        logger.info("[Check] Track object ready.")
        has_track_info = True
    else:
        logger.warning("[Error] Track object incomplete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- WandB Setup ---
    config.exp_name = f'DDPG-Recovery-{config.AssettoCorsa.track}'
    config.action_dim = env.action_space.shape[0]
    config.steps = config.Agent.num_steps
    
    wandb_logger = None
    if not config.get("disable_wandb", False):
        try:
            wandb_logger = Logger(config.copy())
            logger.info("WandB initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to init WandB: {e}")

    # --- DDPG Setup ---
    ddpg_params = config.get("DDPG", config.get("SAC", {}))
    
    try: policy_hidden = OmegaConf.to_container(ddpg_params.get("policy_hidden_units", [256, 256]), resolve=True)
    except: policy_hidden = list(ddpg_params.get("policy_hidden_units", [256, 256]))
    
    try: q_hidden = OmegaConf.to_container(ddpg_params.get("q_hidden_units", [256, 256]), resolve=True)
    except: q_hidden = list(ddpg_params.get("q_hidden_units", [256, 256]))

    if not isinstance(policy_hidden, list): policy_hidden = list(policy_hidden)
    if not isinstance(q_hidden, list): q_hidden = list(q_hidden)

    algo = DDPG(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device, seed=config.seed,
        policy_hidden_units=policy_hidden, q_hidden_units=q_hidden,
        policy_lr=ddpg_params.get("policy_lr", 3e-4), q_lr=ddpg_params.get("q_lr", 3e-4),
        exploration_noise=0.1
    )

    agent = Agent(env=env, test_env=env, algo=algo, log_dir=work_dir,
                  device=device, seed=config.seed, **config.Agent, 
                  wandb_logger=wandb_logger)

    # --- MPC & Supervisor Setup ---
    target_speed_kmh = args.target_speed
    flick_intensity = args.flick_intensity
    
    logger.info("=" * 60)
    logger.info(f"Training Configuration:")
    logger.info(f"  Target Speed: {target_speed_kmh} km/h")
    logger.info(f"  Flick Intensity: {flick_intensity}")
    logger.info("=" * 60)
    
    mpc = MPC_controller(config, env.norm_steer_at_max, target_speed_kmh)

    # Create supervisor
    supervisor = ScenarioSupervisor(
        normal_agent=agent, env=env, mpc=mpc, 
        target_speed_kmh=target_speed_kmh,
        flick_intensity=flick_intensity
    )

    # --- Training Loop ---
    MAX_STEPS = config.Agent.num_steps
    steps_done = 0
    episode_idx = 0
    
    logger.info(">>> Training Loop Started")

    while steps_done < MAX_STEPS:
        obs = env.reset()
        done = False
        
        # Reset MPC & Supervisor
        supervisor.state = DrivingState.APPROACH
        supervisor.destabilize_step_count = 0
        supervisor.consecutive_slip_count = 0
        supervisor.slip_detected = False
        
        if hasattr(mpc, 'reset'):
            mpc.reset()
        elif hasattr(mpc, 'mpc') and hasattr(mpc.mpc, 'reset_history'):
             mpc.mpc.reset_history()
        
        episode_reward = 0
        recovery_steps = 0
        terminal_val = 0.0
        
        info = {'speed': 0.0, 'heading': 0.0}
        stabilize_steps = 0
        
        # Dense reward tracking
        prev_action = None
        reward_components_history = []

        while not done:
            # Wait for physics stabilization
            current_speed_kmh = info.get('speed', 0.0) * 3.6
            if stabilize_steps < 10 and current_speed_kmh > 5.0:
                 next_obs, env_reward, done, info = env.step(np.array([0.0, -1.0, 0.0]))
                 obs = next_obs
                 stabilize_steps += 1
                 continue

            # --- Control Phase ---
            if supervisor.state != DrivingState.RECOVERY:
                # MPC Phase
                action = supervisor.get_action(obs, info)
                next_obs, env_reward, done, info = env.step(action)
                obs = next_obs
                
            else:
                # DDPG Phase (Training with Dense Reward)
                action, _ = algo.explore(obs)
                next_obs, env_reward, done, info = env.step(action)
                
                # Get current state for reward calculation
                curr_state = raw_env.state if hasattr(raw_env, 'state') else {}
                
                # Calculate dense reward (every step)
                # Calculate dense reward (every step)
                reward, reward_comp = calculate_dense_recovery_reward(
                    state=curr_state,
                    info=info,
                    action=action,
                    prev_action=prev_action,
                    target_speed_kmh=supervisor.target_speed_kmh,
                    verbose=(recovery_steps % 50 == 0)  # Log every 50 steps
                )
                
                reward_components_history.append(reward_comp)
                prev_action = action.copy()
                
                # Add terminal bonus if done successfully
                if done:
                    track_info = {'curvature': 0.0, 'ref_heading': 0.0}
                    if has_track_info:
                        try:
                            if hasattr(env, 'state'):
                                car_pos = np.array([env.state.get('position_x', 0), env.state.get('position_z', 0)])
                            else:
                                car_pos = np.array([raw_env.state.get('position_x', 0), raw_env.state.get('position_z', 0)])
                            
                            closest_idx = raw_env.track.closest_node(car_pos)
                            k_curr = raw_env.track.curvature[closest_idx]
                            h_curr = raw_env.track.heading[closest_idx]
                            track_info = {'curvature': k_curr, 'ref_heading': h_curr}
                        except Exception as e:
                            logger.error(f"Track info error: {e}")
                    
                    terminal_val = calculate_softmin_reward(next_obs, info, track_info, beta=2.0)
                    # Terminal reward scaled as jackpot bonus
                    reward += terminal_val * 2.0  # 2x multiplier for successful completion
                    logger.info(f"Ep {episode_idx} Terminal | Dense: {sum([c['total'] for c in reward_components_history]):.2f}, "
                               f"Softmin Bonus: {terminal_val*2.0:.2f}, Total: {reward:.2f}")

                mask = False if done else True
                agent._replay_buffer.append(obs, action, reward, next_obs, mask)

                if agent._replay_buffer._n > config.Agent.start_steps:
                    agent.update_model()

                episode_reward += reward
                steps_done += 1
                recovery_steps += 1
                obs = next_obs

        # --- Episode End ---
        
        # WandB logging
        if wandb_logger:
            log_dict = {
                "episode_reward": episode_reward,
                "terminal_softmin_reward": terminal_val,
                "recovery_steps": recovery_steps,
                "global_step": steps_done,
                "episode": episode_idx,
                "step": steps_done,
                # Recovery statistics
                "recovery/total_recovery_steps": recovery_steps,
                "recovery/episode_ended_in_recovery": int(supervisor.state == DrivingState.RECOVERY),
                # Training config
                "config/target_speed_kmh": supervisor.target_speed_kmh,
                "config/flick_intensity": supervisor.flick_intensity,
            }
            
            # Add average reward components
            if len(reward_components_history) > 0:
                avg_components = {
                    key: np.mean([comp[key] for comp in reward_components_history if key in comp])
                    for key in ['r_survival', 'r_slip', 'r_speed', 'r_align', 'r_action', 'r_oot']
                }
                log_dict.update({
                    "reward/survival": avg_components.get('r_survival', 0.0),
                    "reward/slip_suppression": avg_components.get('r_slip', 0.0),
                    "reward/speed_maintenance": avg_components.get('r_speed', 0.0),
                    "reward/counter_steer": avg_components.get('r_align', 0.0),
                    "reward/action_smoothness": avg_components.get('r_action', 0.0),
                    "reward/out_of_track_penalty": avg_components.get('r_oot', 0.0),
                })
                
                # Final state metrics
                if len(reward_components_history) > 0:
                    final_comp = reward_components_history[-1]
                    log_dict.update({
                        "final/max_slip_angle": final_comp.get('max_slip', 0.0),
                        "final/speed_kmh": final_comp.get('speed_kmh', 0.0),
                        "final/body_slip": final_comp.get('body_slip', 0.0),
                    })
            
            wandb_logger.log(log_dict)

        episode_idx += 1
        
        if episode_idx % 10 == 0:
            logger.info(f"Episode {episode_idx} | Reward: {episode_reward:.2f} | Steps: {steps_done}")
            agent.save(os.path.join(work_dir, "model_latest"))

    agent.save(os.path.join(work_dir, "model_final"))
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()