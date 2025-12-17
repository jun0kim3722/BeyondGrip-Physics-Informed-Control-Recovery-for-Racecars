from AssettoCorsaEnv.recovery_ac_env import RecoveryAssettoEnv

from AssettoCorsaEnv.ac_env import logger

import numpy as np

class PhysicsRecoveryEnv(RecoveryAssettoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_yaw_rate = 0.0
        self.prev_yaw_accel = 0.0
        self.prev_steer = 0.0
        self.avg_mz = 0.0
    
    def get_reward(self, state, actions_diff, info):
        r = super().get_reward(state, actions_diff, info)

        # call original reward, but also call your debugging.
        if (self.ep_steps % 50) == 0:
                    logger.debug(f't: {self.ep_steps} speed: {state["speed"]:.2f}, oot: {state["out_of_track"]} '
                                f's: {self.actions[0]:.2f} a: {self.actions[1]:.2f} b: {self.actions[2]:.2f} '
                                f'reward: {state["reward"]:.3f} '
                                f'done: {state["done"]:.0f} LapDist: {state["LapDist"]:.0f} gap: {state["gap"]:.1f} '
                                )
        return r

    def get_physics_reward(self, state, load_W=2.0, slip_W=2.0, steer_W=5.0, slip_thresh=3):
        steer = state['steerAngle'] / self.steering_scale_factor # in rads
        steer_vel = steer - self.prev_steer
        self.prev_steer = steer

        mz_fl, mz_fr, mz_rl, mz_rr = state['Mz']
        raw_front_mz = mz_fl + mz_fr
        clipped_mz = np.clip(raw_front_mz, -60.0, 60.0)
        self.avg_mz = 0.2 * clipped_mz + 0.8 * self.avg_mz

        slip_angle_fl = state['SlipAngle_fl']
        slip_angle_fr = state['SlipAngle_fr']
        slip_angle_rl = state['SlipAngle_rl']
        slip_angle_rr = state['SlipAngle_rr']

        front_slip_angle = (slip_angle_fl + slip_angle_fr) / 2.0
        rear_slip_angle = (slip_angle_rl + slip_angle_rr) / 2.0
        abs_front = np.abs(front_slip_angle)
        abs_rear = np.abs(rear_slip_angle)
        total_slip = abs_front + abs_rear + 1e-6
        os_score = abs_rear / total_slip 
        # us_score = abs_front / total_slip

        fl_load = state['fl_wheel_load']
        fr_load = state['fr_wheel_load']
        rl_load = state['rl_wheel_load']
        rr_load = state['rr_wheel_load']
        front_load = fl_load + fr_load
        rear_load = rl_load + rr_load
        total_load = front_load + rear_load
        front_load_ratio = front_load / max(total_load, 1e-6)
        rear_load_ratio = rear_load / max(total_load, 1e-6)

        # fl_slip_ratio = state['tyre_slip_ratio_fl']
        # fr_slip_ratio = state['tyre_slip_ratio_fr']
        rl_slip_ratio = state['tyre_slip_ratio_rl']
        rr_slip_ratio = state['tyre_slip_ratio_rr']

        reward = 0.0
        if self.slip_recovery_mode:
            if abs_rear > slip_thresh:
                # load transfer to rear to increase rear grip
                reward += os_score * load_W * (rear_load_ratio - front_load_ratio)

                # prevent wheel spin while trying to recover
                ratio_error = (rl_slip_ratio + rr_slip_ratio) / 2.0 - 0.015 # target slip ratio 1.5%
                reward += os_score * slip_W * np.exp(-(ratio_error**2) * 500) # reward for controlled rear ratio

                # steering correction
                steer_error = steer - np.clip(np.deg2rad(rear_slip_angle), -self.norm_steer_at_max, self.norm_steer_at_max) # clip to max steer
                reward += os_score * steer_W * np.exp(-(steer_error**2) * 100)

                # reaction steering from aligning torque
                steering_power = self.avg_mz * steer_vel
                if steering_power > 0.0:
                    reward += steer_W * np.tanh(steering_power / 10.0)
                elif abs(self.avg_mz) > 15.0:
                    reward -= steer_W * np.tanh(np.abs(steering_power / 10.0))

            else:
                steer_error = steer
                reward += os_score * steer_W * np.exp(-(steer_error**2) * 50) # reward for centering steering when slip is low

        return reward

    
    def get_reward(self, state, actions_diff, info):
        """
        Overrides the default get_reward
        YOU DO NOT NEED TO MODIFY THIS FUNCTION
        """

        r = self.dense_reward(state, actions_diff, info)
        physics_reward = self.get_physics_reward(state)
        r+= physics_reward
        self.physics_reward = physics_reward

        if state["done"]:
            r += self.terminal_reward(state, info)

        return r