from AssettoCorsaEnv.ac_env import AssettoCorsaEnv
import numpy as np
import csv
import os
from datetime import datetime

class RecoveryAssettoEnv(AssettoCorsaEnv):
    def __init__(self, *args,
                 slip_threshold=7, # measured in degrees as per ACPythonDocumentation.pdf
                 recovery_time=1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.slip_threshold = slip_threshold
        self.recovery_time = recovery_time
        self.required_steps = int(self.ctrl_rate * recovery_time)
        self.slip_counter = 0
        self.slip_recovery_mode = False

        self.early_termination = False



        # Create a top-level reward log file (same folder as train_recovery.py)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._reward_step_counter = 0
        self._low_speed_time = 0

        self.slip_severity = 0
        self.gap_severity = 0
        self.speed_severity = 0
        self.crash_severity = 0

        self.time_bonus = 0
        self.speed_bonus = 0
        self.gap_bonus = 0




    def reset(self):
        self.slip_counter = 0
        self.ema_slip = None
        self.slip_history = []

        self.slip_recovery_mode = False
        self.early_termination = False

        self._reward_step_counter = 0

        slip_deg0 = 0
        SLIP_GOOD = 10.0
        self._slip_norm0 = slip_deg0 / SLIP_GOOD
        self._max_slip_norm = self._slip_norm0
        self._prev_slip_norm = self._slip_norm0
        self._reward_step_counter = 0

        self._low_speed_time = 0

        self._prev_yaw = None
        self._prev_gap = None
        self.slip_severity = 0
        self.gap_severity = 0
        self.speed_severity = 0
        self.crash_severity = 0

        self.time_bonus = 0
        self.speed_bonus = 0
        self.gap_bonus = 0



        return super().reset()

    def expand_state(self, state):
        '''
        Only put additional things you want in the expand_state here. The original function is already called.
        '''
        # use original call first.
        state, buf_infos = super().expand_state(state)


        # Probably don't modify things below this line in this function.
        # Check to see if terminated by base. This means it crashed, low speed, etc.
        if state["done"]:
            buf_infos.setdefault("slip_recovered", False)
            return state, buf_infos

        # Our Special termination condition
        # Terminate After the max slip angle is below 7 degrees for 1 second (based on constructor inputs)
        slip_rl = abs(state["SlipAngle_rl"])
        slip_rr = abs(state["SlipAngle_rr"])
        max_slip = max(slip_rl, slip_rr)


        if state["speed"] < 4:
            self._low_speed_time += 1
        else:
            self._low_speed_time = 0

        # only terminate if slip_recovery_mode is on
        if self.slip_recovery_mode:
            if (state["out_of_track"] and abs(state["gap"]) > 1) or self._low_speed_time > 10:
                state["done"] = 1
                buf_infos["terminated"] = True
                buf_infos["slip_recovered"] = False
                self.early_termination = True
            else:
                if max_slip < self.slip_threshold:
                    self.slip_counter += 1
                else:
                    self.slip_counter = 0

                if self.slip_counter >= self.required_steps and abs(state["gap"]) < 3:
                    state["done"] = 1
                    buf_infos["terminated"] = True
                    buf_infos["slip_recovered"] = True
        else:
            buf_infos["slip_recovered"] = False

        # if max_slip < self.slip_threshold:
        #     self.slip_counter += 1
        # else:
        #     self.slip_counter = 0

        # if self.slip_counter >= self.required_steps:
        #     state["done"] = 1
        #     buf_infos["terminated"] = True # this is used by TD MPC which is not even used, but whatever
        #     buf_infos["slip_recovered"] = True # signal to let us know that it done = 1 
        # else:
        #     buf_infos["slip_recovered"] = False

        return state, buf_infos
    

    def get_reward(self, state, actions_diff, info):
        """
        Overrides the default get_reward
        YOU DO NOT NEED TO MODIFY THIS FUNCTION
        """

        r = self.dense_reward(state, actions_diff, info)

        if state["done"]:
            r += self.terminal_reward(state, info)

        return r


    def dense_reward(self, state, actions_diff, info):
        if hasattr(self, "slip_recovery_mode") and not self.slip_recovery_mode:
            return np.array([0.0], dtype=np.float32)

        slip_deg = max(abs(state["SlipAngle_rl"]), abs(state["SlipAngle_rr"]))
        gap_m    = abs(state.get("gap", 0.0))
        prev_gap = getattr(self, "_prev_gap", None)
        if prev_gap is None:
            gap_delta = 0.0
        else:
            gap_delta = prev_gap - gap_m   # positive if improving bc we take abs of the gap

        self._prev_gap = gap_m

        v_lat = state["local_velocity_y"]



        # normalize (meters)
        gap_delta_norm = np.clip(gap_delta / 0.5, -1.0, 1.0)

        # weight
        W_GAP_IMPROVE = 6.0
        r_gap_improve = W_GAP_IMPROVE * gap_delta_norm



        yaw = state["angular_velocity_z"]  # signed float

        prev_yaw = getattr(self, "_prev_yaw", None)

        # Always safe
        yaw_rate = abs(yaw)

        if prev_yaw is None:
            yaw_acc = 0.0
            yaw_flip = False
        else:
            yaw_acc = yaw - prev_yaw
            yaw_flip = (yaw * prev_yaw) < 0.0

        self._prev_yaw = yaw

        SLIP_GOOD = 5.0
        GAP_GOOD  = 2.0

        # SLIP_LOCK = 2.5    # full yaw damping below this
        # SLIP_FREE = 6.0    # no yaw damping above this

        # yaw_scale = np.clip(
        #     (SLIP_FREE - slip_deg) / (SLIP_FREE - SLIP_LOCK),
        #     0.0, 1.0
        # )

        # r_yaw_rate = -1.2 * yaw_scale * (yaw_rate ** 2)
        # r_yaw_acc  = -0.6 * yaw_scale * (yaw_acc  ** 2)

        # flip pentaliteis



        slip_norm = slip_deg / SLIP_GOOD
        gap_norm  = gap_m  / GAP_GOOD # ALWAYS POSITIVE

        SLIP_CAP = 6.0
        GAP_CAP  = 4.0
        slip_norm = min(slip_norm, SLIP_CAP)
        gap_norm = min(gap_norm, GAP_CAP)


        if slip_norm > self._max_slip_norm:
            self._max_slip_norm = slip_norm

        W_SLIP = 0.8
        W_GAP  = 0.1


        ALIVE  = W_SLIP + W_GAP + 1.0

        if slip_norm < 1.5:
            if gap_norm < 2:
                ALIVE += 3
            elif gap_norm < 4:
                ALIVE += 1.5


        slip_pen = W_SLIP * (slip_norm ** 2)
        gap_pen  = W_GAP  * (gap_norm ** 2)

        slip_pen = min(slip_pen, 20)
        gap_pen = min(gap_pen, 15)

        t = self._reward_step_counter
        if t <= 100:
            late_scale = 1.0
        else:
            late_scale = max(0, (450 - t) / (450 - 100))
        slip_pen *= late_scale
        gap_pen *= late_scale

        raw = ALIVE - slip_pen - gap_pen + r_gap_improve
        flip_pen = 0
        flip_pen = -0.2 if (yaw_flip and slip_deg < SLIP_GOOD) else 0.0


        #raw += flip_pen
        #raw += r_yaw_rate + r_yaw_acc


        # steer_rate_pen =0.0
        # if actions_diff is not None:
        #     ad = np.asarray(actions_diff, dtype=np.float32)
        #     dsteer = float(ad[0])  # assuming action=[steer, gas, brake]

        #     is_stable = (slip_deg < SLIP_GOOD) and (gap_m < 4)
        #     if is_stable and actions_diff is not None:
        #         ad = np.clip(np.asarray(actions_diff, np.float32), -1.0, 1.0)
        #         dsteer, dthrot, dbrake = float(ad[0]), float(ad[1]), float(ad[2])

        #         Ls = 2.0   # steering delta weight
        #         Lt = 0.3   # throttle delta weight (smaller)
        #         Lb = 0.3   # brake delta weight (smaller)

        #         #raw -= (Ls * dsteer**2 + Lt * dthrot**2 + Lb * dbrake**2)

        #raw -= steer_rate_pen

        
        # GAP_SAFE = 2.0   # meters
        # GAP_DANGER = 6.0   # meters (near edge)

        # danger = np.clip((gap_m - GAP_SAFE) / (GAP_DANGER - GAP_SAFE), 0.0, 1.0)

        # v_lat_clip = min(abs(v_lat) / 5.0, 1)
        # raw -= 0.6 * late_scale * danger * v_lat_clip

        # V_LAT_REF = 4.0
        # v_lat_n = np.clip(v_lat / V_LAT_REF, -2.0, 2.0)

        # VLAT_MIN = 0.3
        # VLAT_FULL = 4.0
        # v_lat_scale = np.clip(
        #     (VLAT_FULL - slip_deg) / (VLAT_FULL - VLAT_MIN),
        #     0.3, 1.0
        # )
        # gap_signed = state.get("gap", 0.0)

        # r_vlat_align = 0.8 * v_lat_scale * np.tanh(-gap_signed * v_lat_n)
        # r_vlat_mag   = -0.2 * v_lat_scale * (v_lat_n ** 2)

        #raw += r_vlat_align + r_vlat_mag


        SCALE = 0.2
        r = float(np.clip(SCALE * raw, -8.0, 2.0))

        self._prev_slip_norm = slip_norm
        self._reward_step_counter += 1

        return np.array([r], dtype=np.float32)



    def terminal_reward(self, state, info):
        # """
        # Reward given only on episode termination.
        # There are two ways you can get to the terminal reward. With a TRUE done signal and with slip_recovered true or false.
        # """

        # # successfully recovered from a slip
        # if info["slip_recovered"]:
        #     v_exit = float(state["speed"])

        #     # heading stuff
        #     # car_yaw = float(state["yaw"])
        #     # ref_yaw = float(self.ref_lap.get_yaw(state["LapDist"]))

        #     # delta = car_yaw - ref_yaw
        #     # delta_heading = (delta + np.pi) % (2*np.pi) - np.pi

        #     LOCAL_LA_DIST = 40.0 # look ahead 40 meters ahead to calculate curvature
        #     # curv_vec = self.ref_lap.get_curvature_segment(
        #     #     dist=state["LapDist"],
        #     #     LA_dist=LOCAL_LA_DIST,
        #     #     vector_size=1
        #     # )

        #     curv_vec = self.ref_lap.get_curvature_segment(
        #         dist=state["LapDist"],
        #         LA_dist=LOCAL_LA_DIST,
        #         vector_size=5
        #     )
        #     kappa = np.mean(np.abs(curv_vec))
        #     #kappa = float(curv_vec[0])
        #     #kappa = max(1e-6, abs(kappa))

        #     # curvature based v_max
        #     mu = 1.2
        #     g = 9.81
        #     v_max_phys = np.sqrt(mu * g / kappa)
        #     v_max = min(80, v_max_phys) # 80 is the top speed in m/s

        #     #R1 = 1.0 * v_exit
        #     R1 = np.tanh(v_exit / 20)
        #     speed_excess = max(0.0, v_exit - v_max)
        #     R2 = -1.0 * (speed_excess)**2
        #     #R3 = -0.3 * (delta_heading**2)


        #     steps_taken = self._reward_step_counter
        #     T_max = 350
        #     alpha = 10.0  # global weight for time penalty (tunable)
        #     T_norm = steps_taken / T_max
        #     R3 = -alpha * T_norm


        #     beta = 1.0
        #     vals = np.array([R1, R2, R3])
        #     #vals = np.array([R1, R2])

        #     # numerically stable softmin
        #     scaled = -beta * vals
        #     m = np.max(scaled)
        #     softmin = -(1.0/beta) * (m + np.log(np.sum(np.exp(scaled - m))))

        #     r_T = np.tanh(softmin / 50.0)   # 50 controls sharpness / magnitude

        #     # Range of slip_recovered rewards is between -10 to 10  
        #     return 10 * float(r_T) + 30

        # # normal termination: crashed into wall, low speed, etc. 
        # else:
        #     if self.early_termination:
        #         return -10
        #     return -5


        slip = max(abs(state["SlipAngle_rl"]), abs(state["SlipAngle_rr"]))
        gap  = abs(state["gap"])
        speed = state["speed"] * 3.6 #kmh
        T = self._reward_step_counter

        if info.get("slip_recovered", False):
            time_bonus = 1.0 - (T / 350.0)
            speed_bonus = np.tanh(speed / 30.0)
            GAP_OK = 1.5
            gap_bonus = np.exp(-gap / GAP_OK)

            self.time_bonus = time_bonus
            self.speed_bonus = speed_bonus
            self.gap_bonus = gap_bonus
            return 30.0 + 10.0 * time_bonus + 5.0 * speed_bonus + 15 * gap_bonus

        slip_n = min(slip / 20.0, 2.0)
        gap_n  = min(gap  / 8.0,  1.5)
        speed_n = min(speed / 30, 1.5)

        crash_severity = (
            1.2 * slip_n**2 +
            0.5 * gap_n**2 +
            0.25 * speed_n**2
        )


        self.slip_severity = slip_n**2
        self.gap_severity = gap_n **2
        self.speed_severity = speed_n ** 2
        self.crash_severity = crash_severity
        # time survived softens punishment exponentially

        t = T / 25.0
        decay = np.exp(-t / 6.0)


        return -20.0 - 20.0 * crash_severity * decay #(1.0 - time_factor)