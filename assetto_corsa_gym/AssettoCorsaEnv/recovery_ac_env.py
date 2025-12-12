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

        self.reward_log_path = os.path.join(
            os.getcwd(), f"reward_log_{timestamp}.csv"
        )

        # Write header only once
        with open(self.reward_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "slip_deg",
                "gap_m",
                "slip_norm",
                "gap_norm",
                "slip_pen",
                "gap_pen",
                "brake_help",
                "raw",
                "reward",
                "acc_x",
                "speed_x"
            ])

        self._reward_step_counter = 0



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


        # only terminate if slip_recovery_mode is on
        if self.slip_recovery_mode:
            if state["out_of_track"] and state["gap"] > 4:
                state["done"] = 1
                buf_infos["terminated"] = True
                buf_infos["slip_recovered"] = False
                self.early_termination = True
            else:
                if max_slip < self.slip_threshold:
                    self.slip_counter += 1
                else:
                    self.slip_counter = 0

                if self.slip_counter >= self.required_steps and state["gap"] < 3:
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

        # ---------- basic features ----------
        slip_deg = abs(max(state["SlipAngle_rl"], state["SlipAngle_rr"]))
        gap_m    = abs(state.get("gap", 0.0))
        speed_x  = state.get("local_velocity_x", 0.0)

        SLIP_GOOD = 5.0
        GAP_GOOD  = 2.0

        slip_norm = slip_deg / SLIP_GOOD
        gap_norm  = gap_m  / GAP_GOOD

        SLIP_CAP = 6.0
        GAP_CAP  = 4.0
        GAP_ALPHA = 0.2
        slip_norm = min(slip_norm, SLIP_CAP)
        gap_norm = min(gap_norm, GAP_CAP)
        #gap_norm = np.tanh(GAP_ALPHA * gap_norm)


        # ---------- update max slip ----------
        if slip_norm > self._max_slip_norm:
            self._max_slip_norm = slip_norm

        W_SLIP = 0.8
        W_GAP  = 0.3


        ALIVE  = W_SLIP + W_GAP + 1.0

        slip_pen = W_SLIP * (slip_norm ** 2)
        gap_pen  = W_GAP  * (gap_norm ** 2)


        raw = ALIVE - slip_pen - gap_pen
        #raw = ALIVE - slip_pen - gap_pen + rec_help + brake_help


        action_pen = 0.0
        if actions_diff is not None:
            ad = np.asarray(actions_diff, dtype=np.float32)


            ad = np.clip(ad, -2.0, 2.0)


            ACTION_WEIGHTS = np.array([1.0, 0.3, 0.3], dtype=np.float32)

            action_pen = float(np.sum(ACTION_WEIGHTS * (ad ** 2)))


            if slip_norm > 0.8:   # do not penaliz if > 5
                action_pen = 0.0

        raw -= 0.4 * action_pen
        SCALE = 0.05
        r = float(np.clip(SCALE * raw, -10.0, 2.0))

        self._prev_slip_norm = slip_norm
        self._reward_step_counter += 1

        if speed_x < 3:
            r = 0

        return np.array([r], dtype=np.float32)



    def terminal_reward(self, state, info):
        """
        Reward given only on episode termination.
        There are two ways you can get to the terminal reward. With a TRUE done signal and with slip_recovered true or false.
        """

        # successfully recovered from a slip
        if info["slip_recovered"]:
            v_exit = float(state["speed"])

            # heading stuff
            # car_yaw = float(state["yaw"])
            # ref_yaw = float(self.ref_lap.get_yaw(state["LapDist"]))

            # delta = car_yaw - ref_yaw
            # delta_heading = (delta + np.pi) % (2*np.pi) - np.pi

            LOCAL_LA_DIST = 40.0 # look ahead 40 meters ahead to calculate curvature
            # curv_vec = self.ref_lap.get_curvature_segment(
            #     dist=state["LapDist"],
            #     LA_dist=LOCAL_LA_DIST,
            #     vector_size=1
            # )

            curv_vec = self.ref_lap.get_curvature_segment(
                dist=state["LapDist"],
                LA_dist=LOCAL_LA_DIST,
                vector_size=5
            )
            kappa = np.mean(np.abs(curv_vec))
            #kappa = float(curv_vec[0])
            #kappa = max(1e-6, abs(kappa))

            # curvature based v_max
            mu = 1.2
            g = 9.81
            v_max_phys = np.sqrt(mu * g / kappa)
            v_max = min(80, v_max_phys) # 80 is the top speed in m/s

            R1 = 1.0 * v_exit
            R2 = -1.0 * (v_exit - v_max)**2
            #R3 = -0.3 * (delta_heading**2)


            steps_taken = self._reward_step_counter
            T_max = 350
            alpha = 10.0  # global weight for time penalty (tunable)
            T_norm = steps_taken / T_max
            R3 = -alpha * T_norm


            beta = 5.0
            vals = np.array([R1, R2, R3])
            #vals = np.array([R1, R2])

            # numerically stable softmin
            scaled = -beta * vals
            m = np.max(scaled)
            softmin = -(1.0/beta) * (m + np.log(np.sum(np.exp(scaled - m))))

            r_T = np.tanh(softmin / 50.0)   # 50 controls sharpness / magnitude

            # Range of slip_recovered rewards is between -10 to 10  
            return 10 * float(r_T) + 25

        # normal termination: crashed into wall, low speed, etc. 
        else:
            if self.early_termination:
                return -10
            return -5
