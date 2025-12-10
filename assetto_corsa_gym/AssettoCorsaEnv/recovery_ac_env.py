from AssettoCorsaEnv.ac_env import AssettoCorsaEnv
import numpy as np

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

        if max_slip < self.slip_threshold:
            self.slip_counter += 1
        else:
            self.slip_counter = 0

        if self.slip_counter >= self.required_steps:
            state["done"] = 1
            buf_infos["terminated"] = True # this is used by TD MPC which is not even used, but whatever
            buf_infos["slip_recovered"] = True # signal to let us know that it done = 1 
        else:
            buf_infos["slip_recovered"] = False

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
        """
        MODIFY THIS FUNCTION
        """

        if not hasattr(self.get_reward, "ema_slip"):
            self.get_reward.ema_slip = None

        if not hasattr(self.get_reward, "slip_history"):
            self.get_reward.slip_history = []

        if not hasattr(self.get_reward, "slip_window"):
            self.get_reward.slip_window = 5  # At 25Hz, this is ~0.2 seconds.


        # only look at back wheel slips
        slip_rl = abs(state["SlipAngle_rl"])
        slip_rr = abs(state["SlipAngle_rr"])
        slip_raw = max(slip_rl, slip_rr) / self.obs_channels_info['SlipAngle_rl']

        # EMA smoothing
        alpha = 0.3
        ema = self.get_reward.ema_slip
        slip_smooth = slip_raw if ema is None else alpha * slip_raw + (1 - alpha) * ema
        self.get_reward.ema_slip = slip_smooth


        # slip window
        hist = self.get_reward.slip_history
        hist.append(slip_smooth)

        W = self.get_reward.slip_window
        if len(hist) > W + 1:
            hist.pop(0)

        if len(hist) > W:
            slip_delta = hist[0] - hist[-1]   # improvement: positive means recovering
        else:
            slip_delta = 0.0


        r = 0.0

        # SLIP PENALTY
        r -= 3.0 * slip_smooth

        # IMPROVING SLIP REWARD
        r += 2.5 * slip_delta


        # Piecewise reference line deviation
        # Assumed maximum track deviation is ~10m
        if self.use_reference_line_in_reward:
            gap = abs(state["gap"])
            
            if gap < 2.0:
                r -= 0.05 * (gap / 2.0)
            else:
                r -= 0.20 * ((gap - 2.0) / 10.0)


        # Jerk penalty
        jerk = np.linalg.norm(actions_diff, ord=2)
        r -= 0.001 * jerk**2

        # To provide per-step feedback not to completely slow down to 0.
        # Clipped to small value
        # speed_bonus = 0.02 * state["local_velocity_x"]
        # speed_bonus = np.clip(speed_bonus, 0.0, 0.03)
        # r += speed_bonus


        # NO REWARD if speed is very low (happens becasue the car stalls on a wall), applied after the EMA updates.
        if(state["local_velocity_x"] < 5):
            return np.array([-0.1]).reshape(-1)


        # standard reward.
        return np.array([r]).reshape(-1)






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

            beta = 5.0
            #vals = np.array([R1, R2, R3])
            vals = np.array([R1, R2])

            # numerically stable softmin
            scaled = -beta * vals
            m = np.max(scaled)
            softmin = -(1.0/beta) * (m + np.log(np.sum(np.exp(scaled - m))))

            r_T = np.tanh(softmin / 50.0)   # 50 controls sharpness / magnitude

            # Range of slip_recovered rewards is between -10 to 10
            return 10 * float(r_T)

        # normal termination: crashed into wall, low speed, etc. 
        else:
            return -20
