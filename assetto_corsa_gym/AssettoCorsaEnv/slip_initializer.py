import numpy as np

class SlipInitializer:
    """
    Drives car to a target location + speed, then induces slip.
    Leaves env.state at the slip moment so RL can take over.
    """

    def __init__(self, env, mpc):
        self.env = env
        self.mpc = mpc

        # Slip detection settings
        self.slip_threshold = 0.20  # radians
        self.required_consecutive = 2
        self.feint_steps = 7
        self.flick_steps = 7

    def perform_slip(self, target_dist, target_speed_ms):
        """
        Blocking routine that:
        1. Drives car to target LapDist
        2. Drives to target speed
        3. Executes feint / flick maneuver
        4. Stops when slip is detected
        """
        # =========================================================================
        # 1. Drive to the correct location AND speed before starting slip sequence
        # =========================================================================
        while True:
            state = self.env.state

            # If episode terminated early (crash, out of track) → abort
            if state["done"]:
                print(">>> Early termination detected during drive-to-dist phase.")
                return

            # condition satisfied
            if state["LapDist"] >= target_dist and state["speed"] >= target_speed_ms:
                break

            # Otherwise use MPC
            action = self.mpc.get_action(state)
            self.env.step(action)

        print(f">>> SlipInitializer: reached LapDist={state['LapDist']:.1f}m at {state['speed']*3.6:.1f} km/h")

        # =========================================================================
        # 2. Phase 1 — Feint right with lift-off
        # =========================================================================
        for _ in range(self.feint_steps):
            if self.env.state["done"]:
                print(">>> Early termination during feint.")
                return
            action = np.array([+0.5, 0.0, 0.0])  # steer right, no gas
            self.env.step(action)

        # =========================================================================
        # 3. Phase 2 — Flick left + full gas until slip detected
        # =========================================================================
        consecutive = 0
        steps = 0

        while True:
            if self.env.state["done"]:
                print(">>> Early termination during flick phase.")
                return

            action = np.array([-1.0, 1.0, 0.0])  # aggressive flick
            self.env.step(action)
            steps += 1

            # only check slip after flick_steps
            if steps > self.flick_steps:
                s = self.env.state
                max_slip = max(abs(s["SlipAngle_rl"]), abs(s["SlipAngle_rr"]))

                if max_slip > self.slip_threshold:
                    consecutive += 1
                else:
                    consecutive = 0

                if consecutive >= self.required_consecutive:
                    print(f">>> Slip detected at LapDist={s['LapDist']:.1f} (slip={max_slip:.3f})")
                    return  # leave env in slip state

            # fallback safeguard
            if steps > 20:
                print(">>> SlipInitializer fallback: slip not detected. Aborting.")
                return
