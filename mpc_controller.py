import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import do_mpc
import casadi as ca

def compute_velocity_profile(
    xs, ys,
    mu=1.0, g=9.81,
    a_max=1.0,      # forward accel limit
    b_max=3.0,      # braking limit
    v_global_max=60.0,  # hard top speed [m/s]
    smooth_window=10,
    polyorder=3,
    max_iters=30,
    tol=1e-3
):
    N = len(xs)
    assert N == len(ys)

    w = smooth_window
    if w % 2 == 0:
        w += 1
    w = min(w, N - 1)
    xs_s = savgol_filter(xs, w, polyorder)
    ys_s = savgol_filter(ys, w, polyorder)

    ds_vec = np.hypot(np.diff(xs_s), np.diff(ys_s))
    ds_vec = np.append(ds_vec, np.hypot(xs_s[0]-xs_s[-1], ys_s[0]-ys_s[-1]))  # close loop
    s = np.concatenate(([0.0], np.cumsum(ds_vec[:-1])))

    dx_ds = np.gradient(xs_s, s)
    dy_ds = np.gradient(ys_s, s)
    ddx_ds2 = np.gradient(dx_ds, s)
    ddy_ds2 = np.gradient(dy_ds, s)

    denom = (dx_ds**2 + dy_ds**2)**1.5
    denom[denom < 1e-9] = 1e-9
    curvature = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom

    v_kappa = np.sqrt(np.maximum(mu * g / (np.abs(curvature) + 1e-9), 0.0))
    v_kappa = np.minimum(v_kappa, v_global_max)

    v = np.minimum(v_kappa, 50.0)

    for _ in range(max_iters):
        v_old = v.copy()

        # forward pass over full loop
        for i in range(1, N):
            ds = ds_vec[i-1]
            v[i] = min(v_kappa[i], np.sqrt(v[i-1]**2 + 2*a_max*ds), v_global_max)

        # connect last to first
        ds = ds_vec[-1]
        v[0] = min(v_kappa[0], np.sqrt(v[-1]**2 + 2*a_max*ds), v_global_max)

        # backward pass over full loop
        for i in range(N-2, -1, -1):
            ds = ds_vec[i]
            v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2*b_max*ds), v_kappa[i], v_global_max)

        # backward edge connecting
        ds = ds_vec[-1]
        v[-1] = min(v[-1], np.sqrt(v[0]**2 + 2*b_max*ds), v_kappa[-1], v_global_max)

        # convergence check
        if np.max(np.abs(v - v_old)) < tol:
            break

    return v

class MPC_controller:
    def __init__(self, config, steering_scale, target_kmh):
        track_name = config['AssettoCorsa']['track'] + "-racing_line.csv"
        track_dir = "assetto_corsa_gym/AssettoCorsaConfigs/tracks"
        racing_line_file = os.path.join(track_dir, track_name)
        self.racing_line = pd.read_csv(racing_line_file)
        self.rline_x = self.racing_line['pos_x'].values
        self.rline_y = self.racing_line['pos_y'].values
        self.rline_v = compute_velocity_profile(self.rline_x, self.rline_y, v_global_max=max(60, target_kmh / 3.6))
        self.steering_scale = 1 - steering_scale

        # plot racing line
        # plt.figure(figsize=(10, 6))
        # sc = plt.scatter(self.rline_x, self.rline_y, c=self.rline_v, cmap='jet', s=15)
        # cbar = plt.colorbar(sc)
        # cbar.set_label("Velocity")
        # plt.xlabel("X Position")
        # plt.ylabel("Y Position")
        # plt.title("Racing Line Colored by Velocity")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        self.spline_x = CubicSpline(np.arange(len(self.rline_x)), self.rline_x)
        self.spline_y = CubicSpline(np.arange(len(self.rline_y)), self.rline_y)

        self.n_horizon = 10
        self.look_ahead = 10
        self.steering_min = -np.pi
        self.steering_max = np.pi

        self.cur_x_ref = self.rline_x[0:self.n_horizon+1]
        self.cur_y_ref = self.rline_y[0:self.n_horizon+1]
        self.cur_v_ref = self.rline_v[0:self.n_horizon+1]

        self.model = self._build_model()
        self.mpc = self._build_mpc(self.model)
        self.sim = self._build_simulator(self.model)
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
    
        x0 = np.zeros((4,)) 
        self.sim.x0 = x0
        self.mpc.x0 = x0
        self.estimator.x0 = x0
        self.mpc.set_initial_guess()

        self.curr_steering = 0
        self.curr_throttle = 0
        self.curr_brake = 0

    def _build_model(self, wheelbase=2.5):
        model = do_mpc.model.Model('continuous')

        # states
        x = model.set_variable('_x', 'x')
        y = model.set_variable('_x', 'y')
        psi = model.set_variable('_x', 'psi')
        v = model.set_variable('_x', 'v')

        # control
        delta = model.set_variable('_u', 'delta')
        a = model.set_variable('_u', 'a')

        x_ref = model.set_variable('_tvp', 'x_ref')
        y_ref = model.set_variable('_tvp', 'y_ref')
        v_ref = model.set_variable('_tvp', 'v_ref')

        # bicycle dynamics
        L = wheelbase
        model.set_rhs('x', v * ca.cos(psi))
        model.set_rhs('y', v * ca.sin(psi))
        model.set_rhs('psi', v / L * ca.tan(delta * self.steering_scale))
        model.set_rhs('v', a)

        model.setup()
        return model

    def _build_mpc(self, model, t_step=0.01):
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(n_horizon=self.n_horizon, t_step=t_step, u_deriv={'delta': False, 'a': True})

        x = model.x['x']
        y = model.x['y']
        v = model.x['v']
        delta = model.u['delta']
        a = model.u['a']

        x_ref = model.tvp['x_ref']
        y_ref = model.tvp['y_ref']
        v_ref = model.tvp['v_ref']

        pos_err = (x - x_ref)**2 + (y - y_ref)**2
        spd_err = (v - v_ref)**2

        mpc.set_objective(
            mterm=pos_err,
            lterm=pos_err + 0.5 * spd_err
        )
        mpc.set_rterm(delta=0.3, a=2.0)

        # constraints
        mpc.bounds['lower','_u','a'] = -3.0
        mpc.bounds['upper','_u','a'] =  2.0
        mpc.bounds['lower','_u','delta'] = self.steering_min
        mpc.bounds['upper','_u','delta'] = self.steering_max

        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(self.n_horizon + 1):
                tvp_template['_tvp', k, 'x_ref'] = self.cur_x_ref[k]
                tvp_template['_tvp', k, 'y_ref'] = self.cur_y_ref[k]
                tvp_template['_tvp', k, 'v_ref'] = self.cur_v_ref[k]
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        return mpc

    def _build_simulator(self, model, t_step=0.01):
        sim = do_mpc.simulator.Simulator(model)
        sim.set_param(t_step=t_step)
        tvp_template = sim.get_tvp_template()

        def tvp_fun(t_now):
            tvp_template['x_ref'] = self.cur_x_ref[0]
            tvp_template['y_ref'] = self.cur_y_ref[0]
            tvp_template['v_ref'] = self.cur_v_ref[0]
            return tvp_template

        sim.set_tvp_fun(tvp_fun)

        sim.setup()
        return sim

    def _nearest_idx(self, x, y, psi):
        n_points = len(self.rline_x)
        dx = self.rline_x - x
        dy = self.rline_y - y
        closest_idx = int(np.argmin(dx*dx + dy*dy))

        search_range = self.n_horizon
        indices = (np.arange(closest_idx, closest_idx + search_range) % n_points).astype(int)

        best_idx = closest_idx
        min_distance = np.inf
        heading_vector = np.array([np.cos(psi), np.sin(psi)])

        for i in indices:
            px, py = self.rline_x[i], self.rline_y[i]
            path_vector = np.array([px - x, py - y])
            if np.dot(heading_vector, path_vector) > 0:
                dist_sq = (px - x)**2 + (py - y)**2
                if dist_sq < min_distance:
                    min_distance = dist_sq
                    best_idx = i

        idxs = (np.arange(best_idx, best_idx + self.n_horizon + 1) % n_points).astype(int)
        return idxs

    def get_action(self, vehicle_state):
        x = float(vehicle_state['world_position_x'])
        y = float(vehicle_state['world_position_y'])
        psi = float(vehicle_state['yaw'])
        v = float(vehicle_state['speed'])

        self.look_ahead = max(5, int(0.5 * v))

        mpc_state = np.array([x, y, psi, v])

        x_est = self.estimator.make_step(mpc_state)

        idxs = self._nearest_idx(x_est[0], x_est[1], x_est[2])
        horizon_points = np.linspace(idxs[0], idxs[0]+self.look_ahead, self.n_horizon+1)
        self.cur_x_ref = self.spline_x(horizon_points % len(self.rline_x))
        self.cur_y_ref = self.spline_y(horizon_points % len(self.rline_y))
        self.cur_v_ref = self.rline_v[idxs]
        u = self.mpc.make_step(x_est)

        # control processing
        steering = -float(u[0]) - self.curr_steering
        self.curr_steering = -float(u[0])
    
        a = float(u[1])
        if a > 0:
            brake = -1.0
            throttle = min(1.0, a - self.curr_throttle)

            self.curr_throttle += throttle
            self.curr_brake = max(0.0, self.curr_brake + brake)

        else:
            throttle = -1.0
            brake = min(1.0, -a - self.curr_brake)

            self.curr_throttle = max(0.0, self.curr_throttle + throttle)
            self.curr_brake += brake

        # print("Target: ", -float(u[0]), a)
        # print("Prev: ", self.curr_steering, self.curr_throttle, self.curr_brake)
        # print("CTR: ", steering, throttle, brake)

        return np.array([steering, throttle, brake])
    
    def reset(self):
        """
        Resets the MPC controller state and solver memory.
        This is crucial for restarting the episode without warm-start errors.
        """
        # 1. Reset current control inputs
        self.curr_steering = 0
        self.curr_throttle = 0
        self.curr_brake = 0

        # 2. Reset state variables (x0)
        x0 = np.zeros((4, 1))
        self.mpc.x0 = x0
        self.sim.x0 = x0
        self.estimator.x0 = x0

        # 3. remove do_mpc history
        self.mpc.reset_history()
        self.sim.reset_history()
        self.estimator.reset_history()

        # 4. Reset solver's initial guess
        self.mpc.set_initial_guess()