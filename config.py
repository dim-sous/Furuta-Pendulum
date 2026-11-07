"""All plant parameters and run settings (viz.py keeps only layout and styling constants).
SI units; angles in radians (deg() converts from degrees). Sources [n] are the References in METHODS.md.
"""
from numpy import deg2rad as deg, pi, sin

G = 9.81                                # m/s^2 [7]

# Arm: uniform rod pivoted at its end on the motor axis [6]
M_ARM = 0.095                           # kg
L_ARM = 0.085                           # m, motor axis to pendulum pivot
D_ARM = L_ARM / 2                       # m, motor axis to the arm's centre of mass
J_ARM = (0.0, M_ARM * L_ARM**2 / 12, M_ARM * L_ARM**2 / 12)   # kg m^2, about its centre of mass; x along the arm, z up
J_ROTOR = 4.0e-6 + 0.6e-6               # kg m^2, motor rotor + module attachment hub, about the motor axis [6]

# Pendulum: uniform thin rod pivoted at its end [6]
M_PEND = 0.024                          # kg
L_PEND = 0.129                          # m, pivot to tip
D_PEND = L_PEND / 2                     # m, pivot to centre of mass
J_PEND = (M_PEND * L_PEND**2 / 12, M_PEND * L_PEND**2 / 12, 0.0)   # kg m^2, about its centre of mass; x = pivot axis, z along the rod

# Friction torque per joint [4]: viscous [N m s/rad], Coulomb and breakaway [N m], breakaway speed [rad/s]
FRICTION_ARM = dict(viscous=1e-3,       # [7]
                    coulomb=1.02e-4,    # [5]
                    breakaway=2.03e-4,  # [5]
                    w_breakaway=0.1)    # not published by [5]; default of [4]
FRICTION_PEND = dict(viscous=5e-5,      # [7]
                     coulomb=0.0,       # no published value
                     breakaway=0.0,
                     w_breakaway=0.1)

# Simulation
DT = 1e-3                               # s, controller sample period; the torque is held constant over each period
RK4_SUBSTEPS = 4                        # RK4 steps per sample period: integration step DT / RK4_SUBSTEPS
ENERGY_TOL = 1e-6                       # J, largest allowed |E(t) - E(0) - motor work + friction loss|

# Every controller's torque is capped at MAX_TORQUE, the real motor's limit at 10 V [6]: controllers are compared
# under the same rule.
MAX_TORQUE = 0.05                       # N m

# PID on the pendulum angle plus PID on the arm angle (target 0), summed; gains set by hand or by tune.py.
# Kp [N m/rad], Ki [N m/(rad s)], Kd [N m s/rad]; the *_arm gains have the same units.
PID_GAINS = dict(Kp_pendulum=0.673346, Ki_pendulum=5.97908e-10, Kd_pendulum=0.0450316, Kp_arm=-0.0738805, Ki_arm=-0.000453746, Kd_arm=-0.0302108)

# Gain tuning (tune.py, which prints new gains to paste into PID_GAINS). It minimizes, summed over the starts, the
# integral over `horizon` of (pendulum error / max_pendulum_error)^2 + (arm error / max_arm_error)^2
# + (torque / max_torque)^2. Each start is the pendulum at rest at alpha0 with the arm at 0, on one side of
# upright only (180 deg + d mirrors 180 deg - d); starts are added one at a time. PID_GAINS is the first guess,
# and the sign of each gain is kept. Afterwards the tuned gains are run from `check_starts` for `check_t_end`.
TUNE = dict(starts=deg([175, 170, 165, 160]),
            horizon=3.0,                        # s, multiple of DT
            friction=True,
            max_pendulum_error=deg(5),          # rad
            max_arm_error=deg(30),              # rad
            max_torque=MAX_TORQUE,              # N m
            cap=float("inf"),                   # N m, torque cap inside the tuning loop: none (see tune.py)
            max_iter=200,                       # IPOPT iterations per stage
            check_starts=deg([140, 145, 150, 155, 160, 165, 170, 175]),
            check_t_end=10.0)                   # s

# Swing-up by energy control (Astrom & Furuta 2000 [17]), its torque capped at MAX_TORQUE. The PID takes over
# within `enter` of upright [9] and hands back only beyond `leave` (hysteresis), so the PID's own overshoot while
# catching does not cut it off.
SWING_UP = dict(k=50.0,                         # N m/J, torque per joule of missing pendulum energy
                enter=deg(20),                  # rad from upright
                leave=deg(45))                  # rad from upright

# Disturbance: a tap on the pendulum, a torque pulse on its joint (not capped: it is not the controller), once the
# swing-up has balanced the pendulum. One experiment per size, growing, to find the largest tap it recovers from.
TAP = dict(t=5.0,                               # s, start
           duration=0.02,                       # s, multiple of DT
           sizes=[0.0025, 0.005, 0.01, 0.02, 0.04])   # N m

# Encoders on both joints: 2048 counts per revolution in quadrature [22]. A reading is the angle rounded to whole
# counts, zero where the model's angles are zero (a calibrated encoder).
ENCODER_RESOLUTION = 2 * pi / 2048      # rad per count

# State estimation from the two encoder angles, for experiments with state="ekf": a continuous-discrete extended
# Kalman filter (estimator.py) [18]. Its model is the plant's with viscous friction only, while the truth keeps
# the Coulomb and Stribeck terms, so the filter never has the truth's model. Process noise: white torque noise on
# the arm's and the pendulum's joints, entering as the motor torque and the tap do, of intensity Q_c per joint;
# first guess (1e-4 N m)^2 * DT, the dropped Coulomb torque held over one sample, then tuned by hand by NEES/NIS
# [21]. Measurement noise: the quantization's variance, resolution^2 / 12 per angle [19]. The filter starts from
# the first reading with zero velocities, uncertain by P0_std.
EKF = dict(friction_arm=dict(FRICTION_ARM, coulomb=0.0, breakaway=0.0),
           friction_pend=dict(FRICTION_PEND, coulomb=0.0, breakaway=0.0),
           Q_c=[1e-11, 1e-11],                  # (N m)^2 s, arm joint, pendulum joint
           R=[ENCODER_RESOLUTION**2 / 12] * 2,  # rad^2, theta, alpha
           P0_std=[ENCODER_RESOLUTION / 12**0.5] * 2 + [1.0, 1.0])   # rad, rad, rad/s, rad/s

# Experiments: x0 = [theta, alpha, theta_dot, alpha_dot]; alpha = 0 hanging down, +-pi upright.
# Each experiment names its controller and that controller's settings:
#   "open_loop": torque(t) [N m];   "pid": gains, a dict like PID_GAINS;
#   "swing_up": gains (the PID's) and swing_up, a dict like SWING_UP.
# Optional: disturbance(t) [N m], a torque on the pendulum's joint (zero if absent); state, what the controller
# sees: "true" (the true state, the default) or "ekf" (the EKF's estimate from the encoders).
EXPERIMENTS = {
    "frictionless_release": dict(x0=[0, deg(175), 0, 0], t_end=10.0, friction=False,
                                 controller="open_loop", torque=lambda t: 0.0),
    "release_near_upright": dict(x0=[0, deg(175), 0, 0], t_end=15.0, friction=True,
                                 controller="open_loop", torque=lambda t: 0.0),
    "release_horizontal": dict(x0=[0, deg(90), 0, 0], t_end=15.0, friction=True,
                               controller="open_loop", torque=lambda t: 0.0),
    "torque_pulse": dict(x0=[0, 0, 0, 0], t_end=10.0, friction=True,
                         controller="open_loop", torque=lambda t: 0.02 if 0.5 <= t < 0.6 else 0.0),
    "resonant_pumping": dict(x0=[0, 0, 0, 0], t_end=15.0, friction=True,
                             controller="open_loop",
                             torque=lambda t: 0.004 * sin(2 * pi * 2.06 * t)),   # 2.06 Hz: small-swing frequency
    "pid_near_upright": dict(x0=[0, deg(170), 0, 0], t_end=10.0, friction=True, controller="pid", gains=PID_GAINS),
    "pid_from_rest": dict(x0=[0, 0, 0, 0], t_end=10.0, friction=True, controller="pid", gains=PID_GAINS),
    "swing_up_from_rest": dict(x0=[0, 0, 0, 0], t_end=10.0, friction=True, controller="swing_up",
                               gains=PID_GAINS, swing_up=SWING_UP),
    **{f"swing_up_tap_{1000 * size:g}mNm": dict(
        x0=[0, 0, 0, 0], t_end=10.0, friction=True, controller="swing_up", gains=PID_GAINS, swing_up=SWING_UP,
        disturbance=lambda t, size=size: size if TAP["t"] <= t < TAP["t"] + TAP["duration"] else 0.0)
       for size in TAP["sizes"]},
}
EXPERIMENTS |= {f"{name}_ekf": dict(EXPERIMENTS[name], state="ekf")      # twins whose controller sees the EKF
                for name in ("pid_near_upright", "swing_up_from_rest")}
RUN = ["pid_near_upright_ekf", "swing_up_from_rest_ekf"] # list(EXPERIMENTS)

# Output
RESULTS_DIR = "results"
VIDEO_PX = 1080                         # square video, pixels per side
VIDEO_FPS = 30
TRAIL_S = 1.0                           # s of pendulum-tip trail in the video
PLOT_WINDOW_S = 5.0                     # s of history in the video's scrolling plots
