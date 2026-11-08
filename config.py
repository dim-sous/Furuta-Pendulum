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

# Encoders on both joints: 2048 counts per revolution in quadrature [21]. A reading is the angle rounded to whole
# counts, zero where the model's angles are zero (a calibrated encoder).
ENCODER_RESOLUTION = 2 * pi / 2048      # rad per count

# The arm friction identified by friction identification step 1 (printed by sysid.py, pasted here):
# viscous [N m s/rad], Coulomb [N m]. The EKF's arm model uses it.
FRICTION_ID = dict(viscous=9.9945e-04, coulomb=1.1141e-04)

# State estimation from the two encoder angles, for experiments with state="ekf": a continuous-discrete extended
# Kalman filter (estimator.py) [18]. Its arm friction is viscous + Coulomb as identified from encoder data
# (FRICTION_ID; Coulomb smoothed over 0.01 rad/s, the band of the Simscape default breakaway speed 0.1 rad/s
# [4]), while the truth keeps its Stribeck law, so the filter never has the truth's model; its pendulum friction is
# still the plant's viscous value. Process noise: white torque noise on the arm's and the pendulum's joints,
# entering as the motor torque and the tap do, of intensity Q_c per joint, tuned by hand on a sweep (2026-11-07):
# 1e-8, the lowest value at which both EKF experiments keep the pendulum (5e-9 loses it); with viscous-only
# friction the filter needed 1e-7. Measurement: a reading says the angles lie in its count cell, and the update
# conditions on that cell [22-25]; a run fails a check when a reading's cell has probability below min_cell_prob
# under the filter's belief (a big surprise, where the update's moments also start losing accuracy). The filter
# starts from the first reading with zero velocities, uncertain by P0_std (the angles: a uniform spread over one
# count, std resolution / sqrt(12) [19]).
EKF = dict(friction_arm=dict(viscous=FRICTION_ID["viscous"], coulomb=FRICTION_ID["coulomb"],
                             breakaway=FRICTION_ID["coulomb"], w_breakaway=0.1),   # breakaway = Coulomb: no Stribeck
           friction_pend=dict(FRICTION_PEND, coulomb=0.0, breakaway=0.0),
           Q_c=[1e-8, 1e-8],                    # (N m)^2 s, arm joint, pendulum joint
           P0_std=[ENCODER_RESOLUTION / 12**0.5] * 2 + [1.0, 1.0],   # rad, rad, rad/s, rad/s
           min_cell_prob=1e-6,
           # An unknown push on the pendulum's joint as a fifth state, fading as a first-order Gauss-Markov process
           # [34, 28] (estimator.ekf), e.g. dict(tau=0.05, std=0.005) [s, N m]; None: four states only. Tried
           # 2026-11-08 against taps (FINDINGS.md): no push size worked for both controllers, so off.
           disturbance=None)

# Nonlinear MPC (control.nmpc, METHODS.md section 10): one law swings up and balances, with no switching. Every
# `period` it solves the optimal control problem over the next `horizon` s, predicting with the EKF's model
# (identified arm friction), and holds the first torque until its next sample (the plant and the EKF keep DT).
# period: 20-40x the closed-loop bandwidth [31], the bandwidth about twice the upright pendulum's unstable pole,
# 12.9 rad/s [32], gives 6-12 ms; 5 ms, the fast end, worked on a real Furuta pendulum, whose swing-up failed above
# 9 ms [30]. substeps: RK4 steps per period in the prediction. Against a fine-step reference, 1.5 s predictions
# along the energy swing-up erred by at most 0.045 deg with 3 steps (a quarter of an encoder count), 0.146 deg
# with 2 (83% of a count: too little margin, as the error is not monotone in the step count near the Coulomb
# band) (2026-11-08). horizon: published swing-up NMPCs predict about the whole swing-up ([30]: 350 ms ahead for a
# 392 ms swing-up); this one swings up in ~0.55 s. Experiments growing from balancing to swing-up (2026-11-08,
# FINDINGS.md): 0.25 s balanced but left the arm unsettled, 0.5 s settled it and caught the pendulum from every
# start tried, hanging rest included. Cost: the PID tuner's scales (Bryson's rule [16]), periodic in both angles,
# plus the torque change per sample; no terminal ingredients [29]. IPOPT [14] to convergence every sample, exact
# Hessian. ipopt_warm: the options added from the second solve on, which starts from the previous solution and
# its multipliers; with a small initial barrier parameter IPOPT stays near that start (measured 2026-11-08: about
# 3 iterations per sample once the swing-up is under way, 10-65 in its first 0.2 s, against 24-88 without).
NMPC = dict(period=0.005,                                   # s, multiple of DT
            substeps=3,                                     # RK4 steps per period
            horizon=0.5,                                    # s, multiple of period
            max_pendulum_error=TUNE["max_pendulum_error"],  # rad
            max_arm_error=TUNE["max_arm_error"],            # rad
            max_torque=TUNE["max_torque"],                  # N m
            max_torque_change=0.025,                        # N m per sample (5 ms): the full range in about 10 ms
            friction_arm=EKF["friction_arm"], friction_pend=EKF["friction_pend"],
            ipopt=dict(tol=1e-7, max_iter=3000, print_level=0, sb="yes"),
            ipopt_warm=dict(warm_start_init_point="yes", mu_init=1e-8, warm_start_bound_push=1e-9,
                            warm_start_bound_frac=1e-9, warm_start_slack_bound_push=1e-9,
                            warm_start_slack_bound_frac=1e-9, warm_start_mult_bound_push=1e-9))

# Friction identification, step 1 (sysid.py): from rest, with no controller, the motor torque steps through
# `levels` log-spaced magnitudes per direction, run monotonically from -torque_max to +torque_max as in Quanser's own
# test [5], so each step starts near the previous steady state. Each level is held for `wait` s to settle, then the
# arm speed is measured from encoder counts over `average` s. At steady state the torque equals the arm's friction
# at that speed, and linear least squares fits viscous + Coulomb friction to the points.
SYSID = dict(torque_min=3e-4, torque_max=3e-2,  # N m: steady speeds from about 0.1 to 30 rad/s
             levels=20,                         # per direction
             wait=20.0, average=10.0)           # s, multiples of DT

# Friction identification, step 2a (sysid.py): the likelihood surface. One open-loop experiment from rest records
# the torques and the encoder readings: torque pulses of `pulse` s, each followed by `coast` s at zero, with
# amplitudes uniform in +-max_torque from a generator seeded by `seed`. The EKF (config.EKF, its arm friction
# viscous + Coulomb) runs over the record for each (b, tau_C) of a grid spanning the factors b_range x tau_range
# of FRICTION_ID, with the EKF's Q_c set to Q_c here; its negative log-likelihood is -sum_k log(cell probability_k) [25, sec. 12.1.1, eqs. (12.5)-(12.6)].
LIKELIHOOD = dict(duration=300.0, pulse=0.5, coast=0.5, max_torque=6e-3, seed=1,   # s, s, s, N m
                  Q_c=1e-7,                                                        # (N m)^2 s per joint, fixed
                  b_range=(0.5, 1.5), tau_range=(0.0, 2.5), points=7,              # factors of FRICTION_ID; per axis
                  truth_arm=None)   # None: the truth is FRICTION_ARM. DIAGNOSTIC ONLY, never a result: a known-answer
                                    # test with the truth inside the EKF's model class, e.g.
                                    # dict(FRICTION_ARM, breakaway=FRICTION_ARM["coulomb"]) (viscous + Coulomb)

SYSID_RUN = ["steady_state"]            # which of sysid.py's methods to run: "steady_state" (2 min), "likelihood_grid" (30 min)

# Experiments: x0 = [theta, alpha, theta_dot, alpha_dot]; alpha = 0 hanging down, +-pi upright.
# Each experiment names its controller and that controller's settings:
#   "open_loop": torque(t) [N m];   "pid": gains, a dict like PID_GAINS;
#   "swing_up": gains (the PID's) and swing_up, a dict like SWING_UP;   "nmpc": nmpc, a dict like NMPC.
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
    "nmpc_swing_up_from_rest": dict(x0=[0, 0, 0, 0], t_end=10.0, friction=True, controller="nmpc", nmpc=NMPC),
    **{f"nmpc_tap_{1000 * size:g}mNm": dict(                       # the swing_up_tap experiments, on the NMPC
        x0=[0, 0, 0, 0], t_end=10.0, friction=True, controller="nmpc", nmpc=NMPC,
        disturbance=lambda t, size=size: size if TAP["t"] <= t < TAP["t"] + TAP["duration"] else 0.0)
       for size in TAP["sizes"]},
}
EXPERIMENTS |= {f"{name}_ekf": dict(EXPERIMENTS[name], state="ekf")      # twins whose controller sees the EKF
                for name in ("pid_near_upright", "swing_up_from_rest", "nmpc_swing_up_from_rest",
                             *(f"{c}_tap_{1000 * size:g}mNm" for c in ("swing_up", "nmpc") for size in TAP["sizes"]))}
RUN = ["swing_up_from_rest", "nmpc_swing_up_from_rest"] # list(EXPERIMENTS)

# Comparisons: overlays of saved runs (results/compare/<name>.png), drawn by run.py once all of a comparison's runs
# have been saved. The energy swing-up + PID against the NMPC on the same tasks, with the true state and the EKF.
COMPARE = {f"{task}{state}": [f"{benchmark}{state}", f"nmpc_{task}{state}"]
           for task, benchmark in [("swing_up_from_rest", "swing_up_from_rest"),
                                   *((f"tap_{1000 * s:g}mNm", f"swing_up_tap_{1000 * s:g}mNm") for s in TAP["sizes"])]
           for state in ("", "_ekf")}

# Output
RESULTS_DIR = "results"
VIDEO_PX = 1080                         # square video, pixels per side
VIDEO_FPS = 30
TRAIL_S = 1.0                           # s of pendulum-tip trail in the video
PLOT_WINDOW_S = 5.0                     # s of history in the video's scrolling plots
