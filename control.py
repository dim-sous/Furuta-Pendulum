"""Controllers, chosen per experiment in config.EXPERIMENTS. Each returns controller(t, x) -> motor torque [N m],
called once per sample period."""
import casadi as ca
import numpy as np

GAINS = ("Kp_pendulum", "Ki_pendulum", "Kd_pendulum", "Kp_arm", "Ki_arm", "Kd_arm")   # order of the gain vector

# One sample of the two summed PIDs, written once as a CasADi function so that the simulation (pid below) and the
# tuner (tune.py) use the same law:
#   u = Kp_pendulum e + Ki_pendulum I - Kd_pendulum alpha_dot + Kp_arm e_arm + Ki_arm I_arm - Kd_arm theta_dot,
# with e = pi - alpha wrapped to [-pi, pi) (the shortest way up), e_arm = 0 - theta (not wrapped: whole turns
# count), and I, I_arm the forward-Euler sums of e dt and e_arm dt over the sample periods (Astrom & Murray 2021,
# sec. 11.5). The D terms act on the measured rates rather than on de/dt, so the 2 pi jump of the wrap never kicks
# them. pid_step(x, integrals, gains, dt) -> (u, integrals after the sample, errors [e, e_arm]).
_x, _integrals, _gains, _dt = ca.SX.sym("x", 4), ca.SX.sym("integrals", 2), ca.SX.sym("gains", 6), ca.SX.sym("dt")
_r = ca.fmod(np.pi - _x[1] + np.pi, 2 * np.pi)   # the wrap as NumPy's % computes it, so results match to the bit
_errors = ca.vertcat(ca.if_else(_r < 0, _r + 2 * np.pi, _r) - np.pi, -_x[0])
_u = (_gains[0] * _errors[0] + _gains[1] * _integrals[0] - _gains[2] * _x[3]
      + _gains[3] * _errors[1] + _gains[4] * _integrals[1] - _gains[5] * _x[2])
pid_step = ca.Function("pid_step", [_x, _integrals, _gains, _dt], [_u, _integrals + _errors * _dt, _errors])


def open_loop(torque):
    """Applies the prescribed signal torque(t) [N m] and ignores the state."""
    return lambda t, x: torque(t)


def pid(Kp_pendulum, Ki_pendulum, Kd_pendulum, Kp_arm, Ki_arm, Kd_arm, dt):
    """The two summed PIDs of pid_step, holding the pendulum upright and the arm at theta = 0; the controller keeps
    its two integrals between samples. With the arm gains at 0 this is the pendulum-only PID."""
    gains = [Kp_pendulum, Ki_pendulum, Kd_pendulum, Kp_arm, Ki_arm, Kd_arm]
    integrals = np.zeros(2)

    def controller(t, x):
        nonlocal integrals
        u, integrals, _ = pid_step(x, integrals, gains, dt)
        return float(u)

    return controller
