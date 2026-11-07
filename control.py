"""Controllers, chosen per experiment in config.EXPERIMENTS. Each returns controller(t, x) -> motor torque [N m],
called once per sample period."""
import casadi as ca
import numpy as np

import config as c

GAINS = ("Kp_pendulum", "Ki_pendulum", "Kd_pendulum", "Kp_arm", "Ki_arm", "Kd_arm")   # order of the gain vector


def _wrap(a):
    """a wrapped to [-pi, pi), computed as NumPy's (a + pi) % (2 pi) - pi does, so results match to the bit."""
    r = ca.fmod(a + np.pi, 2 * np.pi)
    return ca.if_else(r < 0, r + 2 * np.pi, r) - np.pi


# One sample of the two summed PIDs, written once as a CasADi function so that the simulation (pid below) and the
# tuner (tune.py) use the same law:
#   v = Kp_pendulum e - Kd_pendulum alpha_dot + Kp_arm e_arm - Kd_arm theta_dot + I,    u = sat(v) to +-MAX_TORQUE,
# with e = pi - alpha and e_arm = 0 - theta, both wrapped to [-pi, pi) (the shortest way to the target, so a whole
# turn of the arm counts as none). I is the integral term of both loops in one state, in N m: the forward-Euler sum
# of dt (Ki_pendulum e + Ki_arm e_arm) (Astrom & Murray 2021, sec. 11.5), with conditional integration against
# windup (Astrom & Hagglund 2006, sec. 3.5): a sample's increment is dropped when the torque is capped and the
# increment would push it further into the cap. The D terms act on the measured rates rather than on de/dt, so
# the 2 pi jumps of the wraps never kick them.
# pid_step(x, integral, gains, dt, cap) -> (u, integral after the sample, errors [e, e_arm]); cap is MAX_TORQUE in
# the experiments (inf disables it).
_x, _integral, _gains = ca.SX.sym("x", 4), ca.SX.sym("integral"), ca.SX.sym("gains", 6)
_dt, _cap = ca.SX.sym("dt"), ca.SX.sym("cap")
_errors = ca.vertcat(_wrap(np.pi - _x[1]), _wrap(-_x[0]))
_v = (_gains[0] * _errors[0] - _gains[2] * _x[3] + _gains[3] * _errors[1] - _gains[5] * _x[2] + _integral)
_u = ca.fmin(ca.fmax(_v, -_cap), _cap)
_increment = _dt * (_gains[1] * _errors[0] + _gains[4] * _errors[1])
_winding_up = ca.logic_or(ca.logic_and(_v > _cap, _increment > 0), ca.logic_and(_v < -_cap, _increment < 0))
pid_step = ca.Function("pid_step", [_x, _integral, _gains, _dt, _cap],
                       [_u, _integral + ca.if_else(_winding_up, 0, _increment), _errors])


def open_loop(torque):
    """Applies the prescribed signal torque(t) [N m] and ignores the state."""
    return lambda t, x: torque(t)


def pid(Kp_pendulum, Ki_pendulum, Kd_pendulum, Kp_arm, Ki_arm, Kd_arm, dt):
    """The two summed PIDs of pid_step, holding the pendulum upright and the arm at theta = 0; the controller keeps
    its integral between samples. With the arm gains at 0 this is the pendulum-only PID."""
    gains = [Kp_pendulum, Ki_pendulum, Kd_pendulum, Kp_arm, Ki_arm, Kd_arm]
    integral = 0.0

    def controller(t, x):
        nonlocal integral
        u, integral, _ = pid_step(x, integral, gains, dt, c.MAX_TORQUE)
        return float(u)

    return controller


def swing_up(gains, k, enter, leave, dt):
    """Energy-control swing-up (Astrom & Furuta 2000) that hands over to the PID near upright.

    The PID with `gains` takes over when the pendulum comes within `enter` [rad] of upright, and hands back only
    when it falls beyond `leave` [rad] (hysteresis); its integral restarts from 0 at each hand-over. Swinging up,
    u = -sat(k (E_up - E) sign(alpha_dot cos alpha)), saturated at config.MAX_TORQUE, where
    E = J_pivot alpha_dot^2 / 2 + m_p g d_p (1 - cos alpha) is the pendulum's energy about its pivot (0 hanging,
    E_up = 2 m_p g d_p upright) and k [N m/J] the gain. Since dE/dt = -m_p L_a d_p cos(alpha) alpha_dot theta_ddot
    plus arm-speed terms, this torque pumps energy in below E_up and out above it. At exact rest sign(0) is taken
    as +1, so the swing-up starts by itself.
    """
    j_pivot = c.J_PEND[0] + c.M_PEND * c.D_PEND**2   # the pendulum as config gives it when the controller is built
    mgd = c.M_PEND * c.G * c.D_PEND
    e_up = 2 * mgd
    balance = None

    def controller(t, x):
        nonlocal balance
        error = abs((np.pi - x[1] + np.pi) % (2 * np.pi) - np.pi)
        if error < enter or (balance is not None and error < leave):
            if balance is None:
                balance = pid(**gains, dt=dt)
            return balance(t, x)
        balance = None
        energy = 0.5 * j_pivot * x[3] ** 2 + mgd * (1 - np.cos(x[1]))
        direction = np.sign(x[3] * np.cos(x[1])) or 1.0
        return float(np.clip(-k * (e_up - energy) * direction, -c.MAX_TORQUE, c.MAX_TORQUE))

    return controller
