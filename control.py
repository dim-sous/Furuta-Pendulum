"""Controllers, chosen per experiment in config.EXPERIMENTS. Each returns controller(t, x) -> motor torque [N m],
called once per sample period."""
import numpy as np


def open_loop(torque):
    """Applies the prescribed signal torque(t) [N m] and ignores the state."""
    return lambda t, x: torque(t)


def pid(Kp_pendulum, Ki_pendulum, Kd_pendulum, Kp_arm, Ki_arm, Kd_arm, dt):
    """Two PIDs, summed: one holds the pendulum upright, the other holds the arm at theta = 0.

    u = Kp e + Ki I - Kd alpha_dot + Kp_arm e_arm + Ki_arm I_arm - Kd_arm theta_dot, with e = pi - alpha wrapped
    to [-pi, pi) (the shortest way up), e_arm = 0 - theta (not wrapped: whole turns count), and I, I_arm the
    forward-Euler sums of e dt and e_arm dt over the sample periods of length dt (Astrom & Murray 2021,
    sec. 11.5). The D terms act on the measured rates rather than on de/dt, so the 2 pi jump of the wrap never
    kicks them. With the arm gains at 0 this is the pendulum-only PID.
    """
    integral = integral_arm = 0.0

    def controller(t, x):
        nonlocal integral, integral_arm
        e = (np.pi - x[1] + np.pi) % (2 * np.pi) - np.pi
        e_arm = -x[0]
        u = Kp_pendulum * e + Ki_pendulum * integral - Kd_pendulum * x[3] + Kp_arm * e_arm + Ki_arm * integral_arm - Kd_arm * x[2]
        integral += e * dt
        integral_arm += e_arm * dt
        return u

    return controller
