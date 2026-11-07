"""Runs the experiments listed in config.RUN: simulates each one, checks it, prints the key numbers and saves a
figure and a video per experiment to the results folder. A failed check is reported in the terminal and on the
plots, and the remaining experiments still run."""
import sys
from pathlib import Path

import casadi as ca
import numpy as np

import config as c
import control
import model
import viz


def simulate(x0, t_end, controller, with_friction, dt, substeps):
    """Simulates the pendulum from state x0 for t_end seconds.

    controller(t, x) gives the motor torque every sample period dt, held constant until the next sample
    (zero-order hold); RK4 integrates each period in `substeps` steps.
    Returns times t (n+1), states x (n+1, 4), torques u (n), and the cumulative motor work and
    friction loss (n+1 each), integrated alongside the state for the energy balance.
    """
    n = round(t_end / dt)
    if not np.isclose(n * dt, t_end, rtol=1e-12, atol=0):
        raise ValueError(f"t_end = {t_end} s is not a multiple of dt = {dt} s")
    f, P = model.dynamics(with_friction)
    step = model.rk4_step(f, P, dt, substeps)
    t = dt * np.arange(n + 1)
    x = np.empty((n + 1, 4))
    x[0] = x0
    u = np.empty(n)
    work = np.zeros((n + 1, 2))
    for k in range(n):
        u[k] = controller(t[k], x[k])
        x_next, dwork = step(x[k], u[k])
        x[k + 1] = np.asarray(x_next).ravel()
        work[k + 1] = work[k] + np.asarray(dwork).ravel()
    return t, x, u, work[:, 0], work[:, 1]


def swing_frequency():
    """Frequency [Hz] of small swings about hanging down with the arm free (linearised, no friction)."""
    f, _ = model.dynamics(with_friction=False)
    xs = ca.SX.sym("x", 4)
    A = np.asarray(ca.Function("A", [xs], [ca.jacobian(f(xs, 0), xs)])(np.zeros(4)))
    return np.max(np.abs(np.linalg.eigvals(A).imag)) / (2 * np.pi)


def main():
    out = Path(c.RESULTS_DIR)
    print(f"small-swing frequency about hanging down, arm free: {swing_frequency():.3f} Hz")
    failed_runs = []
    for name in c.RUN:
        e = c.EXPERIMENTS[name]
        if e["controller"] == "open_loop":
            controller = control.open_loop(e["torque"])
            label = "open loop: prescribed torque"
        elif e["controller"] == "pid":
            controller = control.pid(**e["gains"], dt=c.DT)
            g = e["gains"]
            label = (f"PID, pendulum: Kp = {g['Kp_pendulum']:g}, Ki = {g['Ki_pendulum']:g}, Kd = {g['Kd_pendulum']:g};   "
                     f"arm: Kp = {g['Kp_arm']:g}, Ki = {g['Ki_arm']:g}, Kd = {g['Kd_arm']:g}")
        else:
            raise ValueError(f"{name}: unknown controller {e['controller']!r}")
        t, x, u, motor_work, friction_loss = simulate(e["x0"], e["t_end"], controller, e["friction"],
                                                      c.DT, c.RK4_SUBSTEPS)
        # The checks report instead of stopping, so that a run that went wrong can still be looked at.
        failed_checks = []
        finite = np.all(np.isfinite(x), axis=1)
        if not finite.all():  # e.g. a closed loop that is unstable at this sample rate: keep what came before
            k = np.argmin(finite)
            failed_checks.append(f"the state blew up at t = {t[k]:.3f} s; the results end there")
            t, x, u, motor_work, friction_loss = t[:k], x[:k], u[:k - 1], motor_work[:k], friction_loss[:k]
        E = np.asarray(model.energy.map(len(t))(x.T)).ravel()
        residual = np.max(np.abs(E - E[0] - motor_work + friction_loss))
        # After a blow-up the last samples are too large for these two checks to mean anything.
        if finite.all() and not residual <= c.ENERGY_TOL:  # written as 'not <=' so that a NaN fails too
            failed_checks.append(f"energy balance off by {residual:.1e} J (tolerance {c.ENERGY_TOL:.0e} J): "
                                 "RK4 too coarse for this run, try a larger RK4_SUBSTEPS")
        if finite.all() and np.any(np.diff(friction_loss) < 0):  # each RK4 increment is a sum of tau_f * qd >= 0
            failed_checks.append("friction added energy instead of removing it: the friction law is wrong")
        print(f"{name}: E(0) = {E[0]:.4e} J, motor work = {motor_work[-1]:+.4e} J, "
              f"friction loss = {friction_loss[-1]:.4e} J, energy balance residual = {residual:.1e} J")
        for message in failed_checks:
            print(f"  CHECK FAILED: {message}")
        title = (f"{name.replace('_', ' ')}: θ0 = {np.rad2deg(e['x0'][0]):.0f}°, α0 = {np.rad2deg(e['x0'][1]):.0f}°, "
                 f"friction {'on' if e['friction'] else 'off'}")
        folder = out / e["controller"]                     # one folder per controller
        folder.mkdir(parents=True, exist_ok=True)
        viz.figure(folder / f"{name}.png", title, label, t, x, u, E, motor_work, friction_loss, residual, failed_checks)
        viz.video(folder / f"{name}.mp4", title, label, t, x, u, failed_checks)
        if failed_checks:
            failed_runs.append(name)
    if failed_runs:
        sys.exit(f"checks failed in {', '.join(failed_runs)}: see the messages above and the plots")


if __name__ == "__main__":
    main()
