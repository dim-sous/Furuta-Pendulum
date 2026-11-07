"""Runs the experiments listed in config.RUN: simulates each one, checks it, prints the key numbers and saves a
figure and a video per experiment to the results folder. A failed check is reported in the terminal and on the
plots, and the remaining experiments still run."""
import sys
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.stats import chi2

import config as c
import control
import estimator
import model
import viz


def encoder(x):
    """Encoder readings [theta, alpha] of state x: each angle rounded to whole counts of c.ENCODER_RESOLUTION."""
    return c.ENCODER_RESOLUTION * np.round(x[:2] / c.ENCODER_RESOLUTION)


def simulate(x0, t_end, controller, with_friction, dt, substeps, disturbance=lambda t: 0.0, estimate=None):
    """Simulates the pendulum from state x0 for t_end seconds.

    controller(t, x) gives the motor torque and disturbance(t) the torque on the pendulum's joint, both every
    sample period dt and held constant until the next sample (zero-order hold); RK4 integrates each period in
    `substeps` steps. The controller sees the true state, or, given an estimator estimate(y, u_prev) ->
    (x_hat, P, cell_prob) (estimator.py), the estimate made from the encoder readings y.
    Returns times t (n+1), states x (n+1, 4), motor torques u (n), the cumulative motor work and friction loss
    (n+1 each), disturbance torques d (n) and the cumulative disturbance work (n+1), the works integrated
    alongside the state for the energy balance; and, with an estimator, a dict of its estimates x_hat (n, 4),
    covariances P (n, 4, 4) and the probabilities of the read cells (n) at each sample, else None.
    """
    n = round(t_end / dt)
    if not np.isclose(n * dt, t_end, rtol=1e-12, atol=0):
        raise ValueError(f"t_end = {t_end} s is not a multiple of dt = {dt} s")
    f, P = model.dynamics(with_friction)
    step = model.rk4_step(f, P, dt, substeps)
    t = dt * np.arange(n + 1)
    x = np.empty((n + 1, 4))
    x[0] = x0
    u, d = np.empty(n), np.empty(n)
    work = np.zeros((n + 1, 3))
    x_hat, P, cell_prob = np.empty((n, 4)), np.empty((n, 4, 4)), np.empty(n)
    for k in range(n):
        if estimate:
            x_hat[k], P[k], cell_prob[k] = estimate(encoder(x[k]), u[k - 1] if k else 0.0)
        u[k] = controller(t[k], x_hat[k] if estimate else x[k])
        d[k] = disturbance(t[k])
        x_next, dwork = step(x[k], u[k], d[k])
        x[k + 1] = np.asarray(x_next).ravel()
        work[k + 1] = work[k] + np.asarray(dwork).ravel()
    estimates = dict(x_hat=x_hat, P=P, cell_prob=cell_prob) if estimate else None
    return t, x, u, work[:, 0], work[:, 1], d, work[:, 2], estimates


def consistency(x, estimates):
    """The estimates with their error x_hat - x against the true states x added, and the consistency test of
    [20, sec. 5.4]: NEES = e^T P^-1 e at each sample, chi-square with 4 degrees of freedom if the filter's
    covariance P is right; its two-sided 95% bounds, and the fraction of samples inside them, from the second
    sample (the first has no update). A diagnostic, never a failed check. NEES is NaN where P is not a finite,
    positive-definite covariance (a diverged filter, which the surprise check reports), and such samples count
    as outside."""
    error = estimates["x_hat"] - x
    valid = np.all(np.isfinite(estimates["P"]), axis=(1, 2))
    valid[valid] = np.all(np.linalg.eigvalsh(estimates["P"][valid]) > 0, axis=1)
    nees = np.full(len(error), np.nan)
    e, P = error[valid], estimates["P"][valid]
    nees[valid] = np.einsum("ki,ki->k", e, np.linalg.solve(P, e[..., None])[..., 0])
    nees_bounds = chi2.ppf([0.025, 0.975], df=error.shape[1])
    nees_inside = np.mean((nees_bounds[0] <= nees[1:]) & (nees[1:] <= nees_bounds[1]))
    return dict(estimates, error=error, nees=nees, nees_bounds=nees_bounds, nees_inside=nees_inside)


def swing_frequency():
    """Frequency [Hz] of small swings about hanging down with the arm free (linearised, no friction)."""
    f, _ = model.dynamics(with_friction=False)
    xs = ca.SX.sym("x", 4)
    A = np.asarray(ca.Function("A", [xs], [ca.jacobian(f(xs, 0, 0), xs)])(np.zeros(4)))
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
        elif e["controller"] == "swing_up":
            s = e["swing_up"]
            controller = control.swing_up(e["gains"], **s, dt=c.DT)
            label = (f"energy swing-up, k = {s['k']:g} N m/J; PID from {np.rad2deg(s['enter']):g}° of upright, "
                     f"back beyond {np.rad2deg(s['leave']):g}°; |u| ≤ {c.MAX_TORQUE:g} N m")
        else:
            raise ValueError(f"{name}: unknown controller {e['controller']!r}")
        state = e.get("state", "true")
        if state == "ekf":
            settings = c.EKF
            estimate = estimator.ekf(settings["friction_arm"], settings["friction_pend"], settings["Q_c"],
                                     settings["P0_std"], c.ENCODER_RESOLUTION, c.DT, c.RK4_SUBSTEPS)
        elif state == "true":
            estimate = None
        else:
            raise ValueError(f"{name}: unknown state {state!r}")
        t, x, u, motor_work, friction_loss, d, tap_work, estimates = simulate(
            e["x0"], e["t_end"], controller, e["friction"], c.DT, c.RK4_SUBSTEPS,
            e.get("disturbance", lambda t: 0.0), estimate)
        # The checks report instead of stopping, so that a run that went wrong can still be looked at.
        failed_checks = []
        finite = np.all(np.isfinite(x), axis=1)
        if not finite.all():  # e.g. a closed loop that is unstable at this sample rate: keep what came before
            k = np.argmin(finite)
            failed_checks.append(f"the state blew up at t = {t[k]:.3f} s; the results end there")
            t, x, u, motor_work, friction_loss = t[:k], x[:k], u[:k - 1], motor_work[:k], friction_loss[:k]
            d, tap_work = d[:k - 1], tap_work[:k]
            if estimates:
                estimates = {key: value[:k - 1] for key, value in estimates.items()}
        E = np.asarray(model.energy.map(len(t))(x.T)).ravel()
        residual = np.max(np.abs(E - E[0] - motor_work - tap_work + friction_loss))
        # After a blow-up the last samples are too large for these two checks to mean anything.
        if finite.all() and not residual <= c.ENERGY_TOL:  # written as 'not <=' so that a NaN fails too
            failed_checks.append(f"energy balance off by {residual:.1e} J (tolerance {c.ENERGY_TOL:.0e} J): "
                                 "RK4 too coarse for this run, try a larger RK4_SUBSTEPS")
        if finite.all() and np.any(np.diff(friction_loss) < 0):  # each RK4 increment is a sum of tau_f * qd >= 0
            failed_checks.append("friction added energy instead of removing it: the friction law is wrong")
        if estimates:
            cell_prob = estimates["cell_prob"][1:]                  # the first sample has no update
            if not np.all(cell_prob >= c.EKF["min_cell_prob"]):    # written so that a NaN fails too
                k = 1 + np.argmax(~(cell_prob >= c.EKF["min_cell_prob"]))
                failed_checks.append(
                    f"EKF surprised: at t = {t[k]:.3f} s a reading's cell had probability {estimates['cell_prob'][k]:.1e} "
                    f"under the filter's belief (threshold {c.EKF['min_cell_prob']:.0e}); its belief was wrong there, "
                    "and the update loses accuracy")
        print(f"{name}: E(0) = {E[0]:.4e} J, motor work = {motor_work[-1]:+.4e} J, "
              + (f"tap work = {tap_work[-1]:+.4e} J, " if d.any() else "")
              + f"friction loss = {friction_loss[-1]:.4e} J, energy balance residual = {residual:.1e} J")
        for message in failed_checks:
            print(f"  CHECK FAILED: {message}")
        if estimates:
            estimates = consistency(x[:len(u)], estimates)
            rms = np.rad2deg(np.sqrt(np.mean(estimates["error"]**2, axis=0)))
            print(f"  EKF: RMS error θ {rms[0]:.3f}°, α {rms[1]:.3f}°, θ̇ {rms[2]:.2f}°/s, α̇ {rms[3]:.2f}°/s; "
                  f"NEES inside its 95% bounds {estimates['nees_inside']:.0%} of samples; "
                  f"smallest cell probability {np.min(estimates['cell_prob'][1:]):.1e}")
        title = (f"{name.replace('_', ' ')}: θ0 = {np.rad2deg(e['x0'][0]):.0f}°, α0 = {np.rad2deg(e['x0'][1]):.0f}°, "
                 f"friction {'on' if e['friction'] else 'off'}")
        folder = out / e["controller"]                     # one folder per controller
        folder.mkdir(parents=True, exist_ok=True)
        viz.figure(folder / f"{name}.png", title, label, t, x, u, d, E, motor_work, tap_work, friction_loss, residual,
                   failed_checks, estimates)
        viz.video(folder / f"{name}.mp4", title, label, t, x, u, d, failed_checks)
        if failed_checks:
            failed_runs.append(name)
    if failed_runs:
        sys.exit(f"checks failed in {', '.join(failed_runs)}: see the messages above and the plots")


if __name__ == "__main__":
    main()
