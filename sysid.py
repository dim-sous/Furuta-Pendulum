"""Identifies the arm's friction from encoder data, by the methods named in config.SYSID_RUN. Sources [n] are the
References in METHODS.md.

steady_state (step 1, config.SYSID): a staircase of constant motor torques. At a steady arm speed with the
pendulum settled, the arm and the pendulum turn as one rigid body at constant speed, so the arm's equation of
motion reduces to u = friction(omega): each torque level gives one point of the friction curve. The speed comes
from the arm's encoder alone, as the change in its reading over the averaging window divided by the window's
length (error at most one count per window). Linear least squares [26, sec. 7.3] fits viscous + Coulomb friction,
friction(omega) = b omega + tau_C sign(omega), to the points; the simulated truth has the smooth Stribeck law
(config.FRICTION_ARM), which that model cannot follow below about 0.4 rad/s. Prints the result for
config.FRICTION_ID.

likelihood_grid (step 2a, config.LIKELIHOOD): the likelihood surface of (b, tau_C) on dynamic data. One open-loop
experiment of torque pulses records the torques and the encoder readings; the EKF (estimator.py), its arm friction
viscous + Coulomb with the grid's (b, tau_C), runs over the record, and its negative log-likelihood is the
prediction-error decomposition [25, sec. 12.1.1, eqs. (12.5)-(12.6)] with the reading "the angles lie in this
cell" as the measurement: -sum_k log P(cell_k | readings before k) = -sum_k log(cell probability_k).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import config as c
import control
import estimator
import model
import run
import viz


def staircase(torque_min, torque_max, levels):
    """The torque levels [N m]: `levels` log-spaced magnitudes per direction, ordered -torque_max ... -torque_min,
    +torque_min ... +torque_max."""
    magnitudes = np.geomspace(torque_min, torque_max, levels)
    return np.concatenate([-magnitudes[::-1], magnitudes])


def least_squares(omega, torque):
    """Fits torque = b omega + tau_C sign(omega) by linear least squares [26, sec. 7.3]: theta = (Phi^T Phi)^-1
    Phi^T U, rows of Phi = [omega_i, sign(omega_i)]. Also returns the parameter covariance lambda (Phi^T Phi)^-1,
    lambda = |U - Phi theta|^2 / (N - 2) [26, App. II], valid only if the residuals are white noise, which a
    deterministic model error is not."""
    Phi = np.column_stack([omega, np.sign(omega)])
    theta = np.linalg.solve(Phi.T @ Phi, Phi.T @ torque)
    residual = torque - Phi @ theta
    covariance = residual @ residual / (len(torque) - 2) * np.linalg.inv(Phi.T @ Phi)
    return theta, covariance


def steady_state(folder):
    """Step 1: the torque staircase, the least-squares fit, its figure; returns the failed checks' messages."""
    S = c.SYSID
    torques = staircase(S["torque_min"], S["torque_max"], S["levels"])
    n_wait, n_average = round(S["wait"] / c.DT), round(S["average"] / c.DT)
    n_hold = n_wait + n_average
    controller = control.open_loop(lambda t: torques[round(t / c.DT) // n_hold])
    t, x, u, motor_work, friction_loss, d, tap_work, _ = run.simulate(
        [0, 0, 0, 0], len(torques) * n_hold * c.DT, controller, True, c.DT, c.RK4_SUBSTEPS)

    if not np.all(np.isfinite(x)):
        return ["steady_state: the state blew up, no friction curve to fit"]
    _, residual, failed_checks = run.energy_checks(x, motor_work, tap_work, friction_loss)

    # Speed of each level from the arm's encoder over its averaging window; and the same over the window's two
    # halves, which agree within their resolution (one count per half, each way) if the arm was at steady state.
    theta = np.array([run.encoder(xk)[0] for xk in x])
    start = np.arange(len(torques)) * n_hold + n_wait
    end, middle = start + n_average, start + n_average // 2
    omega = (theta[end] - theta[start]) / S["average"]
    halves = (theta[end] - theta[middle] - (theta[middle] - theta[start])) / (S["average"] / 2)
    bound = 2 * c.ENCODER_RESOLUTION / (S["average"] / 2)
    unsteady = np.flatnonzero(np.abs(halves) > bound)
    for i in unsteady:
        failed_checks.append(f"not at steady state at {torques[i]:+.2e} N m: the speed changed by {halves[i]:.1e} rad/s "
                             f"between the averaging window's halves (resolution {bound:.1e}); lengthen SYSID['wait']")

    (b, tau_c), covariance = least_squares(omega, torques)
    std = np.sqrt(np.diag(covariance))
    truth = c.FRICTION_ARM
    print(f"steady_state: {len(torques)} torque levels, each held {S['wait']:g} + {S['average']:g} s; energy balance "
          f"residual {residual:.1e} J")
    print("  torque [N m]   speed [rad/s]   fit [N m]    residual [N m]")
    for tau, w in zip(torques, omega):
        fit = b * w + tau_c * np.sign(w)
        print(f"  {tau:+.3e}    {w:+9.4f}      {fit:+.3e}   {tau - fit:+.2e}")
    print(f"\nviscous b     = {b:.4e} +- {std[0]:.1e} N m s/rad   (truth {truth['viscous']:.4e}: "
          f"{b / truth['viscous'] - 1:+.1%})")
    print(f"Coulomb tau_C = {tau_c:.4e} +- {std[1]:.1e} N m       (truth {truth['coulomb']:.4e}: "
          f"{tau_c / truth['coulomb'] - 1:+.1%}; the truth adds a Stribeck peak to {truth['breakaway']:.2e} N m)")
    print("The +- are least squares' standard errors, which assume white residuals; here the residuals are mostly "
          "model error, so the actual error can be larger.")
    print(f"Paste into config.py:\nFRICTION_ID = dict(viscous={b:.5g}, coulomb={tau_c:.5g})\n")
    viz.friction_figure(folder / "friction_staircase.png", t, u, x, torques, omega, (start + end) // 2, b, tau_c,
                        lambda w: float(model.friction(w, **truth)), failed_checks)
    return failed_checks


def viscous_coulomb(b, tau_c):
    """Viscous + Coulomb friction as parameters of model.friction: with breakaway = Coulomb its Stribeck term
    vanishes, leaving b w + tau_C tanh(w / (w_breakaway / 10)), the sign function smoothed over 0.01 rad/s."""
    return dict(viscous=b, coulomb=tau_c, breakaway=tau_c, w_breakaway=c.FRICTION_ARM["w_breakaway"])


def negative_log_likelihood(b, tau_c, y, u):
    """-sum_k log(cell probability_k) of the EKF (config.EKF, but Q_c = config.LIKELIHOOD["Q_c"]) with arm friction
    viscous_coulomb(b, tau_c), run over the encoder
    readings y (n, 2) and the torques u (n); the first sample, the filter's start, has no update. Returns
    (NLL, None), or (inf, k) if at sample k the filter gave the read cell no positive probability: the filter
    broke there, and the record says nothing beyond it."""
    E = c.EKF
    estimate = estimator.ekf(viscous_coulomb(b, tau_c), E["friction_pend"], [c.LIKELIHOOD["Q_c"]] * 2, E["P0_std"],
                             c.ENCODER_RESOLUTION, c.DT, c.RK4_SUBSTEPS)
    estimate(y[0], 0.0)
    nll = 0.0
    for k in range(1, len(y)):
        _, _, cell_prob = estimate(y[k], u[k - 1])
        if not cell_prob > 0:                                         # written so that a NaN fails too
            return np.inf, k
        nll -= np.log(cell_prob)
    return nll, None


def likelihood_grid(folder):
    """Step 2a: one open-loop experiment of torque pulses, the EKF's negative log-likelihood on a (b, tau_C) grid
    around config.FRICTION_ID (grid points in parallel processes), its figure; returns the failed checks'
    messages."""
    L = c.LIKELIHOOD
    n = round(L["duration"] / c.DT)
    n_pulse, n_cycle = round(L["pulse"] / c.DT), round((L["pulse"] + L["coast"]) / c.DT)
    amplitudes = np.random.default_rng(L["seed"]).uniform(-L["max_torque"], L["max_torque"], n // n_cycle + 1)
    torque = lambda t: amplitudes[round(t / c.DT) // n_cycle] if round(t / c.DT) % n_cycle < n_pulse else 0.0
    truth = L["truth_arm"] or c.FRICTION_ARM
    diagnostic = L["truth_arm"] is not None
    t, x, u, motor_work, friction_loss, d, tap_work, _ = run.simulate(
        [0, 0, 0, 0], L["duration"], control.open_loop(torque), True, c.DT, c.RK4_SUBSTEPS, friction_arm=truth)
    if not np.all(np.isfinite(x)):
        return ["likelihood_grid: the state blew up, no record to fit"]
    _, residual, failed_checks = run.energy_checks(x, motor_work, tap_work, friction_loss)
    y = np.array([run.encoder(xk) for xk in x[:-1]])                  # the readings at the n control samples

    b_grid = c.FRICTION_ID["viscous"] * np.linspace(*L["b_range"], L["points"])
    tau_grid = c.FRICTION_ID["coulomb"] * np.linspace(*L["tau_range"], L["points"])
    pairs = [(b, tau) for b in b_grid for tau in tau_grid]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(negative_log_likelihood, *zip(*pairs), [y] * len(pairs), [u] * len(pairs)))
    nll = np.array([r[0] for r in results]).reshape(len(b_grid), len(tau_grid))
    broken = [(b, tau, k) for (b, tau), (_, k) in zip(pairs, results) if k is not None]

    speed = np.abs(x[:, 2])
    if diagnostic:
        print(f"DIAGNOSTIC (not a result): the truth's arm friction is {truth}, inside the EKF's model class")
    print(f"likelihood_grid: {L['duration']:g} s of torque pulses (|u| <= {L['max_torque']:g} N m); arm speed up to "
          f"{speed.max():.2f} rad/s, below 0.4 rad/s {np.mean(speed < 0.4):.0%} of the time; energy balance residual "
          f"{residual:.1e} J; EKF Q_c = {L['Q_c']:g} (N m)^2 s")
    finite = np.isfinite(nll)
    best = np.unravel_index(np.argmin(np.where(finite, nll, np.inf)), nll.shape) if finite.any() else None
    print("  NLL - min over the grid; rows b [N m s/rad], columns tau_C [N m]")
    print("  " + " " * 12 + "".join(f"{tau:>11.3e}" for tau in tau_grid))
    for i, b in enumerate(b_grid):
        row = "".join(f"{nll[i, j] - nll[best]:>11.2f}" if finite[i, j] else f"{'broke':>11}" for j in range(len(tau_grid)))
        print(f"  {b:.4e}  {row}")
    if best is not None:
        print(f"grid minimum: b = {b_grid[best[0]]:.4e}, tau_C = {tau_grid[best[1]]:.4e} (step 1: "
              f"{c.FRICTION_ID['viscous']:.4e}, {c.FRICTION_ID['coulomb']:.4e}; truth: {truth['viscous']:.4e}, "
              f"{truth['coulomb']:.4e}" + (")" if diagnostic else " plus a Stribeck peak)"))
    for b, tau, k in broken:
        failed_checks.append(f"the EKF broke at b = {b:.3e}, tau_C = {tau:.3e}: at t = {t[k]:.3f} s it gave the read "
                             "cell no positive probability (its belief was wrong there)")
    name = "likelihood_grid_diagnostic.png" if diagnostic else "likelihood_grid.png"
    viz.likelihood_figure(folder / name, b_grid, tau_grid, nll, (c.FRICTION_ID["viscous"], c.FRICTION_ID["coulomb"]),
                          (truth["viscous"], truth["coulomb"]), L, failed_checks, diagnostic)
    return failed_checks


def main():
    folder = Path(c.RESULTS_DIR) / "sysid"
    folder.mkdir(parents=True, exist_ok=True)
    methods = dict(steady_state=steady_state, likelihood_grid=likelihood_grid)
    failed_checks = []
    for name in c.SYSID_RUN:
        if name not in methods:
            raise ValueError(f"unknown sysid method {name!r}")
        failed_checks += methods[name](folder)
    for message in failed_checks:
        print(f"  CHECK FAILED: {message}")
    if failed_checks:
        sys.exit("checks failed: see the messages above and the figures")


if __name__ == "__main__":
    main()
