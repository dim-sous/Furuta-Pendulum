"""Identifies the arm's friction from a staircase of constant motor torques (config.SYSID), and prints the
parameters. Step 1 of the friction identification: steady states only.

At a steady arm speed with the pendulum settled, the arm and the pendulum turn as one rigid body at constant
speed, so the arm's equation of motion reduces to u = friction(omega): each torque level gives one point of the
friction curve. The speed comes from the arm's encoder alone, as the change in its reading over the averaging
window divided by the window's length (error at most one count per window). Linear least squares [26, sec. 7.3]
fits viscous + Coulomb friction, friction(omega) = b omega + tau_C sign(omega), to the points; the simulated truth
has the smooth Stribeck law (config.FRICTION_ARM), which that model cannot follow below about 0.4 rad/s. Sources
[n] are the References in METHODS.md.
"""
import sys
from pathlib import Path

import numpy as np

import config as c
import control
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


def main():
    S = c.SYSID
    torques = staircase(S["torque_min"], S["torque_max"], S["levels"])
    n_wait, n_average = round(S["wait"] / c.DT), round(S["average"] / c.DT)
    n_hold = n_wait + n_average
    controller = control.open_loop(lambda t: torques[round(t / c.DT) // n_hold])
    t, x, u, motor_work, friction_loss, d, tap_work, _ = run.simulate(
        [0, 0, 0, 0], len(torques) * n_hold * c.DT, controller, True, c.DT, c.RK4_SUBSTEPS)

    if not np.all(np.isfinite(x)):
        sys.exit("the state blew up: no friction curve to fit")
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
    print(f"{len(torques)} torque levels, each held {S['wait']:g} + {S['average']:g} s; energy balance residual "
          f"{residual:.1e} J")
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
    for message in failed_checks:
        print(f"  CHECK FAILED: {message}")

    folder = Path(c.RESULTS_DIR) / "sysid"
    folder.mkdir(parents=True, exist_ok=True)
    viz.friction_figure(folder / "friction_staircase.png", t, u, x, torques, omega, (start + end) // 2, b, tau_c,
                        lambda w: float(model.friction(w, **truth)), failed_checks)
    if failed_checks:
        sys.exit("checks failed: see the messages above and the figure")


if __name__ == "__main__":
    main()
