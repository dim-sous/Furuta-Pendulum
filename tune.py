"""Tunes the gains of the two summed PIDs (control.pid_step) by optimization, and prints them for config.PID_GAINS.

The closed loop (plant, RK4 sub-steps, torque held over each sample, PID) is rolled out in CasADi from each start
in config.TUNE, so the cost has exact derivatives [2]. IPOPT [14] with a limited-memory BFGS Hessian [15] minimizes,
summed over the starts,

    J = sum_k [ (e_k / max_pendulum_error)^2 + (e_arm,k / max_arm_error)^2 + (u_k / max_torque)^2 ] dt,

an integral of squared errors and torque, each divided by its largest acceptable value (Bryson's rule [16]).
The unknowns are the gains divided by their first guess (config.PID_GAINS), so all six share one scale, and each
is bounded to keep the first guess's sign (zero allowed). Starts are added one at a time, each stage starting from
the previous one's gains. Sources [n] are the References in README.md.

The torque cap is left out of the tuning loop (TUNE["cap"] = inf): with config.MAX_TORQUE in the loop the tuner
returns meaningless gains (it chases bang-bang control). The tuned gains are then checked with the cap, as the
experiments run them.
"""
import tempfile

import casadi as ca
import numpy as np

import config as c
import control
import model
import run


def rollout(jit_dir):
    """CasADi function (z0, gains) -> z after the tuning horizon, z = [x (4), integral (1), accumulated cost].
    One sample is compiled to C (needs gcc): about 30x faster than CasADi's interpreter."""
    T = c.TUNE
    n = round(T["horizon"] / c.DT)
    if not np.isclose(n * c.DT, T["horizon"], rtol=1e-12, atol=0):
        raise ValueError(f"TUNE horizon = {T['horizon']} s is not a multiple of DT = {c.DT} s")
    f, P = model.dynamics(T["friction"])
    step = model.rk4_step(f, P, c.DT, c.RK4_SUBSTEPS)
    z, gains = ca.SX.sym("z", 6), ca.SX.sym("gains", 6)
    u, integral, errors = control.pid_step(z[:4], z[4], gains, c.DT, T["cap"])
    cost = ((errors[0] / T["max_pendulum_error"]) ** 2 + (errors[1] / T["max_arm_error"]) ** 2
            + (u / T["max_torque"]) ** 2) * c.DT
    sample = ca.Function("sample", [z, gains], [ca.vertcat(step(z[:4], u, 0)[0], integral, z[5] + cost)],
                         {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"], "directory": jit_dir}})
    return sample.fold(n)


def main():
    T = c.TUNE
    first_guess = np.array([c.PID_GAINS[name] for name in control.GAINS])
    if not np.all(first_guess):
        raise ValueError("every gain in config.PID_GAINS must be nonzero: it sets the gain's scale and sign")
    q = ca.MX.sym("q", 6)                                   # gain / first guess: 1 at the start, >= 0 by sign
    gains = ca.DM(first_guess) * q
    q_opt = np.ones(6)
    with tempfile.TemporaryDirectory() as jit_dir:   # the compiler fails on the space in this project's path
        roll = rollout(jit_dir + "/")
        for n in range(1, len(T["starts"]) + 1):
            z0 = ca.DM(np.vstack([np.zeros(n), T["starts"][:n], np.zeros((4, n))]))
            J = ca.sum2(roll.map(n, "thread", n)(z0, ca.repmat(gains, 1, n))[5, :])   # starts on parallel threads
            solver = ca.nlpsol("solver", "ipopt", {"x": q, "f": J},
                               {"ipopt.hessian_approximation": "limited-memory", "ipopt.max_iter": T["max_iter"],
                                "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False})
            solution = solver(x0=q_opt, lbx=0, ubx=np.inf)
            q_opt = np.asarray(solution["x"]).ravel()
            stats = solver.stats()
            print(f"starts {np.round(np.rad2deg(T['starts'][:n])).astype(int).tolist()} deg: "
                  f"{stats['return_status']}, {stats['iter_count']} iterations, J = {float(solution['f']):.4g}")
    tuned = dict(zip(control.GAINS, first_guess * q_opt))

    print(f"\nTuned gains, each run from rest for {T['check_t_end']:g} s with the arm at 0, torque capped at "
          f"{c.MAX_TORQUE:g} N m:")
    print("  alpha0   max |alpha - 180|   final |alpha - 180|   final theta   peak |torque|")
    for alpha0 in T["check_starts"]:
        t, x, u, *_ = run.simulate([0, alpha0, 0, 0], T["check_t_end"], control.pid(**tuned, dt=c.DT),
                                   T["friction"], c.DT, c.RK4_SUBSTEPS)
        finite = np.all(np.isfinite(x), axis=1)
        if not finite.all():
            print(f"  {np.rad2deg(alpha0):5.0f}°   the state blew up at t = {t[np.argmin(finite)]:.3f} s")
            continue
        error = np.rad2deg(np.abs((np.pi - x[:, 1] + np.pi) % (2 * np.pi) - np.pi))
        print(f"  {np.rad2deg(alpha0):5.0f}°   {error.max():14.2f}°   {error[-1]:18.3f}°   "
              f"{np.rad2deg(x[-1, 0]):10.2f}°   {np.abs(u).max():10.3f} N m")
    print("\nPaste into config.py:\nPID_GAINS = dict(" + ", ".join(f"{k}={v:.6g}" for k, v in tuned.items()) + ")")


if __name__ == "__main__":
    main()
