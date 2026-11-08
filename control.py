"""Controllers, chosen per experiment in config.EXPERIMENTS. Each returns controller(t, x) -> motor torque [N m],
called once per sample period."""
import fcntl
import hashlib
import subprocess
import tempfile
from pathlib import Path

import casadi as ca
import numpy as np

import config as c
import model

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


def nmpc(period, substeps, horizon, max_pendulum_error, max_arm_error, max_torque, max_torque_change, friction_arm,
         friction_pend, ipopt, ipopt_warm, dt):
    """Nonlinear model predictive control: one law swings the pendulum up and balances it, with no switching.

    It samples every `period` s, a multiple of the loop's sample period dt, and holds its torque in between.
    Each of its samples it solves, from the state x it sees and the torque u_prev it applied the sample before,
        minimize over u_0 .. u_{N-1}:   period sum_{k=0}^{N-1} l(x_k, u_k, u_k - u_{k-1}),    u_{-1} = u_prev,
        l = 2 (1 + cos alpha) / max_pendulum_error^2 + 2 (1 - cos theta) / max_arm_error^2
            + (u / max_torque)^2 + ((u_k - u_{k-1}) / max_torque_change)^2,
        subject to x_0 = x,  x_{k+1} = F(x_k, u_k),  |u_k| <= MAX_TORQUE,
    and applies u_0 until its next sample. N = horizon / period. F is `substeps` RK4 steps over one period with u
    held, of the model with the friction laws friction_arm and friction_pend. Near upright
    2 (1 + cos alpha) ~ (alpha - pi)^2 and 2 (1 - cos theta) ~ theta^2, so each term is an error divided by its
    largest acceptable value, squared (Bryson's rule, as tune.py), while staying periodic: a whole turn of either
    angle counts as none, and no wrap is needed. No terminal cost and no terminal constraint.

    Multiple shooting: the states x_1 .. x_N are unknowns too, tied by the continuity constraints. F is built once
    and mapped over the N intervals, so the problem's graph grows linearly with N. IPOPT solves it to convergence
    with the exact Hessian, its functions compiled to C (_compiled). The first solve (options `ipopt`) starts from
    the states on a straight line from x to the nearest upright at rest, torque 0: it picks the swing direction, as
    at hanging rest the cost is flat by symmetry. Each later solve starts from the previous solution shifted by one
    sample, its multipliers included, and IPOPT warm-starts from them (options `ipopt` and `ipopt_warm`: a small
    initial barrier parameter, so it does not first move away from the previous solution). A solve that does not
    converge is recorded in controller.failures, with its time and IPOPT's status, and from then on the controller
    returns NaN: the run blows up there, and run.py reports both and plots the run up to that moment.
    """
    n, hold = round(horizon / period), round(period / dt)
    if not (np.isclose(n * period, horizon, rtol=1e-12, atol=0) and np.isclose(hold * dt, period, rtol=1e-12, atol=0)):
        raise ValueError(f"NMPC: horizon = {horizon} s must be a multiple of period = {period} s, and period a "
                         f"multiple of dt = {dt} s")
    # F calls one RK4 step `substeps` times rather than unrolling them into one expression: the same arithmetic, but
    # the compiled derivatives stay small (unrolled, the exact Hessian's C code exhausted the compiler's memory).
    f, P = model.dynamics(True, friction_arm, friction_pend)
    step = model.rk4_step(f, P, period / substeps, 1)
    xs, us = ca.SX.sym("x", 4), ca.SX.sym("u")
    rk4 = ca.Function("rk4", [xs, us], [step(xs, us, 0)[0]])
    xm, um = ca.MX.sym("x", 4), ca.MX.sym("u")
    x_end = xm
    for _ in range(substeps):
        x_end = rk4(x_end, um)
    F = ca.Function("F", [xm, um], [x_end]).map(n)

    # Unknowns: the states x_0 .. x_N, then the torques; parameters: x and u_prev.
    X, U, p = ca.MX.sym("X", 4, n + 1), ca.MX.sym("U", 1, n), ca.MX.sym("p", 5)
    du = U - ca.horzcat(p[4], U[:, :-1])
    cost = period * ca.sum2(2 * (1 + ca.cos(X[1, :-1])) / max_pendulum_error**2
                            + 2 * (1 - ca.cos(X[0, :-1])) / max_arm_error**2
                            + (U / max_torque)**2 + (du / max_torque_change)**2)
    nlp = {"x": ca.vertcat(ca.vec(X), ca.vec(U)), "p": p, "f": cost,
           "g": ca.vertcat(X[:, 0] - p[:4], ca.vec(X[:, 1:] - F(X[:, :-1], U)))}
    library = _compiled(ca.nlpsol("nlp", "ipopt", nlp))
    cold = ca.nlpsol("cold", "ipopt", library, {"print_time": False, "ipopt": ipopt})
    warm = ca.nlpsol("warm", "ipopt", library, {"print_time": False, "ipopt": ipopt | ipopt_warm})
    m = 4 * (n + 1)                     # state unknowns; the n torques follow
    lbx = np.r_[np.full(m, -np.inf), np.full(n, -c.MAX_TORQUE)]

    start = None                        # what the next solve starts from: unknowns and multipliers
    u_last = 0.0                        # the torque applied the sample before; the motor is off before the start
    calls = 0                           # loop samples so far
    failures = []                       # messages of solves that did not converge

    def controller(t, x):
        nonlocal start, u_last, calls
        calls += 1
        if (calls - 1) % hold or failures:  # between the NMPC's samples, or after a failed solve: hold
            return u_last
        if start is None:
            wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
            upright = [x[0] + wrap(-x[0]), x[1] + wrap(np.pi - x[1]), 0.0, 0.0]
            solver, guess = cold, dict(x0=np.r_[np.linspace(x, upright, n + 1).ravel(), np.zeros(n)])
        else:
            solver, guess = warm, start
        solution = solver(**guess, p=np.r_[x, u_last], lbx=lbx, ubx=-lbx, lbg=0, ubg=0)
        if not solver.stats()["success"]:
            failures.append(f"the NMPC's solve at t = {t:.3f} s did not converge (IPOPT: "
                            f"{solver.stats()['return_status']}); the controller gives NaN from there")
            u_last = np.nan
            return u_last
        w, lam_g, lam_x = (np.asarray(solution[key]).ravel() for key in ("x", "lam_g", "lam_x"))
        states, torques = w[:m].reshape(n + 1, 4), w[m:]
        continuity = lam_g[4:].reshape(n, 4)    # one row per interval; the new x_0 is the old x_1
        start = dict(x0=np.r_[states[1:].ravel(), states[-1], torques[1:], torques[-1]],
                     lam_g0=np.r_[continuity.ravel(), continuity[-1]],
                     lam_x0=np.r_[np.zeros(m), lam_x[m + 1:], lam_x[-1]])
        u_last = float(torques[0])
        return u_last

    controller.failures = failures
    return controller


def _compiled(solver):
    """Path of a shared library with the functions the nlpsol `solver` evaluates, compiled by gcc -O2 (minutes for
    the NMPC). It is cached in the system's temporary folder under a hash of its C source, so a rerun with unchanged
    settings loads it instead of compiling again. Runs in parallel processes each generate their own source file,
    and compile one at a time under a lock (one gcc needs up to ~2 GB for the NMPC: several at once once exhausted
    the machine's memory); a process that waited finds the library built and loads it."""
    folder = Path(tempfile.gettempdir()) / "furuta_nmpc"
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as own:
        generator = ca.CodeGenerator("nlp.c")
        generator.add(solver.oracle())      # the problem itself, (x, p) -> (f, g), which nlpsol looks up by name
        for name in solver.get_function():
            generator.add(solver.get_function(name))
        source = Path(generator.generate(own + "/"))
        library = folder / f"{hashlib.sha256(source.read_bytes()).hexdigest()[:16]}.so"
        with open(folder / "compile.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library.exists():
                built = Path(own) / "nlp.so"
                subprocess.run(["gcc", "-O2", "-fPIC", "-shared", str(source), "-o", str(built)], check=True)
                built.replace(library)      # only a complete library gets the name
    return str(library)
