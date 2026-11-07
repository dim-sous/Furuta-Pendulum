"""Furuta pendulum plant: the Quanser QUBE-Servo 2 with its rotary pendulum module.

The equations of motion come from Lagrange's equations, derived symbolically by CasADi from the kinematics of
two rigid bodies with full inertia tensors. This is the general model of Cazzolato & Prime (2011), eq. (19),
plus the motor rotor and hub inertia and a smooth Stribeck friction law on both joints.

State x = [theta, alpha, theta_dot, alpha_dot]:
  theta  arm angle about the upward vertical, counter-clockwise seen from above;
  alpha  pendulum angle about the arm's outward axis (right-hand rule), 0 = hanging down.
Input u: motor torque on the arm [N m].
"""
import casadi as ca
import numpy as np

import config as c

q = ca.SX.sym("q", 2)                   # [theta, alpha]
qd = ca.SX.sym("qd", 2)                 # [theta_dot, alpha_dot]
x = ca.vertcat(q, qd)
u = ca.SX.sym("u")


def _rot_z(a):
    return ca.blockcat([[ca.cos(a), -ca.sin(a), 0], [ca.sin(a), ca.cos(a), 0], [0, 0, 1]])


def _rot_x(a):
    return ca.blockcat([[1, 0, 0], [0, ca.cos(a), -ca.sin(a)], [0, ca.sin(a), ca.cos(a)]])


def _ddt(expr):
    """Time derivative of an expression in q along the motion: (d expr/dq) qd."""
    return ca.jtimes(expr, q, qd)


def _omega(R):
    """Angular velocity in body axes, from skew(omega) = R^T dR/dt."""
    S = R.T @ ca.reshape(_ddt(ca.reshape(R, 9, 1)), 3, 3)
    return ca.vertcat(S[2, 1], S[0, 2], S[1, 0])


# Kinematics. Arm axes: x along the arm, z up. Pendulum axes: the arm axes turned by alpha about the arm's x;
# the pendulum lies along their -z.
R_arm = _rot_z(q[0])
R_pend = R_arm @ _rot_x(q[1])
pivot = R_arm @ ca.vertcat(c.L_ARM, 0, 0)
com_arm = R_arm @ ca.vertcat(c.D_ARM, 0, 0)
com_pend = pivot + R_pend @ ca.vertcat(0, 0, -c.D_PEND)
tip = pivot + R_pend @ ca.vertcat(0, 0, -c.L_PEND)

# Kinetic energy: translation of each centre of mass plus rotation about it; potential: gravity (zero at the
# height of the arm).
v_arm, v_pend = _ddt(com_arm), _ddt(com_pend)
w_arm, w_pend = _omega(R_arm), _omega(R_pend)
T = (0.5 * c.J_ROTOR * qd[0]**2
     + 0.5 * c.M_ARM * ca.dot(v_arm, v_arm) + 0.5 * ca.dot(w_arm, ca.DM(c.J_ARM) * w_arm)
     + 0.5 * c.M_PEND * ca.dot(v_pend, v_pend) + 0.5 * ca.dot(w_pend, ca.DM(c.J_PEND) * w_pend))
V = c.G * (c.M_ARM * com_arm[2] + c.M_PEND * com_pend[2])
L = T - V

energy = ca.Function("energy", [x], [T + V])
points = ca.Function("points", [q], [pivot, tip])


def friction(w, viscous, coulomb, breakaway, w_breakaway):
    """Friction torque [N m] opposing the joint speed w [rad/s]: viscous + Coulomb + Stribeck breakaway peak,
    as in the MathWorks Simscape "Rotational Friction" block. Smooth, so RK4 converges at 4th order once the
    step resolves the Coulomb band |w| < w_breakaway/10. Dissipative when breakaway >= coulomb >= 0 and
    viscous >= 0."""
    w_st = np.sqrt(2) * w_breakaway     # Stribeck velocity threshold
    w_coul = w_breakaway / 10           # Coulomb velocity threshold
    return (np.sqrt(2 * np.e) * (breakaway - coulomb) * ca.exp(-(w / w_st)**2) * w / w_st
            + coulomb * ca.tanh(w / w_coul) + viscous * w)


def dynamics(with_friction):
    """CasADi functions of the plant: xdot = f(x, u), and P(x, u) = [motor power, friction power loss].

    Lagrange's equations d/dt(dL/dqd) - dL/dq = Q, solved for the accelerations:
    M(q) qdd = Q + dL/dq - (d/dq dL/dqd) qd, with the mass matrix M = d/dqd dL/dqd.
    """
    tau_f = (ca.vertcat(friction(qd[0], **c.FRICTION_ARM), friction(qd[1], **c.FRICTION_PEND))
             if with_friction else ca.DM.zeros(2))
    Q = ca.vertcat(u, 0) - tau_f        # generalised forces
    dL_dqd = ca.gradient(L, qd)
    M = ca.jacobian(dL_dqd, qd)
    qdd = ca.solve(M, Q + ca.gradient(L, q) - ca.jacobian(dL_dqd, q) @ qd)
    f = ca.Function("f", [x, u], [ca.vertcat(qd, qdd)])
    P = ca.Function("P", [x, u], [ca.vertcat(u * qd[0], ca.dot(tau_f, qd))])
    return f, P


def rk4_step(f, P, dt, substeps):
    """Builds step(x, u) -> (x after dt, [motor work, friction loss] over dt): `substeps` classical
    Runge-Kutta 4 steps of length dt / substeps, with u held constant over the whole of dt.

    Integrating the two powers alongside the state makes E(t) - E(0) = motor work - friction loss hold to
    RK4 accuracy.
    """
    def ode(z):
        return ca.vertcat(f(z[:4], u), P(z[:4], u))

    h = dt / substeps
    z = ca.vertcat(x, 0, 0)
    for _ in range(substeps):
        k1 = ode(z)
        k2 = ode(z + h / 2 * k1)
        k3 = ode(z + h / 2 * k2)
        k4 = ode(z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("step", [x, u], [z[:4], z[4:]])
