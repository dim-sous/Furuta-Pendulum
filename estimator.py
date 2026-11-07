"""State estimation from the encoder angles: a continuous-discrete extended Kalman filter, what the controller
sees in experiments with state="ekf". Sources [n] are the References in METHODS.md."""
import casadi as ca
import numpy as np

import model


def ekf(friction_arm, friction_pend, Q_c, R, P0_std, dt, substeps):
    """Builds estimator(y, u_prev) -> (x_hat, P, nis), called once per sample period dt with the encoder angles
    y = [theta, alpha] and the motor torque held over the period that just ended.

    The filter's model is xdot = f(x, u, 0) + L(x) w: the plant's equations of motion with the friction laws
    friction_arm and friction_pend, plus white torque noise w on the arm's and the pendulum's joints, of
    intensity diag(Q_c) [(N m)^2 s]. L = [df/du, df/dd], so w enters exactly as the motor torque and a tap do.

    Predict [18, Alg. 10.24]: from one sample to the next, the estimate and its covariance follow
        dx/dt = f(x, u, 0),    dP/dt = A P + P A^T + L diag(Q_c) L^T,    A = df/dx at the estimate,
    integrated together over dt in `substeps` RK4 steps with u held (zero-order hold, as the plant).
    Update: y = H x + v, H = [I 0], v of covariance diag(R); innovation nu = y - H x, S = H P H^T + diag(R),
    gain K = P H^T S^-1, and the covariance in Joseph form P = (I - K H) P (I - K H)^T + K diag(R) K^T [20],
    symmetric and positive definite by construction. nis = nu^T S^-1 nu, the normalised innovation squared [21].

    The first call returns the start, x_hat = [y, 0, 0] and P = diag(P0_std)^2, without an update (that would
    count the first reading twice); its nis is NaN.
    """
    f, _ = model.dynamics(True, friction_arm, friction_pend)
    x, P, u, d = ca.SX.sym("x", 4), ca.SX.sym("P", 4, 4), ca.SX.sym("u"), ca.SX.sym("d")
    xdot = f(x, u, d)
    linearised = ca.Function("linearised", [x, u, d], [xdot, ca.jacobian(xdot, x), ca.jacobian(xdot, ca.vertcat(u, d))])
    W = ca.diag(ca.DM(Q_c))

    def ode(xs, Ps):
        xdot, A, L = linearised(xs, u, 0)
        return xdot, A @ Ps + Ps @ A.T + L @ W @ L.T

    h = dt / substeps
    xs, Ps = x, P
    for _ in range(substeps):
        k1 = ode(xs, Ps)
        k2 = ode(xs + h / 2 * k1[0], Ps + h / 2 * k1[1])
        k3 = ode(xs + h / 2 * k2[0], Ps + h / 2 * k2[1])
        k4 = ode(xs + h * k3[0], Ps + h * k3[1])
        xs = xs + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Ps = Ps + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    predict = ca.Function("predict", [x, P, u], [xs, Ps])

    y = ca.SX.sym("y", 2)
    H = ca.horzcat(ca.DM.eye(2), ca.DM.zeros(2, 2))
    Rm = ca.diag(ca.DM(R))
    nu = y - H @ x
    S = H @ P @ H.T + Rm
    K = ca.solve(S.T, H @ P.T).T                     # K S = P H^T
    I_KH = ca.DM.eye(4) - K @ H
    update = ca.Function("update", [x, P, y],
                         [x + K @ nu, I_KH @ P @ I_KH.T + K @ Rm @ K.T, ca.dot(nu, ca.solve(S, nu))])

    x_hat, P_hat = None, None

    def estimator(y, u_prev):
        nonlocal x_hat, P_hat
        if x_hat is None:
            x_hat, P_hat = np.concatenate([y, [0.0, 0.0]]), np.diag(np.square(P0_std))
            return x_hat, P_hat, np.nan
        x_pred, P_pred = predict(x_hat, P_hat, u_prev)
        x_new, P_new, nis = update(x_pred, P_pred, y)
        x_hat, P_hat = np.asarray(x_new).ravel(), np.asarray(P_new)
        return x_hat, P_hat, float(nis)

    return estimator
