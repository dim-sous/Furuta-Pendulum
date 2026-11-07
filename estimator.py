"""State estimation from the encoder angles: a continuous-discrete extended Kalman filter whose update conditions
on the encoder's count cell, what the controller sees in experiments with state="ekf". Sources [n] are the
References in METHODS.md."""
import casadi as ca
import numpy as np
from scipy.special import ndtr, owens_t        # standard normal CDF; Owen's T function

import model


def ekf(friction_arm, friction_pend, Q_c, P0_std, resolution, dt, substeps):
    """Builds estimator(y, u_prev) -> (x_hat, P, cell_prob), called once per sample period dt with the encoder
    readings y = [theta, alpha] and the motor torque held over the period that just ended.

    The filter's model is xdot = f(x, u, 0) + L(x) w: the plant's equations of motion with the friction laws
    friction_arm and friction_pend, plus white torque noise w on the arm's and the pendulum's joints, of
    intensity diag(Q_c) [(N m)^2 s]. L = [df/du, df/dd], so w enters exactly as the motor torque and a tap do.

    Predict [18, Alg. 10.24]: from one sample to the next, the estimate and its covariance follow
        dx/dt = f(x, u, 0),    dP/dt = A P + P A^T + L diag(Q_c) L^T,    A = df/dx at the estimate,
    integrated together over dt in `substeps` RK4 steps with u held (zero-order hold, as the plant).
    Update: a reading says only that the true angles z = H x, H = [I 0], lie in its count cell, y +- resolution/2
    in each angle. With the predicted belief z ~ N(H x, S), S = H P H^T, let m and V be the mean and covariance of
    that normal truncated to the cell (truncated_normal_2d). Then, exactly for a Gaussian belief [25, 24],
        K = P H^T S^-1,    x_hat = x + K (m - H x),    P = P - K (S - V) K^T,
    and the belief is taken as Gaussian again with that mean and covariance. cell_prob is the belief's
    probability of the reported cell: how surprising the reading was.

    The first call returns the start, x_hat = [y, 0, 0] and P = diag(P0_std)^2, without an update (that would
    count the first reading twice); its cell_prob is NaN.
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

    x_hat, P_hat = None, None

    def estimator(y, u_prev):
        nonlocal x_hat, P_hat
        if x_hat is None:
            x_hat, P_hat = np.concatenate([y, [0.0, 0.0]]), np.diag(np.square(P0_std))
            return x_hat, P_hat, np.nan
        x_pred, P_pred = (np.asarray(a) for a in predict(x_hat, P_hat, u_prev))
        x_pred = x_pred.ravel()
        S = P_pred[:2, :2]
        m, V, cell_prob = truncated_normal_2d(x_pred[:2], S, y - resolution / 2, y + resolution / 2)
        K = np.linalg.solve(S.T, P_pred[:, :2].T).T                   # K S = P H^T
        x_hat = x_pred + K @ (m - x_pred[:2])
        P_hat = P_pred - K @ (S - V) @ K.T
        return x_hat, P_hat, cell_prob

    return estimator


def truncated_normal_2d(mean, cov, lower, upper):
    """Mean m, covariance V and probability of the bivariate normal N(mean, cov) truncated to the rectangle
    [lower[0], upper[0]] x [lower[1], upper[1]], in closed form [22, eqs. (10), (11), (16)-(18), (21)]: everything
    but the rectangle probability needs only the univariate normal density and CDF."""
    sd = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (sd[0] * sd[1])
    s = np.sqrt(1 - rho**2)
    lo, up = (lower - mean) / sd, (upper - mean) / sd                 # standardized limits
    prob = _rectangle_probability(lo, up, rho)
    pdf = lambda t: np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
    pdf2 = lambda a, b: np.exp(-(a**2 - 2 * rho * a * b + b**2) / (2 * s**2)) / (2 * np.pi * s)
    # c_k(t): density of the k-th standardized coordinate at t, times the other's probability of its interval
    # given that value [22, eq. (10)]
    c = [lambda t, j=j: pdf(t) * _interval_probability((lo[j] - rho * t) / s, (up[j] - rho * t) / s)
         for j in (1, 0)]
    A = np.array([c[k](lo[k]) - c[k](up[k]) for k in (0, 1)])
    B = np.array([lo[k] * c[k](lo[k]) - up[k] * c[k](up[k]) for k in (0, 1)])
    D = pdf2(lo[0], lo[1]) - pdf2(lo[0], up[1]) - pdf2(up[0], lo[1]) + pdf2(up[0], up[1])
    E = np.array([A[0] + rho * A[1], rho * A[0] + A[1]]) / prob       # E[standardized coordinate]
    E2 = np.array([[1 + (B[0] + rho**2 * B[1] + rho * s**2 * D) / prob, rho + (rho * B[0] + rho * B[1] + s**2 * D) / prob],
                   [rho + (rho * B[0] + rho * B[1] + s**2 * D) / prob, 1 + (rho**2 * B[0] + B[1] + rho * s**2 * D) / prob]])
    return mean + sd * E, np.outer(sd, sd) * (E2 - np.outer(E, E)), prob


def _interval_probability(a, b):
    """P(a < Z < b) for a standard normal Z, written on the side of the tail where the CDF is small, so the
    difference does not cancel: Phi(b) - Phi(a) = Phi(-a) - Phi(-b)."""
    return ndtr(-a) - ndtr(-b) if a > 0 else ndtr(b) - ndtr(a)


def _rectangle_probability(lo, up, rho):
    """P(lo < Z < up) for standard normals Z = (Z1, Z2) with correlation rho, by inclusion-exclusion of the
    bivariate CDF. Each coordinate whose interval centre lies above 0 is mirrored first (Z_k -> -Z_k, which flips
    the sign of rho once per mirrored coordinate): an exact symmetry that puts the rectangle where the four
    corner CDFs are small, so their sum does not cancel."""
    lo, up = np.array(lo, float), np.array(up, float)
    for k in (0, 1):
        if lo[k] + up[k] > 0:
            lo[k], up[k] = -up[k], -lo[k]
            rho = -rho
    F = lambda a, b: _bivariate_cdf(a, b, rho)
    return F(up[0], up[1]) - F(lo[0], up[1]) - F(up[0], lo[1]) + F(lo[0], lo[1])


def _bivariate_cdf(h, k, rho):
    """P(Z1 <= h, Z2 <= k) for standard normals with correlation rho, from Owen's T function [23]:
    Phi2 = Phi(h)/2 + Phi(k)/2 - T(h, a_h) - T(k, a_k) - beta, a_h = (k - rho h)/(h s), a_k = (h - rho k)/(k s),
    s = sqrt(1 - rho^2), beta = 1/2 when h and k have opposite signs (or one is 0 and their sum is negative), else
    0. a_h -> sign(k) inf as h -> 0+. With beta = 1/2 the constant part is written Phi(lo)/2 - Phi(-hi)/2,
    lo = min(h, k), hi = max(h, k), an exact rewrite that avoids cancelling Phi(hi)/2 against 1/2."""
    if h == 0 and k == 0:
        return 0.25 + np.arcsin(rho) / (2 * np.pi)
    s = np.sqrt(1 - rho**2)
    a_h = (k - rho * h) / (h * s) if h != 0 else np.copysign(np.inf, k)
    a_k = (h - rho * k) / (k * s) if k != 0 else np.copysign(np.inf, h)
    if h * k < 0 or (h * k == 0 and h + k < 0):
        base = 0.5 * ndtr(min(h, k)) - 0.5 * ndtr(-max(h, k))
    else:
        base = 0.5 * ndtr(h) + 0.5 * ndtr(k)
    return base - owens_t(h, a_h) - owens_t(k, a_k)
