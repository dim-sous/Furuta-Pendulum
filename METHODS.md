# Methods

How the simulation and its controllers work: the model, the numerics, and each controller's law, with sources.

## 1. Model

### The system

A Quanser QUBE-Servo 2 with its rotary pendulum module. A DC motor turns a horizontal arm; a pendulum hangs
from a free joint at the arm's tip and swings in the plane perpendicular to the arm. One input (the motor
torque $u$) drives two degrees of freedom, so the system is underactuated.

State $x = [\theta,\ \alpha,\ \dot\theta,\ \dot\alpha]$:

- $\theta$: arm angle about the upward vertical, counter-clockwise seen from above.
- $\alpha$: pendulum angle about the arm's outward axis (right-hand rule); $\alpha = 0$ hanging down,
  $\alpha = \pm\pi$ upright.

The angles follow Cazzolato & Prime [1]. One difference: [1] puts the pendulum's x axis along the rod and z
on the pivot axis, while here x is the pivot axis and z the rod, so their $J_{2xx}, J_{2yy}, J_{2zz}$ are
$I_{p,zz}, I_{p,yy}, I_{p,xx}$ here.

### Kinematics

Two rotations place each body: the arm frame is $R_a = R_z(\theta)$ (x along the arm, z up) and the pendulum
frame is $R_p = R_z(\theta)\thinspace R_x(\alpha)$, with the pendulum along its $-z$ axis. The centres of mass are

```math
r_a = R_a \begin{bmatrix} d_a \\ 0 \\ 0 \end{bmatrix},\qquad
r_p = R_a \begin{bmatrix} L_a \\ 0 \\ 0 \end{bmatrix} + R_p \begin{bmatrix} 0 \\ 0 \\ -d_p \end{bmatrix},
```

their velocities are $v = \frac{\partial r}{\partial q}\dot q$ with $q = [\theta,\ \alpha]$, and the angular
velocities in body axes follow from $[\omega]_\times = R^\top \dot R$ [3]:

```math
\omega_a = \begin{bmatrix} 0 \\ 0 \\ \dot\theta \end{bmatrix},\qquad
\omega_p = \begin{bmatrix} \dot\alpha \\ \dot\theta \sin\alpha \\ \dot\theta \cos\alpha \end{bmatrix}.
```

`model.py` writes exactly these definitions and lets CasADi [2] differentiate them; nothing below was
expanded by hand in the code.

### Energy and equations of motion

Each body contributes translational energy of its centre of mass plus rotational energy about it, with its
full inertia tensor $I$ (about the centre of mass, in body axes) [3]. The motor rotor and hub add $J_m$:

```math
T = \tfrac12 J_m \dot\theta^2
  + \tfrac12 m_a \lVert v_a\rVert^2 + \tfrac12\, \omega_a^\top I_a\, \omega_a
  + \tfrac12 m_p \lVert v_p\rVert^2 + \tfrac12\, \omega_p^\top I_p\, \omega_p,
\qquad
V = -m_p\, g\, d_p \cos\alpha .
```

Lagrange's equations with generalised forces [3],

```math
\frac{d}{dt}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = Q,\qquad L = T - V,\qquad
Q = \begin{bmatrix} u - \tau_a(\dot\theta) \\ d - \tau_p(\dot\alpha) \end{bmatrix},
```

where $\tau_a$ and $\tau_p$ are the friction torques of the arm and pendulum joints and $d$ is a disturbance
torque on the pendulum's joint, zero except in the tap experiments (section 6). Expanding the time
derivative gives $M(q)\thinspace \ddot q = Q + \frac{\partial L}{\partial q} - \frac{\partial^2 L}{\partial \dot q\thinspace \partial q}\thinspace \dot q$
with the mass matrix $M = \frac{\partial^2 L}{\partial \dot q^2}$; `model.dynamics` solves this for
$\ddot q$. Written out, the result is

```math
\begin{aligned}
&\big(J_0 + (m_p d_p^2 + I_{p,yy})\sin^2\alpha + I_{p,zz}\cos^2\alpha\big)\,\ddot\theta
 + m_p L_a d_p \cos\alpha\;\ddot\alpha
\\ &\qquad + (m_p d_p^2 + I_{p,yy} - I_{p,zz})\sin 2\alpha\;\dot\theta\dot\alpha
 - m_p L_a d_p \sin\alpha\;\dot\alpha^2 = u - \tau_a(\dot\theta), \\[4pt]
&m_p L_a d_p \cos\alpha\;\ddot\theta + (I_{p,xx} + m_p d_p^2)\,\ddot\alpha
 - \tfrac12 (m_p d_p^2 + I_{p,yy} - I_{p,zz})\sin 2\alpha\;\dot\theta^2
 + m_p g d_p \sin\alpha = d - \tau_p(\dot\alpha),
\end{aligned}
```

with $J_0 = J_m + I_{a,zz} + m_a d_a^2 + m_p L_a^2$. This is Cazzolato & Prime's full model, their eq. (19) [1],
plus the rotor and hub inertia and friction. They point out that most published Furuta models drop the
pendulum's own inertia terms ($I_{p,yy}\sin^2\alpha$ and the velocity terms that come with it); the classic
QUBE workbook model [9] drops them too. They are kept here.

### Friction

Both joints use the smooth Stribeck law of the MathWorks Simscape "Rotational Friction" block [4], the block
Quanser used to model the QUBE-Servo 2 motor [5]:

```math
\tau(\omega) = \sqrt{2e}\,(\tau_{brk} - \tau_C)\, e^{-(\omega/\omega_{St})^2}\, \frac{\omega}{\omega_{St}}
 + \tau_C \tanh\!\Big(\frac{\omega}{\omega_C}\Big) + b\,\omega,
\qquad \omega_{St} = \sqrt2\,\omega_{brk},\quad \omega_C = \omega_{brk}/10 .
```

In plain terms: viscous drag $b\omega$, a Coulomb level $\tau_C$ reached smoothly within about $\omega_C$,
and a Stribeck term that lifts the Coulomb level to $\tau_{brk}$ at speed $\omega_{brk}$. With the arm's values,
$b\thinspace \omega_{brk}$ = 1e-4 N m is as large as that lift, so the total arm friction still rises monotonically with
speed; the pendulum's friction is purely viscous. The law is smooth everywhere, and it dissipates
($\tau(\omega)\thinspace \omega \ge 0$) whenever $\tau_{brk} \ge \tau_C \ge 0$ and $b \ge 0$, as here.

### Parameters

| Symbol | `config.py` | Value | Source |
|---|---|---|---|
| $m_a$, $L_a$ | `M_ARM`, `L_ARM` | 0.095 kg, 0.085 m | QUBE-Servo 2 manual, Table 2.2 [6] |
| $d_a$, $I_{a,zz}$ | `D_ARM`, `J_ARM` | $L_a/2$, $m_a L_a^2/12$ | uniform rod pivoted at its end |
| $J_m$ | `J_ROTOR` | 4.0e-6 + 0.6e-6 kg m² | rotor + hub, [6] |
| $m_p$, $L_p$ | `M_PEND`, `L_PEND` | 0.024 kg, 0.129 m | [6] |
| $d_p$, $I_{p,xx} = I_{p,yy}$, $I_{p,zz}$ | `D_PEND`, `J_PEND` | $L_p/2$, $m_p L_p^2/12$, 0 | uniform thin rod pivoted at its end |
| $b$ arm, pendulum | `FRICTION_*["viscous"]` | 1e-3, 5e-5 N m s/rad | Quanser parameter file [7] |
| $\tau_C$, $\tau_{brk}$ arm | `FRICTION_ARM` | 1.02e-4, 2.03e-4 N m | Quanser motor friction test [5] |
| $\tau_C$, $\tau_{brk}$ pendulum | `FRICTION_PEND` | 0, 0 | no published value |
| $\omega_{brk}$ | `w_breakaway` | 0.1 rad/s | not published; Simscape default [4] |
| $g$ | `G` | 9.81 m/s² | as in [7] |

Two choices settle conflicts between Quanser's own sources. The arm is a rod pivoted at its end
($m_a L_a^2/3 = 2.29\times10^{-4}$ kg m² about the motor axis), as the manual's geometry [6] and the current
parameter file [7] state; the 2014 workbook's printed matrices [9] are reproduced only with $m_a L_a^2/12$, the
formula for a rod pivoted at its centre. Friction values in circulation differ by up to 300× (arm) and 500×
(pendulum); the set above is Quanser's current file plus its own motor measurement.

### Assumptions

- Rigid bodies; uniform rods; the pendulum has no inertia about its own long axis.
- Torque input: the motor's electrical dynamics and back-EMF are not modelled, so zero torque means a freely
  spinning motor. On the real rig, 0 V shorts the motor and brakes the arm (about 2.1e-4 N m s/rad).
- The plant itself has no torque limit. Every controller's torque is capped at `MAX_TORQUE` = 0.05 N m, about
  what the motor gives at 10 V ($k_t \cdot 10\thinspace \text{V}/R_m$), so all controllers compete under the same rule;
  the open-loop experiments stay at or below 0.02 N m.
- Quanser's 2016 product sheet [10] gives the pendulum as 9.5 cm pivot-to-tip, against 0.129 m in the
  manual [6]; Quanser's parameter file [7] uses 0.129 m, as here.

## 2. Numerical integration and the energy checks

The torque is held constant over each sample period of 1 ms (`DT`, zero-order hold), as a digital controller
would apply it; the hold is part of the simulated system. Within each period the state is integrated with the
classical Runge–Kutta 4 method [8] in `RK4_SUBSTEPS` = 4 steps of 0.25 ms. `DT` is therefore a physical
setting and the RK4 step a numerical one: refining the RK4 step leaves the held torque unchanged.

The energy check rests on an exact identity. Multiplying Lagrange's equations by $\dot q^\top$, and using that
$T$ is quadratic in $\dot q$ and $V$ depends on $q$ only, gives [3]

```math
\frac{d}{dt}(T + V) = \dot q^\top Q = u\,\dot\theta + d\,\dot\alpha - \tau_a(\dot\theta)\,\dot\theta
 - \tau_p(\dot\alpha)\,\dot\alpha .
```

`model.rk4_step` integrates the motor power $u\dot\theta$, the disturbance power $d\dot\alpha$ and the friction
power alongside the state, so $E(t) - E(0) - (\text{motor work}) - (\text{tap work}) + (\text{friction loss})$ is
zero up to RK4's truncation error. Every run
is flagged if its state stops being finite (an unstable closed loop can blow up; the plots then end there) or
if this residual exceeds `ENERGY_TOL` = 1e-6 J. A flag does not stop anything: it is printed, written in red
on the run's figure and video, and `run.py` ends by listing the flagged runs. With friction off and $u = 0$ this
is the classic check that energy is conserved. With friction and torque on, it also catches a sign error in
how $u$ or the friction torques enter the equations of motion. It cannot catch a sign error inside the
friction law itself, because that changes the equations and the integrated power together. A second check
covers that case: the friction loss must never decrease, since each RK4 increment of it is a positive
combination of $\tau(\omega)\thinspace \omega \ge 0$ terms.

## 3. PID control

Two PIDs, summed: one holds the pendulum upright, the other holds the arm at $\theta = 0$. Every 1 ms (`DT`)
the controller reads the state and sets a torque that is held until the next sample:

```math
\begin{aligned}
e_k &= \mathrm{wrap}(\pi - \alpha_k),\qquad e_{a,k} = \mathrm{wrap}(0 - \theta_k)\qquad \in [-\pi, \pi),
\\ v_k &= K_p\, e_k - K_d\, \dot\alpha_k + K_{p,a}\, e_{a,k} - K_{d,a}\, \dot\theta_k + I_k,
 \qquad u_k = \mathrm{sat}_{u_{max}}(v_k),
\\ I_{k+1} &= I_k + (K_i\, e_k + K_{i,a}\, e_{a,k})\,\Delta t
 \quad\text{unless } u_k \text{ is capped and that increment would push it further into the cap.}
\end{aligned}
```

$I$ is the integral term of both loops in one state, in N m; below the cap it equals
$K_i \sum e\thinspace \Delta t + K_{i,a} \sum e_a\thinspace \Delta t$ exactly.

With the arm gains at 0 this is the pendulum-only PID. The arm error is wrapped too, so a whole turn of the arm
counts as none.

- The error is measured from upright and wrapped, so the controller always takes the shorter way up. It is
  Quanser's balance angle, $\alpha \bmod 2\pi - \pi$ [9, Rotary Pendulum Modeling lab, eq. (1.1)], with the sign
  flipped so that $e$ = setpoint − measurement. The two differ only hanging straight down, where the wraps
  break the tie in opposite directions: here $e = -\pi$, while the flipped Quanser angle is $+\pi$.
- The D terms act on the measured rates $\dot\alpha$, $\dot\theta$ rather than on the errors' derivatives, i.e.
  setpoint weight $\gamma = 0$ in [11, eq. (11.15)]. For the pendulum the two agree except at the wrap's $2\pi$
  jump, where $\dot e$ would be an impulse.
- The integral is the forward-Euler sum of the standard computer implementation [11, eq. (11.17)],
  [12, eq. (13.7)]. Against windup it uses conditional integration [12, §3.5]: while the torque sits at the cap,
  increments that would push it further are dropped. Back-calculation [11, §11.4] is not used: with
  integral gains ≈ 0, as the tuned ones are, nothing pulls its integrator back, and it keeps a stuck torque offset.
- The controller reads the exact simulated state: no encoder quantization, noise or rate filter.
- The torque is capped at $u_{max}$ = `MAX_TORQUE` = 0.050 N m, what the real motor delivers at the recommended
  10 V (0.075 N m at its 15 V maximum) [6, §2.2.4 and Table 2.2].
- The gains are set by hand or by `tune.py` in `config.PID_GAINS` ($K_p$ in N m/rad, $K_i$ in N m/(rad s), $K_d$ in
  N m s/rad; the arm gains `*_arm` in the same units). Quanser's own pendulum controllers sample at 500 Hz [13];
  here `DT` is 1 ms.

## 4. Gain tuning by optimization

**The problem.** From each start $i$ (pendulum at rest at $\alpha_0$, arm at 0) the closed loop is simulated for
a horizon $T$ = 3 s with exactly the plant, RK4 sub-steps, held torque and PID formula of the experiments
(`control.pid_step` serves both). The tuner minimizes

```math
J = \sum_i \sum_k \left[ \left(\frac{e_k}{e_{max}}\right)^2 + \left(\frac{e_{a,k}}{\theta_{max}}\right)^2
 + \left(\frac{u_k}{u_{max}}\right)^2 \right] \Delta t ,
```

an integral of squared errors and torque in which each term is divided by its largest acceptable value
(Bryson's rule [16]): $e_{max}$ = 5°, $\theta_{max}$ = 30°, $u_{max}$ = 0.05 N m, the real motor's limit at
10 V [6]. The tuning loop leaves out the torque cap (`TUNE["cap"]` = ∞), because IPOPT fails with the cap in
the loop: gains that drop the pendulum give a high but nearly flat cost (a dropped pendulum hardly depends on
the gains), so the optimizer can stop on that plateau and report success, and between catching and dropping
the gradients become huge. Without the cap, a dropped pendulum blows the state up instead, and the resulting
NaN keeps the optimizer away. Torque is only penalized, and the tuned gains are then checked with the cap, as
the experiments run them.

**The method.** CasADi builds the simulation symbolically, so $J$ has exact derivatives [2], and IPOPT [14]
minimizes it with a limited-memory BFGS Hessian [15]. The unknowns are the gains divided by the current
`PID_GAINS`, so all six share one scale; each is bounded to keep that gain's sign (zero allowed), since the
pendulum gains must be positive and the arm gains negative. Three things make it fast enough:

- one sample of the closed loop is compiled to C (it needs `gcc`), about 30× faster than CasADi's interpreter;
- the starts are simulated on parallel threads;
- only one side of upright is used: a start at 180° + δ mirrors one at 180° − δ, with the same cost (the plant
  and the controller are odd in $(\theta, \alpha - \pi)$; checked numerically to 13 digits).

The starts (175°, 170°, 165°, 160°) are added one at a time, each stage starting from the previous one's gains:
the hand-tuned first guess could not even start on the full set, as it drops the pendulum from 160°.

## 5. Swing-up by energy control

A PID only holds the pendulum near upright; from hanging down it has to be swung up first. The swing-up follows
Åström & Furuta [17]: pump energy into the pendulum until it has exactly what it needs to reach upright, then
hand over to the PID of section 3 (`control.swing_up`, settings `config.SWING_UP`). The pendulum's energy about
its pivot is

```math
E = \tfrac12 J_{pivot}\,\dot\alpha^2 + m_p g d_p (1 - \cos\alpha),\qquad J_{pivot} = I_{p,xx} + m_p d_p^2 ,
```

0 hanging at rest and $E_{up} = 2 m_p g d_p$ = 30.4 mJ upright. From the pendulum row of the equations of motion,
the arm drives it as $\dot E = -m_p L_a d_p \cos\alpha\thinspace \dot\alpha\thinspace \ddot\theta$ plus terms in the arm's speed. A
torque that accelerates the arm against the sign of $\cos\alpha\thinspace \dot\alpha$ therefore adds energy:

```math
u = -\mathrm{sat}_{u_{max}}\!\big(k\,(E_{up} - E)\,\mathrm{sign}(\dot\alpha\cos\alpha)\big).
```

- **Torque form.** [17] commands the arm's acceleration; here the torque itself follows the same switching
  structure. The direction of energy pumping is exact; its rate is set by $k$.
- **Saturation.** The whole controller, PID included, is capped at $u_{max}$ = `MAX_TORQUE` = 0.05 N m;
  $k$ = 50 N m/J, so the law runs mostly at the cap (bang-bang) and eases off as $E \to E_{up}$.
- **Hand-over, with hysteresis.** The PID takes over when the pendulum comes within 20° of upright, as in
  Quanser's swing-up lab [9, Swing-Up Control lab, eq. (1.6)], and hands back to the swing-up only if it falls
  beyond 45°, so a missed catch is retried. The gap matters: while catching, the PID lets the pendulum overshoot
  briefly to 28°–35° from upright, and with one 20° line that overshoot handed control back mid-catch. The
  PID's integral restarts at 0 at each hand-over.
- **From exact rest** $\dot\alpha = 0$, so the law would give no torque; $\mathrm{sign}(0)$ is taken as
  +1, so the swing-up starts by itself.

## 6. Disturbance: a tap on the pendulum

Once the swing-up has balanced the pendulum, a tap hits it: a torque pulse $d$ on the pendulum's joint lasting
20 ms from $t$ = 5 s (`config.TAP`), one experiment per size, `swing_up_tap_<size>mNm`. The tap is not capped, as
it is not the controller; its work $\int d\thinspace \dot\alpha\thinspace dt$ enters the energy balance of section 2.

## 7. State estimation: an EKF on the encoder angles

The rig measures only the two angles, with optical encoders of 2048 counts per revolution in quadrature [21]. In
experiments with `state="ekf"` the controller sees an extended Kalman filter's estimate made from those readings,
instead of the true state (`estimator.py`, settings `config.EKF`).

**The encoders.** A reading is each angle rounded to whole counts, $y = \Delta\thinspace \mathrm{round}([\theta, \alpha]/\Delta)$
with $\Delta = 2\pi/2048$ = 3.07e-3 rad (0.176°), zero where the model's angles are zero (a calibrated encoder).
There is no other noise: optical encoders have none to speak of.

**The filter's model.** The filter's model is the plant of section 1, with its arm friction identified from
encoder data in section 8, viscous + Coulomb ($b$ = 9.9945e-4 N m s/rad, $\tau_C$ = 1.1141e-4 N m, the sign function
smoothed over 0.01 rad/s), and its pendulum friction viscous, the plant's value. The truth keeps its Stribeck law,
so the filter never has the truth's model. Unmodelled torques are white noise $w$ on the two joints:

```math
\dot x = f(x, u, 0) + L(x)\,w, \qquad L = \begin{bmatrix} \dfrac{\partial f}{\partial u} & \dfrac{\partial f}{\partial d} \end{bmatrix},
\qquad E[w(t)\,w(s)^T] = Q_c\,\delta(t - s),
```

so $w$ enters exactly as the motor torque and a tap do, and $Q_c = \mathrm{diag}(q_{arm}, q_{pend})$ has units
(N m)² s. Its first guess was the Coulomb torque a viscous-only model leaves out, held for one sample,
$q = (10^{-4}\ \text{N m})^2\thinspace \Delta t$ = 1e-11 per joint: a constant torque $\tau$ over $\Delta t$ gives an impulse of
variance $(\tau\Delta t)^2$, and white noise of intensity $q$ gives $q\thinspace \Delta t$. Tuned by hand on a sweep, $q$ = 1e-8 per
joint: the lowest value at which both EKF experiments keep the pendulum (5e-9 loses it). Below it, the filter
grows too sure of its own predictions, a count change becomes a reading it considers impossible, and it
diverges. With a viscous-only arm model the filter needed 1e-7; the identified friction lets it trust its model
ten times more, which halves its velocity errors and the balancing torque's chatter.

**Predict** (continuous-discrete EKF [18, Alg. 10.24]). Between samples the estimate and its covariance follow

```math
\dot{\hat x} = f(\hat x, u, 0), \qquad \dot P = A P + P A^T + L Q_c L^T, \qquad A = \frac{\partial f}{\partial x}(\hat x, u, 0),
```

integrated together by the same RK4 sub-steps as the plant (section 2), with $u$ held. CasADi gives $A$ and $L$
exactly [2].

**Update: conditioning on the count cell.** A reading is not the angle plus noise; it says only that the true
angles $z = Hx$, $H = [I\ \ 0]$, lie in the cell $C = [y_\theta \pm \Delta/2] \times [y_\alpha \pm \Delta/2]$. (Treating it as
$y = Hx + v$ with white $v$ of variance $\Delta^2/12$ fails: while an angle moves slowly its rounding error repeats
from sample to sample, a deterministic sawtooth [19, App. E], and a filter that takes each repeat as new
information grows far too confident.) Given the predicted belief $x \sim N(\hat x^-, P^-)$, the angles have the prior
$z \sim N(H\hat x^-, S)$ with $S = H P^- H^T$. Let $m$ and $V$ be the mean and covariance of that bivariate normal
truncated to $C$. Since $x$ given $z$ is normal with a mean linear in $z$ and a covariance independent of it
[25, Lemma A.2], the laws of total expectation and variance give the exact posterior moments [24, Prop. 1]:

```math
K = P^- H^T S^{-1}, \qquad \hat x = \hat x^- + K\thinspace(m - H\hat x^-), \qquad P = P^- - K\thinspace(S - V)\thinspace K^T .
```

$P$ is positive semidefinite by construction: $P^- - K S K^T$ is a Schur complement of the joint covariance of
$(x, z)$, and $K V K^T$ is a covariance. The belief is then taken as Gaussian again with these moments
(moment matching, an approximation: see below).

- **The truncated moments** are in closed form [22, eqs. (10), (11), (16)–(18), (21)]: with standardized limits,
  everything but the rectangle's probability $\alpha_C$ needs only the univariate normal density and CDF.
- **The rectangle probability** comes from the bivariate normal CDF by inclusion–exclusion of its four corners,
  the CDF from Owen's T function [23]:
  $\Phi_2(h, k; \rho) = \tfrac12\Phi(h) + \tfrac12\Phi(k) - T(h, a_h) - T(k, a_k) - \beta$ with
  $a_h = (k - \rho h)/(h\sqrt{1-\rho^2})$, $a_k$ likewise, and $\beta = \tfrac12$ when $h$ and $k$ have opposite signs, else 0.
  SciPy's `owens_t` is deterministic and accurate to about 1e-14; SciPy's own bivariate CDF is not used, as it
  integrates by randomized quasi-Monte Carlo to a tolerance of 1e-5.
- **Against cancellation**, two exact identities, not taken from the literature: a coordinate whose interval
  lies above the prior's mean is mirrored first ($z_k \to -z_k$, flipping the sign of $\rho$), so the four corner
  CDFs are small; and $\Phi(b) - \Phi(a)$ is evaluated as $\Phi(-a) - \Phi(-b)$ when $a \gt 0$.
- **Checked** against 2-D quadrature and 40-digit arithmetic: agreement to about 1e-14.

**Within each 1 ms sample**, in order: read the encoders, update the estimate, run the controller on it, then
predict to the next sample with the torque it chose. The filter starts from the first reading with zero
velocities, $P_0 = \mathrm{diag}(\Delta^2/12, \Delta^2/12, 1, 1)$ (rad², rad²/s²): the angles spread uniformly over one
count, and the velocities only roughly known, since it is not told that the rig starts at rest.

**Is the filter consistent?** Against the true state, the normalised estimation error squared
$\text{NEES} = (\hat x - x)^T P^{-1} (\hat x - x)$ should follow a $\chi^2$ distribution with 4 degrees of freedom if
$P$ is right [20, §5.4]. Each run plots it against its two-sided 95% bounds and prints the fraction of samples
inside: a diagnostic, not a check. Each run also plots $\alpha_C$, the probability the filter gave the cell it
then read, and **fails a check** if it ever drops below `min_cell_prob` = 1e-6: the reading was a big surprise,
the filter's belief was wrong, and the truncated moments also start losing accuracy there.

**Known limitation: repeated readings are still double counted.** The encoder has no noise, so in exact Bayes a
second identical reading changes nothing. After each cut, though, the filter re-fits a Gaussian and forgets
that its belief was already confined to the cell, so the next identical reading trims the Gaussian's tails
again. On a still pendulum with no process noise, the belief's std after 10, 100 and 1000 identical readings is
0.58, 0.45 and 0.38 of the exact $\Delta/\sqrt{12}$ (with the white-noise update: 0.32, 0.10, 0.03). Every filter
whose belief is a single Gaussian shares this; the established exact remedies carry the cell in the belief
(particle filters, set-membership bounds) or change the measurement (updates at the instants a count edge is
crossed). The effect grows with the time spent on one reading: under the NMPC of section 10, whose torque is
smoother than the PID's, the pendulum's reading changes 28 times a second while balancing against the PID's 69,
and the filter's smallest cell probability falls from 9.1e-5 to 2.9e-6, near the check.

**Optional: a push state.** `config.EKF["disturbance"] = dict(tau, std)` adds an unknown torque $d$ on the
pendulum's joint as a fifth state, a first-order Gauss-Markov process $\dot d = -d/\tau + w_d$ [34] of stationary
standard deviation `std` (so $w_d$ has intensity $2\,\mathrm{std}^2/\tau$): the disturbance augmentation of
offset-free estimation [28, §1.5, §5.5] with a disturbance that fades rather than a random walk. It is off. Tried
against taps ($\tau$ = 50 ms, std 1–20 mN m), it let the energy swing-up + PID survive taps up to 20 mN m at 2.3×
the chatter (std 5 mN m), but with the NMPC every setting made balancing itself break: the filter explains
quantization steps as pushes, and the double counting above makes the resulting surprises run away. A rare,
brief push is what a single constant-intensity model cannot express; a two-mode (quiet / pushed) filter is
the documented next step.

## 8. Friction identification, step 1: the steady-state friction curve

`sysid.py` (settings `config.SYSID`) estimates the arm's friction from data the rig could record, its encoder
readings and the commanded torque, the way Quanser measured the real motor's friction [5].

**The idea.** The arm's equation of motion reads $J\thinspace \dot\omega = u - \tau_f(\omega) - \tau_{pend}$, with $\tau_{pend}$
the torque the pendulum exerts on the arm. Hold $u$ constant and wait: the arm settles at a constant speed and
the pendulum settles too, so arm and pendulum turn as one rigid body at constant speed, $\tau_{pend} = 0$, and

```math
u = \tau_f(\omega) .
```

Each torque level gives one point of the friction curve. Above
$\omega = \sqrt{m_p g d_p / (m_p d_p^2 + I_{p,yy})}$ = 10.7 rad/s, hanging straight down stops being stable and the
pendulum settles on a cone instead, with $\cos\alpha = m_p g d_p / ((m_p d_p^2 + I_{p,yy})\thinspace\omega^2)$ (derived from the
equations of motion of section 1); the reduction to $u = \tau_f(\omega)$ still holds there.

**The experiment.** From rest, with no controller, the torque steps through 20 log-spaced magnitudes per
direction, from 3e-4 to 3e-2 N m (steady speeds from about 0.1 to 30 rad/s), ordered from $-u_{max}$ to $+u_{max}$
as in Quanser's test [5], so that each step starts near the previous steady state. Each level is held for 20 s,
then the speed is measured over the next 10 s from the arm's encoder alone:
$\hat\omega = (y_\theta(t_2) - y_\theta(t_1))/(t_2 - t_1)$, in error by at most one count per window, 3.1e-4 rad/s.

**Checks**, reported like `run.py`'s: the energy balance of section 2, and steady state, i.e. the speeds over the
averaging window's two halves agree within their resolution.

**The fit.** Viscous + Coulomb friction, $\tau_f(\omega) = b\thinspace\omega + \tau_C\thinspace\mathrm{sign}(\omega)$, is linear in
$\theta = (b, \tau_C)$, so linear least squares gives it in closed form [26, §7.3]:

```math
\hat\theta = (\Phi^T \Phi)^{-1} \Phi^T U, \qquad \Phi = \begin{bmatrix} \hat\omega_i & \mathrm{sign}(\hat\omega_i) \end{bmatrix}_{i=1..N},
\qquad U = [u_i]_{i=1..N},
```

with the standard error estimate $\lambda (\Phi^T\Phi)^{-1}$, $\lambda = |U - \Phi\hat\theta|^2/(N-2)$ [26, App. II]. That estimate
assumes white residuals; here they are mostly model error, so the actual error can be larger.

**The model class does not contain the truth.** The simulated truth has the smooth Stribeck law of section 1.
Above about 0.4 rad/s it equals viscous + Coulomb (its Stribeck term has died out), but below that it rises to
a breakaway peak that a straight line plus a step cannot follow. Where the experiment puts its points decides
how that mismatch spreads into $\hat b$ and $\hat\tau_C$. Static least squares is the prediction-error method with a
predictor that has no dynamics, and maximum likelihood for i.i.d. Gaussian errors [26, §7.2–7.4]; step 2 will
use the transients too.

## 9. Friction identification, step 2a: the likelihood surface

Step 1 used steady states only. Step 2 uses the transients too, by maximum likelihood through the EKF of
section 7; step 2a only maps the likelihood, before any optimizer.

**The likelihood.** By the prediction-error decomposition [25, §12.1.1, eqs. (12.5)–(12.6)], the probability of
the whole record is the product of one-step predictive probabilities:

```math
p(y_{1:N} \mid \theta) = \prod_{k} p(y_k \mid y_{1:k-1}, \theta),
\qquad p(y_k \mid y_{1:k-1}) = \int p(y_k \mid x)\thinspace p(x \mid y_{1:k-1})\thinspace dx .
```

The encoder is noise-free, so $p(y_k \mid x) = 1$ if $Hx$ lies in the reading's cell $C_k$ and 0 otherwise, and
the integral is the probability of $C_k$ under the predicted belief. Under the EKF's Gaussian prediction that is
exactly the cell probability $\alpha_{C,k}$ of section 7, so

```math
-\log p(y_{1:N} \mid \theta) = -\sum_k \log \alpha_{C,k}(\theta) .
```

For cells small against the prediction's spread this tends to the Gaussian-innovation likelihood of
Kristensen, Madsen and Jørgensen (2004) with $R = S$, plus a constant; ours are not small. The filter's
Gaussian approximation biases such likelihoods and makes them too sharp [25, §12.3.3].

**What is estimated.** $\theta = (b, \tau_C)$: the EKF's arm friction is viscous + Coulomb, i.e. section 1's law with
breakaway = Coulomb, so that its Stribeck term vanishes; the truth keeps its Stribeck law, outside that class.
$Q_c$ stays fixed at 1e-7 (`config.LIKELIHOOD`, the EKF's value when step 2a was run): the truth has no process noise, so maximum likelihood would push $Q_c$
towards 0 until the filter breaks.

**The experiment.** One open-loop record from rest (open loop, so the input is not correlated with the model
error through feedback): torque pulses of 0.5 s, each followed by 0.5 s at zero, with amplitudes uniform in
±6e-3 N m from a seeded generator, for 300 s. Only the torques and the encoder readings are kept.

**The surface.** The EKF runs over the record for each $(b, \tau_C)$ of a 7×7 grid spanning 0.5–1.5 times step 1's
$b$ and 0–2.5 times its $\tau_C$ (`config.LIKELIHOOD`), the grid points in parallel processes. A point where the
filter gives a read cell no positive probability is reported as broken. The truth here is deterministic: the
curvature of the surface says how sharp the likelihood is under the assumed $Q_c$, not how far the minimum is
from the truth.

**Known limitation: through this filter the likelihood is biased.** A known-answer diagnostic (`truth_arm` in
`config.LIKELIHOOD`, which puts the truth inside the model class on purpose) still puts the minimum at about
+13% on $b$ and −66% on $\tau_C$ at $Q_c$ = 1e-7, so the bias comes from the filter's Gaussian approximation, not from
the Stribeck mismatch. At $Q_c$ = 1e-8 the minimum sits within one grid cell of the truth, but the filter then
breaks for most parameters away from it. Step 1 remains the estimate to trust.

## 10. Nonlinear MPC

One law swings the pendulum up from hanging rest and balances it, with no switching (`control.nmpc`, settings
`config.NMPC`). Every $\Delta$ = 5 ms it solves, from the state it sees ($x$, or the EKF's $\hat x$) and the torque
$u_{-1}$ it applied the sample before,

```math
\min_{u_0, \dots, u_{N-1}} \;\Delta \sum_{k=0}^{N-1} \left[ \frac{2(1 + \cos\alpha_k)}{e_{max}^2} + \frac{2(1 - \cos\theta_k)}{\theta_{max}^2}
+ \left(\frac{u_k}{u_{max}}\right)^2 + \left(\frac{u_k - u_{k-1}}{\delta u_{max}}\right)^2 \right]
\quad \text{s.t.}\quad x_0 = x,\;\; x_{k+1} = F(x_k, u_k),\;\; |u_k| \le u_{max},
```

and applies $u_0$ for the next 5 ms; the plant and the EKF keep their 1 ms step.

- **Sample period.** 20–40× the closed-loop bandwidth [31], the bandwidth about twice the upright pendulum's
  unstable pole, 12.9 rad/s [32], gives 6–12 ms; 5 ms, the fast end, worked on a real Furuta pendulum whose
  swing-up failed above 9 ms [30].
- **Horizon** $N\Delta$ = 0.5 s ($N$ = 100). Published swing-up NMPCs predict about the whole swing-up [30]; this one
  swings up in about 0.55 s. Growing the task from balancing at 170° to swing-up from rest: 0.25 s balanced but
  left the arm unsettled, 0.5 s settled it and caught the pendulum from 150°, 120°, 90° and hanging rest.
- **Prediction model.** The equations of motion of section 1 with the EKF's friction (section 7: the arm's as
  identified in section 8), never the truth's Stribeck law. $F$ is 3 RK4 steps of 5/3 ms with $u$ held: against a
  fine-step reference, 1.5 s predictions along the energy swing-up erred by at most 0.045° (a quarter of an
  encoder count), 2 steps by 0.146°.
- **Cost.** Each term is a quantity over its largest acceptable value, squared (Bryson's rule [16]), with the PID
  tuner's scales of section 4: $e_{max}$ = 5°, $\theta_{max}$ = 30°, $u_{max}$ = 0.05 N m, so both controllers chase the same
  objective. Near upright $2(1 + \cos\alpha) \approx (\alpha - \pi)^2$ and $2(1 - \cos\theta) \approx \theta^2$, yet the terms are
  periodic: a whole turn of either angle costs nothing, and no angle needs wrapping. $\delta u_{max}$ = 0.025 N m per
  sample (the full range in about 10 ms) penalizes chatter. Velocities are not weighted.
- **No terminal cost or constraint**: stability then rests on the horizon being long enough [29].
- **Solver.** Multiple shooting [27]: the predicted states are unknowns too, tied by the dynamics. IPOPT [14]
  solves each problem to convergence (tolerance 1e-7) with the exact Hessian. CasADi [2] compiles the problem's
  functions to C once (gcc, about 2 min) and caches the library in the system's temporary folder. The first
  solve starts from states on a straight line to the nearest upright, at rest, torque 0 [33], which picks the
  swing direction (at hanging rest the cost is flat by symmetry). Every later solve starts from the previous
  solution shifted by one sample, multipliers included, with a small initial barrier parameter, so IPOPT warm-
  starts near it: about 3 iterations per sample once the swing-up is under way.
- **Failures.** A solve that does not converge is reported as a failed check with IPOPT's status; from then on
  the controller gives NaN, so the run blows up there and is plotted up to that moment.
- **Run time** on a laptop, one core: about 1 min for a 10 s run on the true state, about 2.5 min with the EKF.

## References

1. B. S. Cazzolato, Z. Prime, "On the dynamics of the Furuta pendulum", *Journal of Control Science and
   Engineering*, vol. 2011, art. 528341, 2011. doi:10.1155/2011/528341
2. J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, M. Diehl, "CasADi: a software framework for
   nonlinear optimization and optimal control", *Mathematical Programming Computation* 11(1), 1–36, 2019.
3. H. Goldstein, C. Poole, J. Safko, *Classical Mechanics*, 3rd ed., Addison-Wesley, 2002, ch. 1, 2, 4, 5.
4. MathWorks, "Rotational Friction", Simscape block documentation, R2024b. Cites B. Armstrong,
   C. Canudas de Wit, "Friction modeling and compensation", *The Control Handbook*, CRC Press, 1995.
5. M. Lévis, "Leveraging the Simulink Simscape to model the QUBE-Servo 2 with friction", Quanser blog,
   12 July 2021.
6. Quanser, *QUBE-Servo 2 User Manual*, v1.0, 2016, Table 2.2.
7. Quanser, `qube3_rotpen_param.m`, Quanser_Academic_Resources (GitHub, branch dev-windows),
   `6_teaching/1_Controls/Qube_Servo_3/sp6_pendulum_control/1b_lqr_control/hardware/matlab/`.
8. E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*,
   2nd ed., Springer, 1993, §II.1.
9. J. Apkarian, M. Lévis, *Student Workbook: QUBE-Servo Experiment for MATLAB/Simulink Users*, v2.0, Quanser,
   2014: State Space Modeling lab, eqs. (1.1)–(1.2); Rotary Pendulum Modeling lab, eq. (1.1); Swing-Up Control
   lab, eq. (1.6); Optimal LQR Control lab.
10. Quanser, *QUBE-Servo 2 Product Information Sheet*, v1.0, 2016.
11. K. J. Åström, R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, 2nd ed.,
    Princeton University Press, 2021. Section and equation numbers from the authors' electronic edition, PID
    chapter v3.0j (2019): §11.5, eqs. (11.15) and (11.17).
12. K. J. Åström, T. Hägglund, *Advanced PID Control*, ISA, 2006: §3.5 (conditional integration, pp. 84–85);
    §13.3, eq. (13.7).
13. Quanser, Quanser_Academic_Resources (GitHub, branch dev-windows): the Qube-Servo 3 pendulum control models
    in `6_teaching/1_Controls/Qube_Servo_3/sp6_pendulum_control/` (fixed step 0.002 s), and
    `0_libraries/python/pal/products/qube.py` (class `QubeServo2`, default 500 Hz).
14. A. Wächter, L. T. Biegler, "On the implementation of an interior-point filter line-search algorithm for
    large-scale nonlinear programming", *Mathematical Programming* 106(1), 25–57, 2006.
15. J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, §7.2.
16. A. E. Bryson, Y.-C. Ho, *Applied Optimal Control*, Hemisphere, 1975 (the rule of choosing quadratic-cost
    weights as 1/(largest acceptable value)²; page not verified).
17. K. J. Åström, K. Furuta, "Swinging up a pendulum by energy control", *Automatica* 36(2), 287–295, 2000.
18. S. Särkkä, A. Solin, *Applied Stochastic Differential Equations*, Cambridge University Press, 2019, §10.7,
    Algorithm 10.24 (continuous-discrete EKF).
19. B. Widrow, I. Kollár, *Quantization Noise*, Cambridge University Press, 2008: §4.2 (the uniform noise
    model), §4.3 (Quantizing Theorems I and II), Appendix E (the error conditioned on the input).
20. Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley,
    2001, §5.4 (consistency of state estimators: NEES and NIS).
21. Quanser, *QUBE-Servo 2 Product Information Sheet*, 2017, "Device Specifications"; Quanser, *Qube-Servo 3
    User Manual*, v1.1, 2024, §2.2.2 and Table 2.2 (the same encoders).
22. B. G. Manjunath, S. Wilhelm, "Moments calculation for the doubly truncated multivariate normal density",
    arXiv:1206.5387v1, 2012, eqs. (10), (11), (16)–(18), (21), after G. M. Tallis, *J. R. Stat. Soc. B* 23(1),
    1961. Journal version: *J. Behavioral Data Science* 1(1), 17–33, 2021 (its equation numbers not checked).
23. D. B. Owen, "A table of normal integrals", *Communications in Statistics: Simulation and Computation* 9(4),
    389–419, 1980, p. 416, formula 3.1, as given in D. Komelj, *American J. Computational Mathematics* 13,
    476–504, 2023 (arXiv:2312.00011), eqs. (13)–(14); Owen's paper itself not seen. Owen's T as computed by
    SciPy: M. Patefield, D. Tandy, *J. Statistical Software* 5(5), 2000.
24. E. J. Msechu, S. I. Roumeliotis, A. Ribeiro, G. B. Giannakis, "Decentralized quantized Kalman filtering with
    scalable communication cost", *IEEE Trans. Signal Processing* 56(8), 3727–3741, 2008, Proposition 1,
    eqs. (18)–(21). The approach goes back to R. E. Curry, W. E. Vander Velde, J. E. Potter, *IEEE Trans.
    Information Theory* 16(2), 152–161, 1970.
25. S. Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013, Appendix A.1, Lemma A.2,
    eqs. (A.4)–(A.5) (the conditional distribution of jointly Gaussian variables); §12.1.1, eqs. (12.5)–(12.6)
    (the prediction-error decomposition of the likelihood); §12.3.3 (bias of Gaussian-filter likelihoods).
26. L. Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999: §7.2–7.4 (prediction-error
    methods, linear regression and least squares, p. 203; maximum likelihood, p. 212), Appendix II (pp. 543, 551).
27. H. G. Bock, K. J. Plitt, "A multiple shooting algorithm for direct solution of optimal control problems",
    *IFAC Proceedings Volumes* 17(2), 1603–1608, 1984.
28. J. B. Rawlings, D. Q. Mayne, M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed.,
    Nob Hill Publishing, 2017: §1.5 and §5.5 (disturbance models, offset-free estimation).
29. L. Grüne, "NMPC without terminal constraints", *IFAC Proceedings Volumes* 45(17), 1–13, 2012.
    doi:10.3182/20120823-5-NL-3013.00030
30. H. Homburger, J. Frey, S. Wirtensohn, M. Diehl, J. Reuter, "Millisecond NMPC for swing-up and stabilization of
    the Furuta pendulum in real world", *IEEE Transactions on Control Systems Technology*, 2025.
    doi:10.1109/TCST.2025.3616367
31. G. F. Franklin, J. D. Powell, M. L. Workman, *Digital Control of Dynamic Systems*, 3rd ed., Addison-Wesley, 1998:
    the chapter on sample-rate selection.
32. G. Stein, "Respect the unstable", *IEEE Control Systems Magazine* 23(4), 12–25, 2003.
33. J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed., SIAM,
    2010 (initial guesses by interpolation between the boundary conditions).
34. R. A. Singer, "Estimating optimal tracking filter performance for manned maneuvering targets", *IEEE
    Transactions on Aerospace and Electronic Systems* AES-6(4), 473–483, 1970.
