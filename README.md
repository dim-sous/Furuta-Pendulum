# Furuta Pendulum

A rotary (Furuta) inverted pendulum, simulated from first principles in Python.

## Setup

Needs [uv](https://docs.astral.sh/uv/). From this folder:

```
uv sync
```

This creates `.venv/` with Python 3.13 (pinned in `.python-version`) and the exact versions of the
libraries listed in `pyproject.toml` (pinned in `uv.lock`).

## Running

```
uv run python run.py
```

`run.py` simulates every experiment named in `config.RUN` with the controller that experiment names, checks
the run (finite state, energy balance, dissipation), prints the key numbers, and writes a figure
(`results/<name>.png`) and a real-time video (`results/<name>.mp4`) per experiment. Every parameter and setting
lives in `config.py`; an experiment is an initial state, a duration, friction on or off, and a controller with
its settings: a torque signal `torque(t)` for the open-loop controller, gains for the PID.

| File | Role |
|---|---|
| `config.py` | parameters, PID gains, experiments, output settings |
| `model.py` | the plant: kinematics, Lagrangian, equations of motion, friction, RK4 step |
| `control.py` | controllers (`open_loop`, `pid`) |
| `run.py` | simulation loop, checks, key numbers |
| `viz.py` | figures and videos |

## 1. First-principles model and open-loop simulation

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
frame is $R_p = R_z(\theta)\,R_x(\alpha)$, with the pendulum along its $-z$ axis. The centres of mass are

$$
r_a = R_a \begin{bmatrix} d_a \\ 0 \\ 0 \end{bmatrix},\qquad
r_p = R_a \begin{bmatrix} L_a \\ 0 \\ 0 \end{bmatrix} + R_p \begin{bmatrix} 0 \\ 0 \\ -d_p \end{bmatrix},
$$

their velocities are $v = \frac{\partial r}{\partial q}\dot q$ with $q = [\theta,\ \alpha]$, and the angular
velocities in body axes follow from $[\omega]_\times = R^\top \dot R$ [3]:

$$
\omega_a = \begin{bmatrix} 0 \\ 0 \\ \dot\theta \end{bmatrix},\qquad
\omega_p = \begin{bmatrix} \dot\alpha \\ \dot\theta \sin\alpha \\ \dot\theta \cos\alpha \end{bmatrix}.
$$

`model.py` writes exactly these definitions and lets CasADi [2] differentiate them; nothing below was
expanded by hand in the code.

### Energy and equations of motion

Each body contributes translational energy of its centre of mass plus rotational energy about it, with its
full inertia tensor $I$ (about the centre of mass, in body axes) [3]. The motor rotor and hub add $J_m$:

$$
T = \tfrac12 J_m \dot\theta^2
  + \tfrac12 m_a \lVert v_a\rVert^2 + \tfrac12\, \omega_a^\top I_a\, \omega_a
  + \tfrac12 m_p \lVert v_p\rVert^2 + \tfrac12\, \omega_p^\top I_p\, \omega_p,
\qquad
V = -m_p\, g\, d_p \cos\alpha .
$$

Lagrange's equations with generalised forces [3],

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = Q,\qquad L = T - V,\qquad
Q = \begin{bmatrix} u - \tau_a(\dot\theta) \\ -\tau_p(\dot\alpha) \end{bmatrix},
$$

where $\tau_a$ and $\tau_p$ are the friction torques of the arm and pendulum joints. Expanding the time
derivative gives $M(q)\,\ddot q = Q + \frac{\partial L}{\partial q} - \frac{\partial^2 L}{\partial \dot q\,\partial q}\,\dot q$
with the mass matrix $M = \frac{\partial^2 L}{\partial \dot q^2}$; `model.dynamics` solves this for
$\ddot q$. Written out, the result is

$$
\begin{aligned}
&\big(J_0 + (m_p d_p^2 + I_{p,yy})\sin^2\alpha + I_{p,zz}\cos^2\alpha\big)\,\ddot\theta
 + m_p L_a d_p \cos\alpha\;\ddot\alpha \\
&\qquad + (m_p d_p^2 + I_{p,yy} - I_{p,zz})\sin 2\alpha\;\dot\theta\dot\alpha
 - m_p L_a d_p \sin\alpha\;\dot\alpha^2 = u - \tau_a(\dot\theta), \\[4pt]
&m_p L_a d_p \cos\alpha\;\ddot\theta + (I_{p,xx} + m_p d_p^2)\,\ddot\alpha
 - \tfrac12 (m_p d_p^2 + I_{p,yy} - I_{p,zz})\sin 2\alpha\;\dot\theta^2
 + m_p g d_p \sin\alpha = -\tau_p(\dot\alpha),
\end{aligned}
$$

with $J_0 = J_m + I_{a,zz} + m_a d_a^2 + m_p L_a^2$. This is Cazzolato & Prime's full model, their eq. (19) [1],
plus the rotor and hub inertia and friction. They point out that most published Furuta models drop the
pendulum's own inertia terms ($I_{p,yy}\sin^2\alpha$ and the velocity terms that come with it); the classic
QUBE workbook model [9] drops them too. They are kept here.

### Friction

Both joints use the smooth Stribeck law of the MathWorks Simscape "Rotational Friction" block [4], the block
Quanser used to model the QUBE-Servo 2 motor [5]:

$$
\tau(\omega) = \sqrt{2e}\,(\tau_{brk} - \tau_C)\, e^{-(\omega/\omega_{St})^2}\, \frac{\omega}{\omega_{St}}
 + \tau_C \tanh\!\Big(\frac{\omega}{\omega_C}\Big) + b\,\omega,
\qquad \omega_{St} = \sqrt2\,\omega_{brk},\quad \omega_C = \omega_{brk}/10 .
$$

In plain terms: viscous drag $b\omega$, a Coulomb level $\tau_C$ reached smoothly within about $\omega_C$,
and a Stribeck term that lifts the Coulomb level to $\tau_{brk}$ at speed $\omega_{brk}$. With the arm's values,
$b\,\omega_{brk}$ = 1e-4 N m is as large as that lift, so the total arm friction still rises monotonically with
speed; the pendulum's friction is purely viscous. The law is smooth everywhere, and it dissipates
($\tau(\omega)\,\omega \ge 0$) whenever $\tau_{brk} \ge \tau_C \ge 0$ and $b \ge 0$, as here.

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

### Numerical integration and the energy checks

The torque is held constant over each sample period of 1 ms (`DT`, zero-order hold), as a digital controller
would apply it; the hold is part of the simulated system. Within each period the state is integrated with the
classical Runge–Kutta 4 method [8] in `RK4_SUBSTEPS` = 4 steps of 0.25 ms. `DT` is therefore a physical
setting and the RK4 step a numerical one: refining the RK4 step leaves the held torque unchanged.

The energy check rests on an exact identity. Multiplying Lagrange's equations by $\dot q^\top$, and using that
$T$ is quadratic in $\dot q$ and $V$ depends on $q$ only, gives [3]

$$
\frac{d}{dt}(T + V) = \dot q^\top Q = u\,\dot\theta - \tau_a(\dot\theta)\,\dot\theta - \tau_p(\dot\alpha)\,\dot\alpha .
$$

`model.rk4_step` integrates the motor power $u\dot\theta$ and the friction power alongside the state, so
$E(t) - E(0) - (\text{motor work}) + (\text{friction loss})$ is zero up to RK4's truncation error. Every run
is flagged if its state stops being finite (an unstable closed loop can blow up; the plots then end there) or
if this residual exceeds `ENERGY_TOL` = 1e-6 J. A flag does not stop anything: it is printed, written in red
on the run's figure and video, and `run.py` ends by listing the flagged runs. With friction off and $u = 0$ this
is the classic check that energy is conserved. With friction and torque on, it also catches a sign error in
how $u$ or the friction torques enter the equations of motion. It cannot catch a sign error inside the
friction law itself, because that changes the equations and the integrated power together. A second check
covers that case: the friction loss must never decrease, since each RK4 increment of it is a positive
combination of $\tau(\omega)\,\omega \ge 0$ terms.

### Verification

- Equations of motion: at 2000 random states and torques, the accelerations from `model.py` match
  Cazzolato & Prime's eq. (19) [1], coded independently in a one-off script (not kept in the repo), to a
  relative difference of 1e-13 (machine precision), with friction on and off.
- Signs: a positive torque step turns the arm counter-clockwise ($\theta$ up) while the pendulum lags
  ($\alpha < 0$), as in [1], Fig. 2(d).
- Integration: halving the RK4 step shrinks the energy-balance residual 15–21× (4th order predicts 16×). At
  the 0.25 ms step, every open-loop run agrees with a 16× finer step to within 1.2e-7 rad in the angles and
  4.5e-7 rad/s in the rates, far below what the plots can show. With friction the trajectories converge
  irregularly until the step resolves the steep Coulomb band $|\dot\theta| < \omega_C$ at the arm's reversals.
  Changing `DT` instead changes the held torque, i.e. the sampled system itself, so it is not an accuracy
  setting.

### Experiments and results

Small-swing frequency about hanging down, with the arm free (from the linearised model): **2.061 Hz**. With
the arm clamped it would be 1.700 Hz; a free arm recoils as the pendulum swings, which lowers the pendulum's
effective inertia from $I_{p,xx} + m_p d_p^2$ to $M_{22} - M_{12}^2/M_{11}$.

| Experiment | $\theta_0$, $\alpha_0$ | Torque | Friction | Motor work [J] | Friction loss [J] | Balance residual [J] |
|---|---|---|---|---|---|---|
| frictionless release | 0°, 175° | 0 | off | 0 | 0 | 6.2e-13 |
| release near upright | 0°, 175° | 0 | on | 0 | 3.031e-2 | 4.8e-13 |
| release horizontal | 0°, 90° | 0 | on | 0 | 1.519e-2 | 8.2e-14 |
| torque pulse | 0°, 0° | 0.02 N m for 0.1 s at t = 0.5 s | on | 6.033e-3 | 6.033e-3 | 1.9e-14 |
| resonant pumping | 0°, 0° | 0.004 sin(2π·2.06 t) N m | on | 5.532e-2 | 5.352e-2 | 1.8e-13 |

**Frictionless release** ([video](results/frictionless_release.mp4)). The pendulum falls from 5° off
upright, swings through the bottom to −175° and back, periodically, while the arm twists between −15° and
+18° in reaction. Energy stays at 1.5128e-2 J to within 6.2e-13 J. The small error that remains is RK4's:
spikes during the fast passages through the bottom and a slow drift, because RK4 does not conserve energy
exactly.

![frictionless release](results/frictionless_release.png)

**Release near upright** ([video](results/release_near_upright.mp4)). Same start with friction on. The swing
decays within about 6 s and the pendulum comes to rest hanging down. Friction removes 3.031e-2 J, exactly the
potential-energy drop from the release point to hanging, $m_p g d_p (1 - \cos 175°)$.

![release near upright](results/release_near_upright.png)

**Release horizontal** ([video](results/release_horizontal.mp4)). The pendulum starts at 90°. Friction removes
$m_p g d_p$ = 1.519e-2 J by the time it hangs at rest.

![release horizontal](results/release_horizontal.png)

**Torque pulse** ([video](results/torque_pulse.mp4)). A 0.1 s pulse of 0.02 N m does 6.0e-3 J of work, mostly
spinning the arm; 1.2e-3 J of it is already lost to friction while the pulse lasts. The pendulum first swings
backwards (it lags the accelerating arm), then oscillates and decays; the arm coasts to 109° and stops. All the
motor work ends up as friction loss.

![torque pulse](results/torque_pulse.png)

**Resonant pumping** ([video](results/resonant_pumping.mp4)). A small torque at the 2.06 Hz small-swing
frequency builds the swing up to about ±27°, where friction removes per cycle what the motor puts in.

![resonant pumping](results/resonant_pumping.png)

### Assumptions and known gaps

- Rigid bodies; uniform rods; the pendulum has no inertia about its own long axis.
- Torque input: the motor's electrical dynamics and back-EMF are not modelled, so zero torque means a freely
  spinning motor. On the real rig, 0 V shorts the motor and brakes the arm (about 2.1e-4 N m s/rad).
- No torque limit is enforced. At 10 V the motor stalls at about 0.05 N m ($k_t \cdot 10\,\text{V}/R_m$); the
  open-loop experiments stay at or below 0.02 N m, and the PID of section 2 is not limited either.
- Quanser's 2016 product sheet [10] gives the pendulum as 9.5 cm pivot-to-tip, against 0.129 m in the
  manual [6]; Quanser's parameter file [7] uses 0.129 m, as here.

## 2. PID control

### The controller

Two PIDs, summed: one holds the pendulum upright, the other holds the arm at $\theta = 0$. Every 1 ms (`DT`)
the controller reads the state and sets a torque that is held until the next sample:

$$
\begin{aligned}
e_k &= \operatorname{wrap}(\pi - \alpha_k) \in [-\pi, \pi), &
I_{k+1} &= I_k + e_k\,\Delta t, \\
e_{a,k} &= 0 - \theta_k, &
I_{a,k+1} &= I_{a,k} + e_{a,k}\,\Delta t, \\
u_k &= K_p\, e_k + K_i\, I_k - K_d\, \dot\alpha_k
 + K_{p,a}\, e_{a,k} + K_{i,a}\, I_{a,k} - K_{d,a}\, \dot\theta_k . &&
\end{aligned}
$$

With the arm gains at 0 this is the pendulum-only PID. The arm error is not wrapped: whole turns of the arm
count.

- The error is measured from upright and wrapped, so the controller always takes the shorter way up. It is
  Quanser's balance angle, $\alpha \bmod 2\pi - \pi$ [9, Rotary Pendulum Modeling lab, eq. (1.1)], with the sign
  flipped so that $e$ = setpoint − measurement. The two differ only hanging straight down, where the wraps
  break the tie in opposite directions: here $e = -\pi$, while the flipped Quanser angle is $+\pi$.
- The D terms act on the measured rates $\dot\alpha$, $\dot\theta$ rather than on the errors' derivatives, i.e.
  setpoint weight $\gamma = 0$ in [11, eq. (11.15)]. For the pendulum the two agree except at the wrap's $2\pi$
  jump, where $\dot e$ would be an impulse.
- The integrals are the forward-Euler sums of the standard computer implementation [11, eq. (11.17)],
  [12, eq. (13.7)]. They have no anti-windup term, because nothing saturates.
- The controller reads the exact simulated state: no encoder quantization, noise or rate filter.
- The torque is not limited. For scale, the real motor delivers at most 0.050 N m at the recommended 10 V and
  0.075 N m at its 15 V maximum [6, §2.2.4 and Table 2.2].
- The gains are set by hand in `config.PID_GAINS` ($K_p$ in N m/rad, $K_i$ in N m/(rad s), $K_d$ in N m s/rad;
  the arm gains `*_arm` in the same units). The pendulum gains started from $K_p = K_i = K_d = 1$, the arm gains
  from 0. Quanser's own pendulum controllers sample at 500 Hz [13]; here `DT` is 1 ms.

### Experiments

| Experiment | $\theta_0$, $\alpha_0$ | Friction | Duration |
|---|---|---|---|
| PID near upright | 0°, 170° | on | 10 s |
| PID from rest | 0°, 0° | on | 10 s |

The first asks whether the PID catches a pendulum released 10° from upright; the second, whether it can swing
the pendulum up from rest hanging down and catch it.

### Results with the starting gains ($K_p = K_i = K_d = 1$)

Both runs blow up within 20 ms; their plots end at the blow-up.

- **PID near upright.** The torque flips sign at every sample and grows about 2.4× per sample: 0.18, −0.39,
  0.98, −2.3, 5.7 N m, … The state stops being finite at t = 0.018 s.
- **PID from rest.** The first torque is −π N m (the error hanging down is −π); the run blows up by
  t = 0.015 s.

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
   2014: State Space Modeling lab, eqs. (1.1)–(1.2); Rotary Pendulum Modeling lab, eq. (1.1); Optimal LQR
   Control lab.
10. Quanser, *QUBE-Servo 2 Product Information Sheet*, v1.0, 2016.
11. K. J. Åström, R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, 2nd ed.,
    Princeton University Press, 2021. Section and equation numbers from the authors' electronic edition, PID
    chapter v3.0j (2019): §11.5, eqs. (11.15) and (11.17).
12. K. J. Åström, T. Hägglund, *Advanced PID Control*, ISA, 2006, §13.3, eq. (13.7).
13. Quanser, Quanser_Academic_Resources (GitHub, branch dev-windows): the Qube-Servo 3 pendulum control models
    in `6_teaching/1_Controls/Qube_Servo_3/sp6_pendulum_control/` (fixed step 0.002 s), and
    `0_libraries/python/pal/products/qube.py` (class `QubeServo2`, default 500 Hz).
