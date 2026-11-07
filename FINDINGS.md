# Findings

Questions that came out of the simulations, worked as exercises. Claude records each finding and how it was
checked; the answers are yours.

## 1. RK4's energy error spikes at the bottom of each swing

**Observation.** In `frictionless_release` (released from rest at $\alpha_0 = 175°$, friction off, RK4 step
0.25 ms), the energy error $E(t) - E(0)$ in the bottom panel of `results/frictionless_release.png` peaks every
time the pendulum passes through the bottom of its swing, never at the top. Checked on the run: every peak is at
a bottom crossing ($t$ = 0.37, 1.12, 1.87 s, …). The pattern was the same with the earlier 1 ms step, about
250× larger.

**Question.** Why there?

### Part 1: where RK4 makes its error

**The fact you need.** One RK4 step matches the true solution's Taylor expansion up to the $\Delta t^4$ term. So
the error it makes in one step is of order $\Delta t^5$, and its size depends on the 5th time derivative of the
motion (more exactly, on terms of the same kind). Put simply: each step fits a 4th-degree polynomial in time to
the next 0.25 ms of motion, and the error is whatever that polynomial can't follow.

**Task.** In `results/frictionless_release.png`, pendulum rate panel ($\dot\alpha$):

1. Read the pendulum rate $\dot\alpha$ at the bottom of a swing and at the moment of release.
2. Convert each to degrees turned in one 0.25 ms RK4 step.
3. From that, say where in the swing the motion's high derivatives are largest, and so where RK4 makes most of
   its error.

**Your answer.**

### Part 2: why the error cancels (after Part 1)

The error doesn't pile up like a staircase. It rises into each pass through the bottom and falls almost all the
way back on the way out. Why does it cancel?

**Your answer.**

## 2. With all gains at 1, the PID blows up within 20 ms

**Observation.** `pid_near_upright` ($\alpha_0 = 170°$, $K_p = K_i = K_d = 1$, sample period 1 ms): the torque
flips sign at every sample and grows about 2.4× per sample, 0.18, −0.39, 0.98, −2.3, 5.7, −13.7 N m, … The state
stops being finite at $t$ = 0.018 s. `pid_from_rest` blows up by $t$ = 0.015 s.

**Question.** Why?

### Part 1: read the signal

A sampled signal that flips sign every sample and grows by a constant factor: where is the closed-loop pole that
produces it, in the z-plane?

**Your answer.**

### Part 2: find the culprit

**The fact you need.** Linearized about upright (friction off, from `model.py`), the pendulum obeys
$\ddot\alpha = 167.7\,(\alpha - \pi) + 3571.5\,u$, with $u$ in N m.

**Task.** Keep only the D term: $u_k = -K_d\,\dot\alpha_k$, held for one sample of $\Delta t$ = 1 ms. Over one
sample, how does $\dot\alpha_{k+1}$ relate to $\dot\alpha_k$ (ignore the small gravity term)? For which $K_d$ does
that stay bounded? Compare with $K_d = 1$.

**Your answer.**

### Part 3: whose fault is it?

Is the blow-up a property of the plant, of the PID, or of sampling it every 1 ms? What would the same three
gains do to this fast oscillation in continuous time?

**Your answer.**
