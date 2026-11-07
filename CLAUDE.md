# Furuta Pendulum (P1)

Toy project: a rotary (Furuta) inverted pendulum, simulated from first principles.
Background spec: Shared instructions + P1 section of `/home/dsous/Toy Project Prompts.md`.
Mode FULL BUILD: Claude writes all the code, the user reviews it.
Where this file and that doc disagree, this file wins: it records the user's later decisions.

## Scope

Current: first-principles model, open-loop simulations, PID control (pendulum and arm loops, summed), plots,
videos.
Nothing else.

Work grows in increments the user picks. The doc's stage order ("work through the stages in order") does
not apply; its P1 stages are background for whichever increment the user picks, and "stage" means
"increment".
Later features stay out of the repo (no roadmap or stage list; Claude's memory holds the list). When
an increment is done, show its results (plots, key numbers) and ask what comes next.

## Principles

- **Minimal code.** A toy that will never ship: the simplest structure that works. Simplicity is what
  keeps future features easy to plug in; abstractions for features that don't exist yet are not.
- **No corners cut on the math.** Exact, full-fidelity formulations. Simplifying the math is the
  user's call, never a default.

## Layout

Flat files in the project root. A file is created only when the work needs it.

```
config.py    every parameter and setting: plant, PID gains, EXPERIMENTS (named test matrix; each names its
             controller and that controller's settings) + RUN, output
model.py     plant: kinematics -> Lagrangian -> equations of motion (CasADi), friction, RK4 step
control.py   controllers (open_loop, pid), chosen per experiment
run.py       simulation loop per experiment, checks, key numbers
viz.py       figures (PNG) and videos (MP4), written to results/ (gitignored)
```

## Commands

```
uv run python run.py     run the experiments in config.RUN
uv add <package>         add a library, only when a feature first needs it
```

## Conventions

- SI units, angles in radians; plots and videos show degrees.
- State x = [theta, alpha, theta_dot, alpha_dot], alpha = 0 hanging down: the angle conventions of
  Cazzolato & Prime (2011), whose eq. (19) the model reproduces (pendulum body axes differ: x = pivot
  axis, z = rod). Input u = motor torque on the arm.
- The input is held constant over each sample period DT (zero-order hold); RK4 integrates each period in
  RK4_SUBSTEPS steps. DT is physical (the controller's sample period), the RK4 step numerical.
- Every run checks that the state stays finite, that E(t) - E(0) = motor work - friction loss within
  config.ENERGY_TOL, and that the friction loss never decreases. A failed check does not stop anything (the
  user's choice, 2026-11-06: the plots are how they debug): it is printed with what it means, written in red
  on the run's figure and video, and run.py ends listing the failed runs (exit status 1). A run whose state
  blows up is plotted up to that moment. Config mistakes (unknown controller, bad t_end) still raise.
  No test suite.
- Videos follow the doc's Video defaults (matplotlib animation, H.264). There is no system ffmpeg;
  viz.py points matplotlib at imageio-ffmpeg's binary.
