"""Figures (PNG) and videos (MP4) of a simulated run, and figures overlaying several runs. Angles are shown in degrees, everything else in SI units.

Each entity (arm, pendulum, motor, energy, friction) keeps one colour in every figure and video, in a light
variant for figures and a dark one for videos.
"""
import textwrap

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import config as c
import model

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()   # no system ffmpeg: use the bundled one

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             arm="#2a78d6", pend="#eb6834", motor="#1baf7a", energy="#eda100", friction="#e87ba4", tap="#008300",
             estimator="#8a63d2", critical="#d03b3b")
# Comparison figures overlay controllers, so there the controller is the entity: one categorical slot each, in a
# fixed order (slots 1-3 of the reference palette, which stay apart for colour-blind readers in all pairs).
CONTROLLERS = dict(swing_up=("energy swing-up + PID", "#2a78d6"), pid=("PID", "#2a78d6"), nmpc=("NMPC", "#eb6834"),
                   open_loop=("open loop", "#1baf7a"))
DARK = dict(surface="#1a1a19", ink="#f0efec", muted="#898781", grid="#2c2c2a", axis="#383835",
            arm="#3987e5", pend="#d95926", motor="#199e70", tap="#008300", critical="#d03b3b")


def _style(ax, palette, fontsize):
    ax.set_facecolor(palette["surface"])
    ax.grid(True, color=palette["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(palette["axis"])
    ax.tick_params(colors=palette["axis"], labelcolor=palette["muted"], labelsize=fontsize)
    ax.xaxis.label.set(color=palette["ink"], size=fontsize)
    ax.yaxis.label.set(color=palette["ink"], size=fontsize)


def _legend(ax, palette, fontsize, **placement):
    """Legend outside the plot area, so it never hides data."""
    ax.legend(fontsize=fontsize, labelcolor=palette["ink"], frameon=False, borderaxespad=0.3, **placement)


def _flag(ax, palette, fontsize, failed_checks, width):
    """Failed checks in red above the panel, wrapped at `width` characters, so a doubtful result is never
    mistaken for a good one."""
    if failed_checks:
        ax.set_title("\n".join(textwrap.fill(f"⚠ CHECK FAILED: {m}", width) for m in failed_checks), loc="left",
                     fontsize=fontsize, color=palette["critical"])


def figure(path, title, controller, t, x, u, d, E, motor_work, tap_work, friction_loss, residual, failed_checks,
           estimates=None):
    """Arm and pendulum angles and rates, torques (the motor's, labelled with the controller, and any tap d on the
    pendulum) and the energy balance E(t) - E(0) = motor work + tap work - friction loss, against time, with any
    failed checks flagged at the top. The arm and the pendulum get separate panels: the arm can drift by hundreds
    of degrees while the pendulum moves by a few. With estimates (run.consistency), four more panels: the
    estimation errors of the angles and of the rates, NEES against its 95% bounds, and the probability of each
    read cell under the filter's belief against the check's threshold."""
    tapped = d.any()
    palette, fontsize, lw = LIGHT, 10, 1.5
    n_panels = 10 if estimates else 6
    fig, ax = plt.subplots(n_panels, 1, figsize=(10, 14 / 6 * n_panels), sharex=True, facecolor=palette["surface"],
                           layout="constrained")
    for a in ax:
        _style(a, palette, fontsize)
    ax[0].plot(t, np.rad2deg(x[:, 0]), color=palette["arm"], lw=lw)
    ax[0].set_ylabel("arm θ [deg]")
    ax[1].plot(t, np.rad2deg(x[:, 1]), color=palette["pend"], lw=lw)
    ax[1].set_ylabel("pendulum α [deg]")
    ax[2].plot(t, np.rad2deg(x[:, 2]), color=palette["arm"], lw=lw)
    ax[2].set_ylabel("arm θ̇ [deg/s]")
    ax[3].plot(t, np.rad2deg(x[:, 3]), color=palette["pend"], lw=lw)
    ax[3].set_ylabel("pendulum α̇ [deg/s]")
    ax[4].stairs(u, t, color=palette["motor"], lw=lw, baseline=None, label="motor")
    if tapped:
        ax[4].stairs(d, t, color=palette["tap"], lw=lw, ls="--", baseline=None, label="tap on the pendulum")
        _legend(ax[4], palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax[4].set_ylabel("torque [N m]" if tapped else "motor torque [N m]")
    ax[4].set_title(controller, loc="left", fontsize=fontsize, color=palette["muted"])
    ax[5].plot(t, E - E[0], color=palette["energy"], lw=3 * lw, label="E(t) − E(0)")   # wide band: visible under the others
    ax[5].plot(t, motor_work, color=palette["motor"], lw=lw, label="motor work")
    if tapped:
        ax[5].plot(t, tap_work, color=palette["tap"], lw=lw, ls="--", label="tap work")
    ax[5].plot(t, -friction_loss, color=palette["friction"], lw=lw, label="− friction loss")
    ax[5].set_ylabel("energy [J]")
    balance = "E − E(0) − motor work" + (" − tap work" if tapped else "") + " + friction loss"
    ax[5].set_title(f"energy balance residual: max |{balance}| = {residual:.1e} J",
                    loc="left", fontsize=fontsize, color=palette["muted"])
    _legend(ax[5], palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if estimates:
        _estimation_panels(ax[6:], palette, fontsize, lw, t[:len(u)], estimates)
    ax[-1].set_xlabel("time [s]")
    _flag(ax[0], palette, fontsize, failed_checks, width=110)
    fig.suptitle(title, color=palette["ink"], fontsize=13)
    fig.savefig(path, dpi=150, facecolor=palette["surface"])
    plt.close(fig)


def comparison(path, title, runs):
    """Runs overlaid, one colour per controller (CONTROLLERS): the pendulum's distance from upright (0 to 180 deg,
    whichever way round), the arm angle, both rates, the motor torque (and the tap on the pendulum, dashed, if the
    runs have one), and the motor work done so far. If every run saw the EKF, two more panels: its errors on the
    pendulum's angle and rate. runs: {experiment name: its saved time series (run.py's .npz)}; failed checks of any
    run are flagged at the top."""
    palette, fontsize, lw = LIGHT, 10, 1.5
    estimated = all("error" in r for r in runs.values())
    n_panels = 8 if estimated else 6
    fig, ax = plt.subplots(n_panels, 1, figsize=(10, 14 / 6 * n_panels), sharex=True, facecolor=palette["surface"],
                           layout="constrained")
    for a in ax:
        _style(a, palette, fontsize)
    for name, r in runs.items():
        label, colour = CONTROLLERS[str(r["controller"])]
        t, x = r["t"], r["x"]
        ax[0].plot(t, np.rad2deg(np.abs(x[:, 1] % (2 * np.pi) - np.pi)), color=colour, lw=lw, label=label)
        ax[1].plot(t, np.rad2deg(x[:, 0]), color=colour, lw=lw)
        ax[2].plot(t, np.rad2deg(x[:, 2]), color=colour, lw=lw)
        ax[3].plot(t, np.rad2deg(x[:, 3]), color=colour, lw=lw)
        ax[4].stairs(r["u"], t, color=colour, lw=lw, baseline=None, label=label)
        ax[5].plot(t, r["motor_work"], color=colour, lw=lw)
        if estimated:
            ax[6].plot(t[:len(r["error"])], np.rad2deg(r["error"][:, 1]), color=colour, lw=lw)
            ax[7].plot(t[:len(r["error"])], np.rad2deg(r["error"][:, 3]), color=colour, lw=lw)
    first = next(iter(runs.values()))
    if first["d"].any():
        ax[4].stairs(first["d"], first["t"], color=palette["tap"], lw=lw, ls="--", baseline=None,
                     label="tap on the pendulum")
    for a, ylabel in zip(ax, ["pendulum from upright [deg]", "arm θ [deg]", "arm θ̇ [deg/s]",
                              "pendulum α̇ [deg/s]", "motor torque [N m]", "motor work [J]",
                              "EKF error, pendulum α [deg]", "EKF error, pendulum α̇ [deg/s]"]):
        a.set_ylabel(ylabel)
    _legend(ax[0], palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _legend(ax[4], palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax[-1].set_xlabel("time [s]")
    _flag(ax[0], palette, fontsize, [f"{name}: {m}" for name, r in runs.items() for m in r["failed_checks"]],
          width=110)
    fig.suptitle(title, color=palette["ink"], fontsize=13)
    fig.savefig(path, dpi=150, facecolor=palette["surface"])
    plt.close(fig)


def _estimation_panels(ax, palette, fontsize, lw, t, estimates):
    """The EKF's errors x_hat - x (angles, then rates); its NEES on a log axis between its 95% bounds (dashed),
    titled with the fraction of samples inside; and the probability of each read cell under its belief, on a log
    axis with the threshold below which a run fails a check (dashed, red)."""
    error = np.rad2deg(estimates["error"])
    for a, (i, j), unit in zip(ax[:2], [(0, 1), (2, 3)], ["deg", "deg/s"]):
        a.plot(t, error[:, i], color=palette["arm"], lw=lw, label="arm")
        a.plot(t, error[:, j], color=palette["pend"], lw=lw, label="pendulum")
        _legend(a, palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax[0].set_ylabel("angle error [deg]")
    ax[0].set_title("EKF estimate − true state", loc="left", fontsize=fontsize, color=palette["muted"])
    ax[1].set_ylabel("rate error [deg/s]")
    lo, hi = estimates["nees_bounds"]
    ax[2].plot(t[1:], estimates["nees"][1:], color=palette["estimator"], lw=lw)
    for bound in (lo, hi):
        ax[2].axhline(bound, color=palette["muted"], lw=1, ls="--")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("NEES")
    ax[2].set_title(f"NEES: {estimates['nees_inside']:.0%} of samples inside the 95% bounds [{lo:.3g}, {hi:.3g}]",
                    loc="left", fontsize=fontsize, color=palette["muted"])
    cell_prob = estimates["cell_prob"][1:]
    ax[3].plot(t[1:], cell_prob, color=palette["estimator"], lw=lw)
    ax[3].axhline(c.EKF["min_cell_prob"], color=palette["critical"], lw=1, ls="--")
    ax[3].set_yscale("log")
    ax[3].set_ylabel("cell probability")
    ax[3].set_title(f"probability of the read cell under the filter's belief: smallest {np.min(cell_prob):.1e}, "
                    f"check threshold {c.EKF['min_cell_prob']:.0e}", loc="left", fontsize=fontsize,
                    color=palette["muted"])


def friction_figure(path, t, u, x, torques, omega, k_mid, b, tau_c, true_friction, failed_checks):
    """The friction identification (sysid.py): the torque staircase and the arm speed against time, with each
    level's speed as measured from the encoder at the middle of its averaging window (samples k_mid); then the
    friction curve: the measured points (omega, torques), the fitted b omega + tau_C sign(omega) and the truth's
    true_friction(omega), over the whole range and zoomed to |omega| <= 1 rad/s."""
    palette, fontsize, lw = LIGHT, 10, 1.5
    fig, ax = plt.subplots(4, 1, figsize=(10, 15), facecolor=palette["surface"], layout="constrained")
    for a in ax:
        _style(a, palette, fontsize)
    ax[0].stairs(u, t, color=palette["motor"], lw=lw, baseline=None)
    ax[0].set_ylabel("motor torque [N m]")
    ax[0].set_title("the experiment: constant torques, no controller", loc="left", fontsize=fontsize,
                    color=palette["muted"])
    ax[1].plot(t, x[:, 2], color=palette["arm"], lw=lw, label="true arm speed")
    ax[1].plot(t[k_mid], omega, "o", color=palette["ink"], ms=4, label="measured from the encoder")
    ax[1].set_ylabel("arm θ̇ [rad/s]")
    ax[1].set_xlabel("time [s]")
    ax[1].sharex(ax[0])
    _legend(ax[1], palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for a, span in [(ax[2], 1.05 * np.max(np.abs(omega))), (ax[3], 1.0)]:
        w = np.linspace(-span, span, 2001)
        a.plot(w, [true_friction(wi) for wi in w], color=palette["friction"], lw=3 * lw, label="truth (Stribeck)")
        for side in (w[w < 0], w[w > 0]):              # the fit jumps by 2 tau_C at 0: draw each side apart
            a.plot(side, b * side + tau_c * np.sign(side), color=palette["ink"], lw=lw, ls="--",
                   label="fit: b ω + τ_C sign ω" if side[0] > 0 else None)
        inside = np.abs(omega) <= span
        a.plot(omega[inside], torques[inside], "o", color=palette["arm"], ms=5, label="measured points")
        a.set_xlabel("arm speed ω [rad/s]")
        a.set_ylabel("friction torque [N m]")
        _legend(a, palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax[2].set_title(f"friction curve: b = {b:.4e} N m s/rad, τ_C = {tau_c:.4e} N m", loc="left", fontsize=fontsize,
                    color=palette["muted"])
    ax[3].set_title("zoom: low speeds, where viscous + Coulomb cannot follow the Stribeck peak", loc="left",
                    fontsize=fontsize, color=palette["muted"])
    ax[3].set_ylim(-1.5 * np.max(np.abs(torques[np.abs(omega) <= 1.0])), 1.5 * np.max(np.abs(torques[np.abs(omega) <= 1.0])))
    _flag(ax[0], palette, fontsize, failed_checks, width=110)
    fig.suptitle("friction identification, step 1: steady states", color=palette["ink"], fontsize=13)
    fig.savefig(path, dpi=150, facecolor=palette["surface"])
    plt.close(fig)


def likelihood_figure(path, b_grid, tau_grid, nll, step1, truth, settings, failed_checks, diagnostic=False):
    """The likelihood surface (sysid.py, step 2a): the EKF's negative log-likelihood above its grid minimum, as
    filled contours over (b, tau_C), one hue from light (near the minimum) to dark; the grid points, the grid
    minimum, step 1's estimate step1 = (b, tau_C) and the truth's viscous and Coulomb values; grid points where the
    filter broke are crossed."""
    palette, fontsize = LIGHT, 10
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=palette["surface"], layout="constrained")
    _style(ax, palette, fontsize)
    finite = np.isfinite(nll)
    excess = np.where(finite, nll - np.min(nll[finite]), np.nan)
    levels = [0, 0.5, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    levels = [lv for lv in levels if lv < np.nanmax(excess)] + [np.nanmax(excess)]
    B, T = np.meshgrid(b_grid, tau_grid, indexing="ij")
    filled = ax.contourf(B, T, excess, levels=levels, cmap="Purples")
    ax.contour(B, T, excess, levels=levels, colors=palette["surface"], linewidths=0.8)
    bar = fig.colorbar(filled, ax=ax)
    bar.set_label("NLL − grid minimum", color=palette["ink"], fontsize=fontsize)
    bar.ax.tick_params(labelcolor=palette["muted"], labelsize=fontsize)
    ax.plot(B[finite], T[finite], ".", color=palette["muted"], ms=3)
    if not finite.all():
        ax.plot(B[~finite], T[~finite], "x", color=palette["critical"], ms=8, mew=2, label="EKF broke")
    i, j = np.unravel_index(np.nanargmin(excess), excess.shape)
    ax.plot(b_grid[i], tau_grid[j], "o", color=palette["ink"], ms=9, mfc="none", mew=2, label="grid minimum")
    ax.plot(*step1, "s", color=palette["arm"], ms=8, label="step 1 (steady states)")
    ax.plot(*truth, "*", color=palette["friction"], ms=14,
            label="truth (viscous, Coulomb)" if diagnostic else "truth (viscous, Coulomb; plus a Stribeck peak)")
    ax.set_xlabel("viscous b [N m s/rad]")
    ax.set_ylabel("Coulomb τ_C [N m]")
    ax.set_title(f"{settings['duration']:g} s of open-loop torque pulses, EKF Q_c = {settings['Q_c']:g} (N m)² s",
                 loc="left", fontsize=fontsize, color=palette["muted"])
    _legend(ax, palette, fontsize, loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=2)
    _flag(ax, palette, fontsize, failed_checks, width=90)
    fig.suptitle("DIAGNOSTIC, truth inside the model class (viscous + Coulomb): the likelihood surface" if diagnostic
                 else "friction identification, step 2a: the likelihood surface",
                 color=palette["critical"] if diagnostic else palette["ink"], fontsize=13)
    fig.savefig(path, dpi=150, facecolor=palette["surface"])
    plt.close(fig)


def video(path, title, controller, t, x, u, d, failed_checks):
    """Real-time MP4: 3D view of the pendulum with a fading tip trail, over scrolling traces of the pendulum
    angle (the 3D view shows the arm) and the motor torque (labelled with the controller); a dashed line marks
    the start of any tap d on the pendulum, and failed checks are flagged above the traces."""
    palette = DARK
    dt = t[1] - t[0]                                              # step of the run being drawn
    n_frames = int(round(t[-1] * c.VIDEO_FPS, 9)) + 1             # round first: 4.1 * 30 = 122.99999999999999
    frames = np.round(np.arange(n_frames) / (c.VIDEO_FPS * dt)).astype(int)   # sample shown in each frame
    pivot, tip = (np.asarray(a).T for a in model.points.map(len(t))(x[:, :2].T))

    fig = plt.figure(figsize=(c.VIDEO_PX / 100, c.VIDEO_PX / 100), dpi=100, facecolor=palette["surface"])
    ax3 = fig.add_axes((0.0, 0.31, 1.0, 0.63), projection="3d", facecolor=palette["surface"])
    axa = fig.add_axes((0.12, 0.2, 0.84, 0.11))
    axu = fig.add_axes((0.12, 0.065, 0.84, 0.11), sharex=axa)

    # 3D scene: motor post, circle swept by the arm, arm, pendulum, tip trail
    reach = np.hypot(c.L_ARM, c.L_PEND)
    z_floor = -1.25 * c.L_PEND                                    # bottom of the motor post
    ax3.set(xlim=(-reach, reach), ylim=(-reach, reach), zlim=(z_floor, c.L_PEND))
    ax3.set_box_aspect((2 * reach, 2 * reach, c.L_PEND - z_floor), zoom=1.55)   # 1.55 keeps an upright tip below the title
    ax3.view_init(elev=20, azim=-60)
    ax3.set_axis_off()
    ax3.plot([0, 0], [0, 0], [z_floor, 0], color=palette["muted"], lw=12, solid_capstyle="butt")
    ring = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(c.L_ARM * np.cos(ring), c.L_ARM * np.sin(ring), 0 * ring, color=palette["axis"], lw=1)
    arm, = ax3.plot([], [], [], color=palette["arm"], lw=7, solid_capstyle="round")
    pend, = ax3.plot([], [], [], color=palette["pend"], lw=7, solid_capstyle="round")
    joint, = ax3.plot([], [], [], "o", color=palette["ink"], ms=8)
    trail = Line3DCollection([tip[[0, 0]]], linewidths=2)        # starts as a zero-length segment at the tip
    ax3.add_collection3d(trail)
    for artist in (*ax3.lines, *ax3.collections):
        artist.set_clip_on(False)           # the 3D axes is a centred square; let the zoomed scene use the full width
    n_trail = int(round(c.TRAIL_S / dt))
    stride = max(1, int(round(0.005 / dt)))                       # a trail point every 5 ms keeps drawing cheap
    trail_rgb = matplotlib.colors.to_rgb(palette["pend"])

    # Scrolling traces
    for a in (axa, axu):
        _style(a, palette, 15)
    alpha_line, = axa.plot([], [], color=palette["pend"], lw=2)
    alpha = np.rad2deg(x[:, 1])
    axa.set_ylim(*_padded(alpha))
    axa.set_ylabel("α [deg]")
    axa.tick_params(labelbottom=False)
    _flag(axa, palette, 13, failed_checks, width=90)
    torque_line, = axu.plot([], [], color=palette["motor"], lw=2, drawstyle="steps-post")
    u = np.append(u, u[-1])                  # u[k] holds over [t_k, t_k+1); repeat the last value at t_end
    axu.set_ylim(*_padded(u))
    axu.set_ylabel("torque [N m]")
    axu.set_xlabel("time [s]")
    axu.set_title(controller, loc="left", fontsize=13, color=palette["muted"])
    if d.any():
        t_tap = t[np.flatnonzero(d)[0]]
        for a in (axa, axu):
            a.axvline(t_tap, color=palette["tap"], lw=1.5, ls="--")
        axa.text(t_tap, axa.get_ylim()[1], " tap", color=palette["muted"], fontsize=13, va="top", clip_on=True)
    fig.text(0.04, 0.965, title, color=palette["ink"], fontsize=22)
    clock = fig.text(0.96, 0.965, "", color=palette["muted"], fontsize=18, ha="right")

    writer = FFMpegWriter(fps=c.VIDEO_FPS, codec="h264")
    with writer.saving(fig, str(path), dpi=100):
        for k in frames:
            arm.set_data_3d([0, pivot[k, 0]], [0, pivot[k, 1]], [0, pivot[k, 2]])
            pend.set_data_3d([pivot[k, 0], tip[k, 0]], [pivot[k, 1], tip[k, 1]], [pivot[k, 2], tip[k, 2]])
            joint.set_data_3d([pivot[k, 0]], [pivot[k, 1]], [pivot[k, 2]])
            seg = tip[max(0, k - n_trail):k + 1][::-stride][::-1]   # every stride-th sample back from the newest
            trail.set_segments(np.stack([seg[:-1], seg[1:]], axis=1))
            trail.set_color([(*trail_rgb, a) for a in np.linspace(0, 0.8, len(seg) - 1)])
            alpha_line.set_data(t[:k + 1], alpha[:k + 1])
            torque_line.set_data(t[:k + 1], u[:k + 1])
            axa.set_xlim(max(0.0, t[k] - c.PLOT_WINDOW_S), max(c.PLOT_WINDOW_S, t[k]))
            clock.set_text(f"t = {t[k]:.2f} s")
            writer.grab_frame()
    plt.close(fig)


def _padded(y):
    """Fixed axis limits that hold the whole run, with a margin (and a span even if y is constant)."""
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi), 1e-3)
    return lo - pad, hi + pad
