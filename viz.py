"""Figures (PNG) and videos (MP4) of a simulated run. Angles are shown in degrees, everything else in SI units.

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
             arm="#2a78d6", pend="#eb6834", motor="#1baf7a", energy="#eda100", friction="#e87ba4",
             critical="#d03b3b")
DARK = dict(surface="#1a1a19", ink="#f0efec", muted="#898781", grid="#2c2c2a", axis="#383835",
            arm="#3987e5", pend="#d95926", motor="#199e70", critical="#d03b3b")


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


def figure(path, title, controller, t, x, u, E, motor_work, friction_loss, residual, failed_checks):
    """Arm and pendulum angles and rates, torque (labelled with the controller) and the energy balance
    E(t) - E(0) = motor work - friction loss, against time, with any failed checks flagged at the top.
    The arm and the pendulum get separate panels: the arm can drift by hundreds of degrees while the
    pendulum moves by a few."""
    palette, fontsize, lw = LIGHT, 10, 1.5
    fig, ax = plt.subplots(6, 1, figsize=(10, 14), sharex=True, facecolor=palette["surface"], layout="constrained")
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
    ax[4].stairs(u, t, color=palette["motor"], lw=lw, baseline=None)
    ax[4].set_ylabel("motor torque [N m]")
    ax[4].set_title(controller, loc="left", fontsize=fontsize, color=palette["muted"])
    ax[5].plot(t, E - E[0], color=palette["energy"], lw=3 * lw, label="E(t) − E(0)")   # wide band: visible under the others
    ax[5].plot(t, motor_work, color=palette["motor"], lw=lw, label="motor work")
    ax[5].plot(t, -friction_loss, color=palette["friction"], lw=lw, label="− friction loss")
    ax[5].set_ylabel("energy [J]")
    ax[5].set_xlabel("time [s]")
    ax[5].set_title(f"energy balance residual: max |E − E(0) − motor work + friction loss| = {residual:.1e} J",
                    loc="left", fontsize=fontsize, color=palette["muted"])
    _legend(ax[5], palette, fontsize, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _flag(ax[0], palette, fontsize, failed_checks, width=110)
    fig.suptitle(title, color=palette["ink"], fontsize=13)
    fig.savefig(path, dpi=150, facecolor=palette["surface"])
    plt.close(fig)


def video(path, title, controller, t, x, u, failed_checks):
    """Real-time MP4: 3D view of the pendulum with a fading tip trail, over scrolling traces of the pendulum
    angle (the 3D view shows the arm) and the torque (labelled with the controller); failed checks are flagged
    above the traces."""
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
