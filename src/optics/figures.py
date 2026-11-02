"""Figures and animations for the virtual spectrometer.

Static figures:   fig_prism_setup, fig_polarisation_filters, fig_polarisation_ellipses
Animations (GIF): animate_filter_scan, animate_zeeman_field_ramp, animate_polariser_rotation,
                  animate_view_angle
Every animation returns the FuncAnimation and saves a GIF if save_path is given.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, Polygon

from ..animation import POL_COLORS
from .color import rainbow_strip, wavelength_to_rgb
from .polarization import (circular_polariser, component_jones, linear_polariser, polarisation_ellipse)
from .prism import (deviation_deg, min_deviation_deg, min_deviation_incidence_deg, screen_position_m)
from .spectrometer import LAMP_LINES, bandpass_transmission, observe, sodium_d_components

YELLOW = wavelength_to_rgb(589.3)
POL_LABEL = {"pi": "π", "sigma+": "σ+", "sigma-": "σ−"}


def _save(anim: FuncAnimation, path: str | None, fps: int):
    if path:
        anim.save(path, writer="pillow", fps=fps)
    return anim


def _sticks(ax, obs, line: str, window_pm: float, ymax: float, legend: bool = False):
    """Stick spectrum of one sodium line: x = shift in pm, height = observed intensity."""
    sub = obs[obs.line == line]
    for pol, color in POL_COLORS.items():
        s = sub[sub.polarisation == pol]
        if len(s):
            ax.vlines(s.shift_pm, 0, s.observed, color=color, lw=3.5, label=POL_LABEL[pol])
    ax.axhspan(-0.08 * ymax, 0, color=YELLOW)                      # the colour of the light itself
    ax.set_xlim(-window_pm, window_pm)
    ax.set_ylim(-0.08 * ymax, 1.12 * ymax)
    ax.set_xlabel("shift from zero-field line (pm)")
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)


# ------------------------------------------------------------------------------------------
# Static figures
# ------------------------------------------------------------------------------------------
def fig_prism_setup(glass: str = "SF11", exaggeration: float = 3.0, distance_m: float = 2.0):
    """Ray diagram of a prism splitting the lamp's lines, and what lands on the screen."""
    apex = 60.0
    inc = min_deviation_incidence_deg(589.3, apex, glass)
    d_ref = min_deviation_deg(589.3, apex, glass)
    fig = plt.figure(figsize=(10, 6.4))
    gs = GridSpec(2, 1, height_ratios=[3, 1.3], hspace=0.35)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    ax.set_facecolor("#101010")
    ax.add_patch(Polygon([(0, 0), (2, 0), (1, np.sqrt(3))], closed=True, fc="#9ecae1", ec="w", alpha=0.55))
    p1, p2 = np.array([0.5, np.sqrt(3) / 2]), np.array([1.5, np.sqrt(3) / 2])
    a_in = np.radians(d_ref / 2)
    start = p1 - 1.8 * np.array([np.cos(a_in), np.sin(a_in)])
    ax.plot(*zip(start, p1), color="0.85", lw=2.5)
    ax.plot(*zip(p1, p2), color="0.85", lw=2.0)
    xs = 4.2
    for r in LAMP_LINES.itertuples():
        d = float(deviation_deg(r.wavelength_nm, inc, apex, glass))
        ang = np.radians(d_ref / 2 - d_ref - exaggeration * (d - d_ref))
        ax.plot([p2[0], xs], [p2[1], p2[1] + (xs - p2[0]) * np.tan(ang)], color=wavelength_to_rgb(r.wavelength_nm),
                lw=0.8 + 2.2 * r.intensity)
    ax.plot([xs, xs], [-3.4, 1.6], color="w", lw=4)
    ax.text(xs + 0.08, 1.4, "screen", color="w", fontsize=9)
    ax.text(-1.7, 1.25, "lamp light\n(all lines mixed)", color="0.85", fontsize=9)
    ax.set_xlim(-1.9, 5.6)
    ax.set_ylim(-3.5, 1.9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{glass} prism, apex 60°, set at minimum deviation for 589 nm ({d_ref:.1f}°); "
                 f"angular spread drawn ×{exaggeration:g} for clarity", fontsize=10)

    ax2.set_facecolor("#101010")
    x_cm = screen_position_m(LAMP_LINES.wavelength_nm.to_numpy(), distance_m, glass=glass) * 100
    for xi, r in zip(x_cm, LAMP_LINES.itertuples()):
        ax2.vlines(xi, 0, r.intensity, color=wavelength_to_rgb(r.wavelength_nm), lw=4)
    # labels at fixed heights so the yellow cluster (Hg 577/579, Na D) does not overprint itself
    heights = {"Hg 577.0": 1.12, "Hg 579.1": 1.30, "Na D2": 1.48, "Na D1": None}
    for xi, r in zip(x_cm, LAMP_LINES.itertuples()):
        if r.name == "Na D1":
            continue
        label = "Na D1+D2" if r.name == "Na D2" else r.name
        y = heights.get(r.name, r.intensity + 0.08)
        ax2.text(xi, y, label, color="w", fontsize=7, ha="center")
        if r.name in heights:
            ax2.plot([xi, xi], [r.intensity, y - 0.02], color="0.5", lw=0.6)
    ax2.set_xlim(x_cm.min() - 1.2, x_cm.max() + 1.2)
    ax2.set_ylim(0, 1.7)
    ax2.set_yticks([])
    ax2.set_xlabel(f"position on screen (cm), true angles, screen {distance_m:g} m from prism (violet to the right)")
    ax2.set_title("What the screen shows", fontsize=10)
    return fig


def fig_polarisation_filters(b_tesla: float = 2.0):
    """Sodium D lines seen through different polarisers and viewing directions (B = 2 T by default)."""
    comps = sodium_d_components(b_tesla)
    panels = [
        ("Across B: no filter", 90, None), ("Across B: polariser ∥ B", 90, linear_polariser(0)),
        ("Across B: polariser ⊥ B", 90, linear_polariser(90)), ("Along B: no filter", 0, None),
        ("Along B: analyser passes σ+", 0, circular_polariser(+1)),
        ("Along B: analyser passes σ−", 0, circular_polariser(-1)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for k, (ax, (title, theta, element)) in enumerate(zip(axes.ravel(), panels)):
        obs = observe(comps, theta, element)
        for pol, color in POL_COLORS.items():
            s = obs[obs.polarisation == pol]
            ax.vlines(s.wavelength_nm, 0, s.observed, color=color, lw=3, label=POL_LABEL[pol])
        ax.axhspan(-0.06, 0, color=YELLOW)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center")
    axes[0, 0].set_xlim(588.85, 589.75)
    axes[0, 0].set_ylim(-0.06, 1.15)
    for ax in axes[1]:
        ax.set_xlabel("wavelength in air (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed intensity")
    fig.suptitle(f"Sodium D2 (left cluster) and D1 (right cluster) at B = {b_tesla:g} T: "
                 "what each viewing direction and filter lets through", fontsize=11)
    fig.tight_layout()
    return fig


def fig_polarisation_ellipses(thetas_deg=(90, 60, 30, 0)):
    """Polarisation of the three kinds of Zeeman component for several viewing angles."""
    fig, axes = plt.subplots(3, len(thetas_deg), figsize=(2.6 * len(thetas_deg), 7.2))
    for j, th in enumerate(thetas_deg):
        for i, (q, name) in enumerate([(0, "π"), (+1, "σ+"), (-1, "σ−")]):
            ax = axes[i, j]
            jv = component_jones(q, np.radians(th))
            e = polarisation_ellipse(jv)
            ax.plot(e[:, 0], e[:, 1], color=POL_COLORS[{0: "pi", 1: "sigma+", -1: "sigma-"}[q]], lw=2.5)
            ax.axhline(0, color="0.8", lw=0.8)
            ax.axvline(0, color="0.8", lw=0.8)
            if np.ptp(e[:, 0]) > 0.05 and np.ptp(e[:, 1]) > 0.05:       # a true ellipse: show the rotation sense
                k = 12
                ax.annotate("", xy=e[k + 3], xytext=e[k], arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=13, rotation=0, labelpad=16)
            if i == 0:
                ax.set_title(f"viewed at θ = {th}°", fontsize=10)
            ax.text(0.04, 0.04, f"I = {float(np.vdot(jv, jv).real):.2f}", transform=ax.transAxes, fontsize=8)
    fig.text(0.5, 0.005, "horizontal axis: along B (as seen on the sky)   |   vertical axis: perpendicular to B", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


# ------------------------------------------------------------------------------------------
# Animations
# ------------------------------------------------------------------------------------------
def animate_filter_scan(save_path: str | None = None, glass: str = "SF11", fwhm_nm: float = 3.0,
                        n_frames: int = 100, distance_m: float = 2.0, fps: int = 14):
    """Sweep a colour filter across the spectrum: only the lamp lines inside the pass band reach the screen."""
    lines = LAMP_LINES
    wl, inten, names = lines.wavelength_nm.to_numpy(), lines.intensity.to_numpy(), lines.name.to_list()
    x_cm = screen_position_m(wl, distance_m, glass=glass) * 100
    centres = np.linspace(395, 665, n_frames)
    wl_axis = np.linspace(380, 700, 800)
    strip = rainbow_strip(380, 700, 400, 8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), gridspec_kw={"height_ratios": [1, 1.5]})

    def draw(i):
        c = centres[i]
        t = bandpass_transmission(wl, c, fwhm_nm)
        ax1.clear()
        ax2.clear()
        ax1.set_facecolor("black")
        xa, xb = (float(screen_position_m(c + fwhm_nm, distance_m, glass=glass)) * 100,
                  float(screen_position_m(c - fwhm_nm, distance_m, glass=glass)) * 100)
        ax1.axvspan(min(xa, xb), max(xa, xb), color="w", alpha=0.18)
        for xi, name, w, ti in zip(x_cm, names, wl, t):
            ax1.vlines(xi, 0.05, 0.95, color=wavelength_to_rgb(w), lw=5, alpha=0.08 + 0.92 * ti)
        ax1.set_xlim(x_cm.min() - 1.2, x_cm.max() + 1.2)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_xlabel("position on screen (cm), violet to the right")
        passed = [n for n, ti in zip(names, t) if ti > 0.3]
        ax1.set_title(f"Filter centred at {c:.0f} nm (width {fwhm_nm:g} nm) → lines reaching the screen: "
                      + (", ".join(passed) if passed else "none"), fontsize=9)
        ax2.bar(wl, inten, width=2.2, color="0.8", label="lamp")
        ax2.bar(wl, inten * t, width=2.2, color=[wavelength_to_rgb(w) for w in wl], edgecolor="k", lw=0.4, label="after filter")
        ax2.plot(wl_axis, bandpass_transmission(wl_axis, c, fwhm_nm), "k--", lw=1, label="filter transmission")
        ax2.imshow(strip, extent=[380, 700, -0.16, -0.04], aspect="auto")
        ax2.set_xlim(380, 700)
        ax2.set_ylim(-0.16, 1.2)
        ax2.set_xlabel("wavelength (nm)")
        ax2.set_ylabel("relative intensity")
        ax2.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.7, bottom=0.14, top=0.93)    # tight_layout ran before the labels existed
    return _save(anim, save_path, fps)


def animate_zeeman_field_ramp(save_path: str | None = None, b_max: float = 3.0, n_frames: int = 60,
                              theta_deg: float = 90.0, fps: int = 12):
    """Ramp the magnetic field: Zeeman components of Na D1 and D2 fan out (weak-field regime)."""
    b_grid = np.linspace(0.02, b_max, n_frames)
    data = [observe(sodium_d_components(b), theta_deg) for b in b_grid]
    window = 5 / 3 * 27.0 * b_max * 1.15
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, hspace=0.45, wspace=0.25)
    axes = {"D1": (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])),
            "D2": (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]))}

    def draw(i):
        for line, (fan, st) in axes.items():
            fan.clear()
            st.clear()
            last = data[i][data[i].line == line].reset_index(drop=True)
            n = len(last)
            for k in range(n):
                y = np.array([d[d.line == line].shift_pm.to_numpy()[k] for d in data[: i + 1]])
                fan.plot(b_grid[: i + 1], y, color=POL_COLORS[last.polarisation[k]], lw=1.8)
            fan.set_xlim(0, b_max)
            fan.set_ylim(-window, window)
            fan.set_xlabel("magnetic field B (T)")
            fan.set_ylabel("shift (pm)")
            fan.set_title(f"Na {line}: component positions", fontsize=10)
            fan.grid(alpha=0.3)
            _sticks(st, data[i], line, window, 1.0, legend=(line == "D1"))
            st.set_title(f"spectrum at B = {b_grid[i]:.2f} T ({n} components)", fontsize=10)
        fig.suptitle("Zeeman effect in the sodium D lines, viewed across the field (weak-field regime, B* = 24.6 T)", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    return _save(anim, save_path, fps)


def animate_polariser_rotation(save_path: str | None = None, b_tesla: float = 2.0, n_frames: int = 61, fps: int = 12):
    """Rotate a linear polariser while looking across the field: π and σ lines trade places (Malus's law)."""
    comps = sodium_d_components(b_tesla)
    phis = np.linspace(0, 180, n_frames)
    window = 5 / 3 * 27.0 * b_tesla * 1.2
    tot = {pol: np.array([observe(comps, 90, linear_polariser(p)).query("polarisation == @pol").observed.sum() for p in phis])
           for pol in ("pi", "sigma+", "sigma-")}
    tot["sigma"] = tot["sigma+"] + tot["sigma-"]
    fig = plt.figure(figsize=(10, 6.2))
    gs = GridSpec(2, 3, hspace=0.45, wspace=0.35)
    a1, a2 = fig.add_subplot(gs[0, 0:1]), fig.add_subplot(gs[0, 1:2])
    acurve, ageo = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2])
    ax_leg = fig.add_subplot(gs[0, 2]); ax_leg.axis("off")

    def draw(i):
        phi = phis[i]
        obs = observe(comps, 90, linear_polariser(phi))
        for a, line in ((a1, "D1"), (a2, "D2")):
            a.clear()
            _sticks(a, obs, line, window, 1.0, legend=False)
            a.set_title(f"Na {line}", fontsize=10)
        a1.set_ylabel("observed intensity")
        ax_leg.clear(); ax_leg.axis("off")
        for k, (pol, color) in enumerate(POL_COLORS.items()):
            ax_leg.plot([], [], color=color, lw=4, label={"pi": "π  (light ∥ B)", "sigma+": "σ+ (light ⊥ B)", "sigma-": "σ− (light ⊥ B)"}[pol])
        ax_leg.legend(loc="center", fontsize=9, frameon=False)
        acurve.clear()
        acurve.plot(phis, tot["pi"], color=POL_COLORS["pi"], lw=2, label="all π lines (cos²φ)")
        acurve.plot(phis, tot["sigma"], color=POL_COLORS["sigma+"], lw=2, label="all σ lines (sin²φ)")
        acurve.plot([phi], [tot["pi"][i]], "o", color=POL_COLORS["pi"])
        acurve.plot([phi], [tot["sigma"][i]], "o", color=POL_COLORS["sigma+"])
        acurve.set_xlabel("polariser angle φ from B (degrees)")
        acurve.set_ylabel("total intensity")
        acurve.legend(fontsize=8, loc="upper center", ncol=2, frameon=False)
        acurve.grid(alpha=0.3)
        ageo.clear()
        ageo.annotate("", xy=(0, 1), xytext=(0, -1), arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        ageo.text(0.08, 0.85, "B", fontsize=12)
        r = np.radians(phi)
        ageo.plot([-np.sin(r), np.sin(r)], [-np.cos(r), np.cos(r)], color="tab:orange", lw=4, solid_capstyle="round")
        ageo.text(0, -1.45, f"polariser axis φ = {phi:.0f}°", ha="center", fontsize=9)
        ageo.set_xlim(-1.3, 1.3); ageo.set_ylim(-1.6, 1.3); ageo.set_aspect("equal"); ageo.axis("off")
        fig.suptitle(f"Looking across B at {b_tesla:g} T through a rotating polariser", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    return _save(anim, save_path, fps)


def animate_view_angle(save_path: str | None = None, b_tesla: float = 2.0, n_frames: int = 61, fps: int = 12,
                       hold: int = 6):
    """Tilt the line of sight from across B (90°) to along B (0°): π fades out, σ become circular."""
    comps = sodium_d_components(b_tesla)
    thetas = np.concatenate([np.full(hold, 90.0), np.linspace(90, 0, n_frames), np.full(hold, 0.0)])
    window = 5 / 3 * 27.0 * b_tesla * 1.2
    fig = plt.figure(figsize=(10, 6.6))
    gs = GridSpec(2, 3, hspace=0.4, wspace=0.3)
    a1, a2, ageo = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ell = [fig.add_subplot(gs[1, k]) for k in range(3)]

    def draw(i):
        th = thetas[i]
        obs = observe(comps, th)
        for a, line in ((a1, "D1"), (a2, "D2")):
            a.clear()
            _sticks(a, obs, line, window, 1.0, legend=(line == "D1"))
            a.set_title(f"Na {line}", fontsize=10)
        a1.set_ylabel("observed intensity")
        ageo.clear()
        ageo.annotate("", xy=(0, 1.1), xytext=(0, -1.1), arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        ageo.text(0.08, 0.95, "B", fontsize=12)
        r = np.radians(th)
        ageo.annotate("", xy=(1.2 * np.sin(r), 1.2 * np.cos(r)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=3))
        ageo.text(1.3 * np.sin(r), 1.3 * np.cos(r), "to observer", color="tab:orange", fontsize=8)
        ageo.add_patch(Arc((0, 0), 1.0, 1.0, theta1=90 - th, theta2=90, color="0.4"))
        ageo.text(0.1, -1.35, f"viewing angle θ = {th:.0f}°", fontsize=9)
        ageo.set_xlim(-1.4, 2.0); ageo.set_ylim(-1.6, 1.6); ageo.set_aspect("equal"); ageo.axis("off")
        for a, (q, key, name) in zip(ell, [(0, "pi", "π"), (+1, "sigma+", "σ+"), (-1, "sigma-", "σ−")]):
            a.clear()
            jv = component_jones(q, r)
            e = polarisation_ellipse(jv)
            a.plot(e[:, 0], e[:, 1], color=POL_COLORS[key], lw=3)
            a.axhline(0, color="0.8", lw=0.8); a.axvline(0, color="0.8", lw=0.8)
            a.set_xlim(-1.05, 1.05); a.set_ylim(-1.05, 1.05); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{name}: I = {float(np.vdot(jv, jv).real):.2f}", fontsize=10)
        ell[0].set_ylabel("⊥ B")
        ell[1].set_xlabel("along B →")
        fig.suptitle(f"Polarisation of the Zeeman components seen at B = {b_tesla:g} T", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(thetas), interval=1000 / fps)
    return _save(anim, save_path, fps)
