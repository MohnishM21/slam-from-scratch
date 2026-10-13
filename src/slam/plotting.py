"""README figures, each drawn for GitHub's light and dark themes.

Colors follow a fixed categorical order checked for color-vision deficiency on
both backgrounds: graph SLAM is always blue, the EKF orange, dead reckoning aqua
(also dashed), and ground truth neutral gray. Legends are always present.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from .metrics import align_se2, nees  # noqa: E402
from .mrclam import load  # noqa: E402

THEMES = {
    "light": {"surface": "#ffffff", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e8e7e3", "faint": "#c9c8c3",
              "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "crash": "#f0efec",
              "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]},
    "dark": {"surface": "#0d1117", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#262b33", "faint": "#4a4a46",
             "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500", "crash": "#383835",
             "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]},
}


def _apply(theme):
    th = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"], "savefig.facecolor": th["surface"],
        "text.color": th["text"], "axes.labelcolor": th["muted"], "axes.edgecolor": th["grid"],
        "xtick.color": th["muted"], "ytick.color": th["muted"], "axes.grid": True, "grid.color": th["grid"],
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
    })
    return th


def _save(fig, out_dir, stem, theme):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}_{theme}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _legend_below(fig, ax, th, pad=0.1, handles=None, labels=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, pad, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9.5, labelcolor=th["text"])


def _style(th, method):
    if method.startswith("Graph"):
        return {"color": th["blue"], "linestyle": "-"}
    if method == "Dead reckoning":
        return {"color": th["aqua"], "linestyle": "--"}
    if method == "EKF + gate + bias":
        return {"color": th["yellow"], "linestyle": "-"}
    return {"color": th["orange"], "linestyle": "-"}


# ------------------------------------------------------------------ MRCLAM

def _aligned(run, data):
    gt = data.gt_at(run["t"])
    R, t = align_se2(run["pose"][:, :2], gt[:, :2])
    traj = (R @ run["pose"][:, :2].T).T + t
    lms = None if run["landmarks"] is None else (R @ run["landmarks"].T).T + t
    return gt, traj, lms


def mrclam_error_time(saved, data_dir, out_dir, theme, robots=(2, 5), methods=None):
    th = _apply(theme)
    methods = methods or ["Dead reckoning", "EKF + gate", saved_graph_name(saved)]
    fig, axes = plt.subplots(1, len(robots), figsize=(10.5, 3.6), sharey=True)
    for ax, robot in zip(axes, robots):
        data = load(data_dir, robot)
        for m in methods:
            run = next(r for r in saved["runs"] if r["robot"] == robot and r["method"] == m)
            gt, traj, _ = _aligned(run, data)
            err = np.linalg.norm(traj - gt[:, :2], axis=1)
            ax.plot(run["t"] - data.t0, np.maximum(err, 1e-3), label=m, **_style(th, m))
        ax.set_yscale("log")
        ax.set_title(f"Robot {robot}", loc="left")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("position error after alignment (m)")
    _legend_below(fig, axes[0], th, 0.12)
    _save(fig, out_dir, "mrclam_error_time", theme)


def saved_graph_name(saved):
    names = [r["method"] for r in saved["runs"] if r["method"].startswith("Graph")]
    robust = [n for n in names if "least squares" not in n]
    return (robust or names)[0]


def mrclam_map(saved, data_dir, out_dir, theme, robot=5):
    th = _apply(theme)
    data = load(data_dir, robot)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(*data.gt_at(np.arange(data.t0, data.odometry[-1, 0], 0.2))[:, :2].T, color=th["faint"], lw=0.8,
            label="robot path (ground truth)")
    ax.scatter(*data.landmarks.T, s=90, facecolors="none", edgecolors=th["text"], linewidths=1.4,
               label="landmarks (surveyed)", zorder=3)
    for m, marker in [("EKF + gate", "x"), (saved_graph_name(saved), "+")]:
        run = next(r for r in saved["runs"] if r["robot"] == robot and r["method"] == m)
        _, _, lms = _aligned(run, data)
        ax.scatter(*lms[run["seen"]].T, marker=marker, s=70, linewidths=2, color=_style(th, m)["color"],
                   label=m, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Robot {robot}: estimated landmarks after alignment", loc="left")
    _legend_below(fig, ax, th, 0.12)
    _save(fig, out_dir, "mrclam_map", theme)


def mrclam_nees(saved, data_dir, out_dir, theme, methods=("EKF + gate", "Dead reckoning", "EKF + gate + bias")):
    th = _apply(theme)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.arange(0, 1500, 25)
    for m in methods:
        per_bin = [[] for _ in bins]
        for robot in range(1, 6):
            run = next(r for r in saved["runs"] if r["robot"] == robot and r["method"] == m)
            data = load(data_dir, robot)
            n = nees(run["pose"], run["cov"], data.gt_at(run["t"]))
            idx = np.clip(((run["t"] - data.t0) // 25).astype(int), 0, len(bins) - 1)
            for k, v in zip(idx, n):
                per_bin[k].append(v)
        ax.plot(bins + 12.5, [np.mean(v) if v else np.nan for v in per_bin], label=m, **_style(th, m))
    ax.axhline(3.0, color=th["muted"], lw=1.2, ls=":")
    ax.text(1490, 3.3, "consistent = 3", color=th["muted"], fontsize=9, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean NEES over 5 robots")
    ax.set_title("Is the filter's uncertainty honest?", loc="left")
    _legend_below(fig, ax, th, 0.14)
    _save(fig, out_dir, "mrclam_nees", theme)


def heatmap(values, row_labels, col_labels, title, out_dir, stem, theme, fmt="{:.2f}"):
    th = _apply(theme)
    grid = np.array(values, float)
    finite = grid[np.isfinite(grid)]
    norm = LogNorm(vmin=finite.min(), vmax=finite.max())
    cmap = LinearSegmentedColormap.from_list("ramp", th["ramp"])
    fig, ax = plt.subplots(figsize=(1.25 * len(col_labels) + 2.6, 0.5 * len(row_labels) + 1.3))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = grid[i, j]
            face = cmap(norm(v)) if np.isfinite(v) else th["crash"]
            ink = "#0b0b0b" if np.dot(face[:3], [0.2126, 0.7152, 0.0722]) > 0.45 else "#ffffff"
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, facecolor=face, edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, fmt.format(v) if np.isfinite(v) else "n/a", ha="center", va="center",
                    color=ink, fontsize=9.5)
    ax.set_xlim(0, len(col_labels))
    ax.set_ylim(len(row_labels), 0)
    ax.set_xticks(np.arange(len(col_labels)) + 0.5, col_labels)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(row_labels)) + 0.5, row_labels)
    ax.tick_params(length=0, labelsize=9.5)
    for label in ax.get_xticklabels():
        label.set_color(th["text"])
        label.set_fontweight("bold")
    ax.set_title(title, loc="left", pad=26, color=th["text"])
    fig.tight_layout()
    _save(fig, out_dir, stem, theme)


def mrclam_figures(saved, rows, data_dir, out_dir: Path):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    values = [[next(r["ate_aligned_m"] for r in rows if r["robot"] == robot and r["method"] == m)
               for robot in range(1, 6)] for m in methods]
    for theme in THEMES:
        heatmap(values, methods, [f"Robot {k}" for k in range(1, 6)],
                "Trajectory error after alignment (m), lower is better", out_dir, "mrclam_summary", theme)
        mrclam_error_time(saved, data_dir, out_dir, theme)
        mrclam_map(saved, data_dir, out_dir, theme)
        mrclam_nees(saved, data_dir, out_dir, theme)
    print(f"figures written to {out_dir}")


# ------------------------------------------------------------------ pose graphs

def _draw_graph(ax, poses, between, th, color):
    loops = np.abs(between.j - between.i) != 1
    for i, j in zip(between.i[loops], between.j[loops]):
        ax.plot(poses[[i, j], 0], poses[[i, j], 1], color=th["faint"], lw=0.4, zorder=1)
    ax.plot(poses[:, 0], poses[:, 1], color=color, lw=0.9, zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])


def pose_graph_panels(name, panels, between, out_dir, theme):
    """panels: list of (title, poses)."""
    th = _apply(theme)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.5))
    for ax, (title, poses) in zip(axes, panels):
        _draw_graph(ax, poses, between, th, th["blue"] if "chordal" in title.lower() else th["orange"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, f"pose_graph_{name}", theme)


def convergence(curves, title, out_dir, stem, theme):
    """curves: list of (label, chi2 per iteration, style key)."""
    th = _apply(theme)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, chi2, key, dashed in curves:
        ax.plot(np.arange(len(chi2)), chi2, color=th[key], ls="--" if dashed else "-", marker="o", ms=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("chi-square (log scale)")
    ax.set_title(title, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, labelcolor=th["text"])
    _save(fig, out_dir, stem, theme)


def outlier_sweep(results, out_dir, theme):
    """results: {dataset: {"counts": [...], "plain": [...], "robust": [...]}}"""
    th = _apply(theme)
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.items()):
        ax.plot(r["counts"], r["plain"], marker="o", color=th["orange"], label="least squares")
        ax.plot(r["counts"], r["robust"], marker="o", color=th["blue"], label="Cauchy kernel")
        ax.set_yscale("log")
        ax.set_xlabel("false loop closures added")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("RMSE vs clean solution (m)")
    _legend_below(fig, axes[0], th, 0.14)
    _save(fig, out_dir, "outliers", theme)


def iterations_gif(name, runs, between, out_dir, theme, fps=3):
    """runs: list of (title, [poses per iteration]). Panels animate side by side."""
    th = _apply(theme)
    frames = max(len(p) for _, p in runs)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.3 * len(runs), 3.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.03, wspace=0.08)
    finals = [p[-1] for _, p in runs]
    lo = np.min([f[:, :2].min(0) for f in finals], 0) - 5
    hi = np.max([f[:, :2].max(0) for f in finals], 0) + 5

    def draw(k):
        for ax, (title, iterates) in zip(axes, runs):
            ax.cla()
            ax.grid(False)
            poses = iterates[min(k, len(iterates) - 1)]
            _draw_graph(ax, poses, between, th, th["blue"] if "chordal" in title.lower() else th["orange"])
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            done = k >= len(iterates) - 1
            ax.set_title(f"{title}\niteration {min(k, len(iterates) - 1)}" + ("  (converged)" if done else ""),
                         loc="left", fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=frames + 4, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    anim.save(out_dir / f"{name}_{theme}.gif", writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
