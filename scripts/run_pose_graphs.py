"""Pose-graph optimization on the standard 2D benchmarks.

    python scripts/run_pose_graphs.py

1. Odometry vs chordal initialization, Gauss-Newton vs Levenberg-Marquardt, on
   INTEL, MIT Killian Court and M3500.
2. Robustness: false loop closures are injected into INTEL and M3500, and plain
   least squares is compared with a Cauchy kernel against the clean solution.
Writes results/pose_graphs.{md,csv}, results/outliers.csv and figures.
"""

import csv
import time
from pathlib import Path

import numpy as np

from slam import plotting
from slam.factor_graph import Between, Graph, optimize
from slam.g2o import is_odometry, read_g2o
from slam.initialization import chordal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pose_graphs"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
DATASETS = {"INTEL": "input_INTEL_g2o.g2o", "MIT": "input_MITb_g2o.g2o", "M3500": "input_M3500_g2o.g2o"}
OUTLIER_COUNTS = [0, 5, 10, 20, 50, 100]


def with_poses(g, poses, extra=()):
    return Graph(poses=poses.copy(), between=list(g.between) + list(extra), priors=g.priors)


def benchmark():
    rows, store = [], {}
    for name, file in DATASETS.items():
        g = read_g2o(DATA / file)
        tic = time.perf_counter()
        init_chordal = chordal(len(g.poses), g.between)
        t_chordal = time.perf_counter() - tic
        store[name] = {"graph": g, "odometry": g.poses, "chordal": init_chordal}
        for init_name, start in [("odometry", g.poses), ("chordal", init_chordal)]:
            for method in ("gn", "lm"):
                tic = time.perf_counter()
                p, _, h = optimize(with_poses(g, start), method=method, max_iter=300,
                                   keep_iterates=(name == "MIT" and method == "gn"))
                seconds = time.perf_counter() - tic + (t_chordal if init_name == "chordal" else 0.0)
                store[name][(init_name, method)] = (p, h)
                rows.append({"dataset": name, "poses": len(g.poses), "edges": len(g.between[0].i),
                             "loop_closures": int((~is_odometry(g.between[0])).sum()), "init": init_name,
                             "method": method.upper(), "chi2_initial": h[0]["chi2"], "chi2_final": h[-1]["chi2"],
                             "iterations": len(h) - 1, "seconds": seconds,
                             "converged": len(h) - 1 < 300})
                print(f"{name:6s} {init_name:8s} {method}: chi2 {h[0]['chi2']:.3e} -> {h[-1]['chi2']:.4f} "
                      f"in {len(h) - 1} its, {seconds:.2f} s")
    return rows, store


def false_loops(g, count, rng):
    n = len(g.poses)
    lc = g.between[0]
    info = lc.info[~is_odometry(lc)]
    i = rng.integers(0, n, count)
    j = (i + rng.integers(20, n - 20, count)) % n
    z = np.column_stack([rng.uniform(-5, 5, (count, 2)), rng.uniform(-np.pi, np.pi, count)])
    return Between(i, j, z, info[rng.integers(0, len(info), count)])


def outlier_sweep(store, seed=0):
    rng = np.random.default_rng(seed)
    out, rows = {}, []
    for name in ("INTEL", "M3500"):
        g = store[name]["graph"]
        clean = store[name][("chordal", "gn")][0]
        plain_err, robust_err = [], []
        for count in OUTLIER_COUNTS:
            extra = [false_loops(g, count, rng)] if count else []
            p_plain, _, _ = optimize(with_poses(g, g.poses, extra), method="gn", max_iter=100)
            p_rob, _, _ = optimize(with_poses(g, g.poses, extra), method="gn", max_iter=100,
                                   robust="cauchy", robust_k=1.0)
            rmse = lambda p: float(np.sqrt(np.mean(((p[:, :2] - clean[:, :2]) ** 2).sum(-1))))  # noqa: E731
            plain_err.append(max(rmse(p_plain), 1e-4))
            robust_err.append(max(rmse(p_rob), 1e-4))
            rows.append({"dataset": name, "false_loop_closures": count, "rmse_least_squares_m": plain_err[-1],
                         "rmse_cauchy_m": robust_err[-1]})
            print(f"{name:6s} {count:4d} false loops: least squares {plain_err[-1]:.3f} m, Cauchy {robust_err[-1]:.3f} m")
        out[name] = {"counts": OUTLIER_COUNTS, "plain": plain_err, "robust": robust_err}
    return out, rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, store = benchmark()
    write_csv(RESULTS / "pose_graphs.csv", rows)
    lines = ["| Dataset | Start | Method | Final chi-square | Iterations | Time (s) |", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        its = f"{r['iterations']}" if r["converged"] else f"{r['iterations']} (cap)"
        lines.append(f"| {r['dataset']} | {r['init']} | {r['method']} | {r['chi2_final']:.2f} | {its} | {r['seconds']:.2f} |")
    (RESULTS / "pose_graphs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sweep, sweep_rows = outlier_sweep(store)
    write_csv(RESULTS / "outliers.csv", sweep_rows)

    for theme in plotting.THEMES:
        for name in ("INTEL", "MIT"):
            s = store[name]
            b = s["graph"].between[0]
            plotting.pose_graph_panels(name.lower(), [
                ("Odometry only", s["odometry"]),
                ("Gauss-Newton from odometry", s[("odometry", "gn")][0]),
                ("Chordal initialization", s["chordal"]),
                ("Gauss-Newton from chordal", s[("chordal", "gn")][0]),
            ], b, FIGS, theme)
            curves = [(f"{m.upper()} from {i}", [x["chi2"] for x in s[(i, m)][1]],
                       {"chordal": {"gn": "blue", "lm": "aqua"}, "odometry": {"gn": "orange", "lm": "yellow"}}[i][m],
                       m == "lm") for i in ("chordal", "odometry") for m in ("gn", "lm")]
            plotting.convergence(curves, f"{name}: convergence", FIGS, f"convergence_{name.lower()}", theme)
        plotting.outlier_sweep(sweep, FIGS, theme)
        mit = store["MIT"]
        plotting.iterations_gif("mit_iterations", [
            ("MIT, Gauss-Newton from odometry", [x["poses"] for x in mit[("odometry", "gn")][1]]),
            ("MIT, Gauss-Newton from chordal", [x["poses"] for x in mit[("chordal", "gn")][1]]),
        ], mit["graph"].between[0], FIGS, theme)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
