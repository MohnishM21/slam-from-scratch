"""EKF-SLAM vs graph SLAM on real robots, scored against motion-capture ground truth.

    python scripts/run_mrclam.py            # about 6 minutes
    python scripts/run_mrclam.py --plots-only

Everything that is tuned is tuned on robot 1 only: the noise model (calibration.py)
and the graph's robust kernel. Robots 2-5 are a held-out test.
Writes results/mrclam.{md,csv}, results/raw/mrclam.pkl and figures.
"""

import argparse
import csv
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from slam.calibration import calibrate
from slam.ekf_slam import run as run_ekf
from slam.metrics import ate, ate_aligned, landmark_rmse, nees
from slam.mrclam import load
from slam.mrclam_graph import solve_incremental

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MRCLAM1"
RESULTS = ROOT / "results"
KERNELS = [(None, 1.0), ("huber", 1.0), ("huber", 2.0), ("cauchy", 1.0), ("cauchy", 2.0), ("cauchy", 3.0)]
EKF_VARIANTS = {
    "EKF": dict(gate=None),
    "EKF + gate": dict(gate=9.21),
    "EKF + gate + bias": dict(gate=9.21, estimate_bias=True),
}


def kernel_name(kernel):
    return "least squares" if kernel[0] is None else f"{kernel[0]} k={kernel[1]:g}"


def graph_job(args):
    robot, cfg, kernel = args
    data = load(DATA, robot)
    tic = time.time()
    poses, lms, times, seen, hist = solve_incremental(data, cfg, method="gn", robust=kernel[0], robust_k=kernel[1])
    return {"robot": robot, "method": f"Graph ({kernel_name(kernel)})", "t": times, "pose": poses,
            "landmarks": lms, "seen": seen, "cov": None, "seconds": time.time() - tic}


def filter_job(args):
    robot, cfg, label, overrides, use = args
    data = load(DATA, robot)
    tic = time.time()
    ekf, rec = run_ekf(data, replace(cfg, **overrides), use_measurements=use)
    return {"robot": robot, "method": label, "t": rec["t"], "pose": rec["pose"],
            "landmarks": ekf.landmarks() if use else None, "seen": ekf.seen if use else None,
            "cov": rec["cov"], "seconds": time.time() - tic, "rejected": ekf.stats["rejected"]}


def score(run):
    data = load(DATA, run["robot"])
    gt = data.gt_at(run["t"])
    raw_pos, raw_head = ate(run["pose"], gt)
    lm = run["landmarks"]
    seen = run["seen"] if lm is not None else None
    al_pos, al_lm, rot = ate_aligned(run["pose"], gt, lm[seen] if lm is not None else None,
                                     data.landmarks[seen] if lm is not None else None)
    row = {"robot": run["robot"], "method": run["method"], "ate_aligned_m": al_pos, "map_aligned_m": al_lm,
           "ate_raw_m": raw_pos, "heading_raw_deg": np.degrees(raw_head),
           "map_raw_m": landmark_rmse(lm, data.landmarks, seen) if lm is not None else float("nan"),
           "frame_rotation_deg": rot, "seconds": run["seconds"]}
    row["nees"] = float(np.mean(nees(run["pose"], run["cov"], gt))) if run["cov"] is not None else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    raw_path = RESULTS / "raw" / "mrclam.pkl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.plots_only:
        tic = time.time()
        cfg, cal_info = calibrate(load(DATA, 1))
        print(f"calibrated on robot 1 in {time.time() - tic:.0f} s: {cfg}\n  {cal_info}")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            selection = list(pool.map(graph_job, [(1, cfg, k) for k in KERNELS]))
        sel_rows = [score(r) for r in selection]
        best = KERNELS[int(np.argmin([r["ate_aligned_m"] for r in sel_rows]))]
        for k, r in zip(KERNELS, sel_rows):
            print(f"  robot 1, {kernel_name(k):14s}: aligned ATE {r['ate_aligned_m']:.3f} m")
        print(f"kernel chosen on robot 1: {kernel_name(best)}")

        jobs_f = [(robot, cfg, "Dead reckoning", {"gate": None}, False) for robot in range(1, 6)]
        jobs_f += [(robot, cfg, label, o, True) for robot in range(1, 6) for label, o in EKF_VARIANTS.items()]
        jobs_g = [(robot, cfg, k) for robot in range(1, 6) for k in ([best] if best == (None, 1.0) else [(None, 1.0), best])]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            runs = list(pool.map(filter_job, jobs_f)) + list(pool.map(graph_job, jobs_g))
        with open(raw_path, "wb") as fh:
            pickle.dump({"cfg": asdict(cfg), "calibration": cal_info, "kernel": best,
                         "selection": sel_rows, "runs": runs}, fh)

    with open(raw_path, "rb") as fh:
        saved = pickle.load(fh)
    rows = [score(r) for r in saved["runs"]]
    with open(RESULTS / "mrclam.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["Aligned trajectory RMSE in m (raw RMSE in brackets). Robot 1 is the calibration robot.", "",
             "| Method | " + " | ".join(f"Robot {k}" for k in range(1, 6)) + " | Mean, robots 2-5 |",
             "|---|" + "---:|" * 6]
    for m in methods:
        cells, test = [], []
        for robot in range(1, 6):
            r = next(x for x in rows if x["robot"] == robot and x["method"] == m)
            cells.append(f"{r['ate_aligned_m']:.2f} ({r['ate_raw_m']:.2f})")
            if robot > 1:
                test.append(r["ate_aligned_m"])
        lines.append(f"| {m} | " + " | ".join(cells) + f" | **{np.mean(test):.2f}** |")
    (RESULTS / "mrclam.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    from slam import plotting
    plotting.mrclam_figures(saved, rows, DATA, RESULTS / "figures")


if __name__ == "__main__":
    main()
