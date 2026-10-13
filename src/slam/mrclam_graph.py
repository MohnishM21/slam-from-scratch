"""Full SLAM on an MRCLAM robot as a factor graph: every pose and landmark solved jointly.

Keyframes are placed at every landmark-measurement time, plus at least one per
`max_gap` seconds. Odometry between consecutive keyframes becomes a Between factor
whose covariance comes from integrating the same calibrated noise model the EKF
uses (motion.integrate_segments). Each landmark sighting becomes a RangeBearing
factor. A tight prior fixes the first pose to the ground-truth start, as for the EKF.

Unlike the EKF, the optimizer can revisit every past pose, so a heading drift or an
outlier early in the run can be corrected by later measurements.
"""

import numpy as np

from .ekf_slam import EKFConfig
from .factor_graph import Between, Graph, Prior, RangeBearing
from .motion import integrate_segments
from .se2 import compose


# Keyframes closer than this are merged. Several sightings often arrive within a
# millisecond; a keyframe for each made odometry factors with near-zero covariance
# (information up to 1e12), which wrecked the conditioning of the normal equations.
MIN_SPACING = 0.05      # s; at MRCLAM speeds (< 0.2 m/s, 0.2 rad/s) snapping costs < 1 cm, 0.01 rad
COV_FLOOR = 1e-6        # added to every odometry covariance (timing jitter)


def keyframe_times(data, max_gap=1.0, min_spacing=MIN_SPACING):
    t_end = data.odometry[-1, 0]
    candidates = np.unique(np.concatenate([[data.t0], data.measurements[:, 0],
                                           np.arange(data.t0, t_end, max_gap), [t_end]]))
    candidates = candidates[(candidates >= data.t0) & (candidates <= t_end)]
    times = [candidates[0]]
    for t in candidates[1:]:
        if t - times[-1] >= min_spacing:
            times.append(t)
    return np.array(times)


def nearest_keyframe(times, t):
    k = np.clip(np.searchsorted(times, t), 1, len(times) - 1)
    return np.where(np.abs(times[k - 1] - t) <= np.abs(times[k] - t), k - 1, k)


def build(data, cfg: EKFConfig, max_gap=1.0, init_poses=None, init_landmarks=None):
    times = keyframe_times(data, max_gap)
    z, cov = integrate_segments(data.odometry, times, cfg.q_v, cfg.q_w)
    info = np.linalg.inv(cov + COV_FLOOR * np.eye(3))
    n = len(times)
    odometry = Between(np.arange(n - 1), np.arange(1, n), z, info)

    x0 = data.gt_at(np.array([times[0]]))[0]
    if init_poses is None:
        poses = np.zeros((n, 3))
        poses[0] = x0
        for k in range(n - 1):
            poses[k + 1] = compose(poses[k], z[k])
    else:
        poses = init_poses.copy()

    meas = data.measurements
    pose_index = nearest_keyframe(times, meas[:, 0])
    lm = meas[:, 1].astype(int)
    rb_info = np.diag([1.0 / cfg.sigma_range ** 2, 1.0 / cfg.sigma_bearing ** 2])
    sightings = RangeBearing(pose_index, lm, meas[:, 2:4].copy(), np.repeat(rb_info[None], len(meas), 0))

    if init_landmarks is None:
        # first sighting from the initial trajectory
        landmarks = np.zeros((len(data.landmarks), 2))
        for j in range(len(landmarks)):
            k = np.argmax(lm == j)
            p = poses[pose_index[k]]
            a = p[2] + meas[k, 3]
            landmarks[j] = p[:2] + meas[k, 2] * np.array([np.cos(a), np.sin(a)])
    else:
        landmarks = init_landmarks.copy()

    graph = Graph(poses=poses, landmarks=landmarks, between=[odometry], range_bearing=[sightings],
                  priors=[Prior(0, x0, 1e8 * np.eye(3))])
    return graph, times


def _subgraph(full: Graph, last_pose, landmark_ids, poses, landmarks):
    """Factors touching poses 0..last_pose and the given landmarks, re-indexed compactly."""
    odo = full.between[0]
    keep_o = odo.j <= last_pose
    rb = full.range_bearing[0]
    keep_r = rb.pose <= last_pose
    remap = -np.ones(len(full.landmarks), dtype=int)
    remap[landmark_ids] = np.arange(len(landmark_ids))
    sub = Graph(
        poses=poses[: last_pose + 1].copy(),
        landmarks=landmarks[landmark_ids].copy(),
        between=[Between(odo.i[keep_o], odo.j[keep_o], odo.z[keep_o], odo.info[keep_o])],
        range_bearing=[RangeBearing(rb.pose[keep_r], remap[rb.landmark[keep_r]], rb.z[keep_r], rb.info[keep_r])],
        priors=full.priors,
    )
    return sub


def solve_incremental(data, cfg: EKFConfig, chunk_s=60.0, max_gap=1.0, robust="huber", robust_k=2.0,
                      method="lm", iters_per_chunk=30):
    """Grow the graph chunk by chunk, warm-starting each solve from the previous one.

    New poses are chained from the last optimized pose by odometry, and a landmark
    enters the graph at its first sighting, placed from the optimized pose that saw
    it. Every solve therefore starts close to its answer, which a single batch
    solve from dead reckoning (5 m off on MRCLAM) does not.
    """
    from .factor_graph import optimize

    full, times = build(data, cfg, max_gap)
    poses, landmarks = full.poses.copy(), full.landmarks.copy()
    odo, rb = full.between[0], full.range_bearing[0]
    seen = np.zeros(len(landmarks), bool)
    history = []
    last = 0
    for t_cut in np.arange(times[0] + chunk_s, times[-1] + chunk_s, chunk_s):
        new_last = int(np.searchsorted(times, t_cut, side="right") - 1)
        for k in range(last, new_last):                      # chain new poses by odometry
            poses[k + 1] = compose(poses[k], odo.z[k])
        in_window = rb.pose <= new_last
        for j in np.unique(rb.landmark[in_window]):
            if not seen[j]:                                  # first sighting, from the optimized pose
                first = np.flatnonzero(in_window & (rb.landmark == j))[0]
                p = poses[rb.pose[first]]
                a = p[2] + rb.z[first, 1]
                landmarks[j] = p[:2] + rb.z[first, 0] * np.array([np.cos(a), np.sin(a)])
                seen[j] = True
        ids = np.flatnonzero(seen)
        sub = _subgraph(full, new_last, ids, poses, landmarks)
        p, l, h = optimize(sub, method=method, max_iter=iters_per_chunk, robust=robust, robust_k=robust_k,
                           robust_on=("range_bearing",))
        poses[: new_last + 1], landmarks[ids] = p, l
        history.append({"t": t_cut - times[0], "poses": new_last + 1, "iterations": len(h) - 1,
                        "cost": h[-1]["cost"], "seconds": h[-1]["seconds"]})
        last = new_last
    return poses, landmarks, times, seen, history
