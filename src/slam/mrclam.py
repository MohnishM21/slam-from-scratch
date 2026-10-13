"""Loader for the UTIAS Multi-Robot Cooperative Localization and Mapping dataset (MRCLAM).

Leung, Halpern, Barfoot and Liu, IJRR 30(8), 2011. Each robot logs:
    odometry      time, forward velocity v (m/s), angular velocity w (rad/s), about 67 Hz
    measurements  time, barcode, range (m), bearing (rad) to landmarks and other robots
    ground truth  time, x, y, theta from a 10-camera Vicon system, 100 Hz
Landmark positions are also surveyed with the Vicon system.

Subjects 1-5 are robots and 6-20 are landmarks. Measurements name the barcode,
which Barcodes.dat maps to a subject. Single-robot SLAM keeps landmark sightings only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .se2 import wrap

N_ROBOTS = 5


@dataclass
class RobotData:
    robot: int
    odometry: np.ndarray        # (n, 3): t, v, w
    measurements: np.ndarray    # (m, 4): t, landmark index (0..14), range, bearing
    groundtruth: np.ndarray     # (k, 4): t, x, y, theta
    landmarks: np.ndarray       # (15, 2) surveyed landmark positions
    t0: float                   # start time (first odometry sample)

    def gt_at(self, t):
        """Ground-truth pose interpolated at times t (angles unwrapped first)."""
        gt = self.groundtruth
        theta = np.unwrap(gt[:, 3])
        return np.column_stack([np.interp(t, gt[:, 0], gt[:, 1]), np.interp(t, gt[:, 0], gt[:, 2]),
                                wrap(np.interp(t, gt[:, 0], theta))])


# Dataset 1 ships with the surveyed positions of landmark subjects 11 and 17 swapped
# (or equivalently their barcodes). With ground-truth robot poses, every sighting of
# either landmark has a median bearing error of about 1.5 rad on all five robots;
# swapping the two positions brings both to about 0.01 rad, like every other landmark.
# scripts/check_landmark_swap.py reproduces this.
KNOWN_SWAPS = {"MRCLAM1": [(11, 17)]}


def _load(path):
    return np.loadtxt(path, comments="#", ndmin=2)


def load(folder, robot, fix_known_swaps=True):
    folder = Path(folder)
    barcodes = _load(folder / "Barcodes.dat").astype(int)
    barcode_to_subject = {b: s for s, b in barcodes}
    lm_gt = _load(folder / "Landmark_Groundtruth.dat")
    lm_subjects = lm_gt[:, 0].astype(int)
    subject_to_index = {s: k for k, s in enumerate(lm_subjects)}
    if fix_known_swaps:
        for a, b in KNOWN_SWAPS.get(folder.name, []):
            ia, ib = subject_to_index[a], subject_to_index[b]
            lm_gt[[ia, ib], 1:3] = lm_gt[[ib, ia], 1:3]

    odo = _load(folder / f"Robot{robot}_Odometry.dat")
    meas = _load(folder / f"Robot{robot}_Measurement.dat")
    gt = _load(folder / f"Robot{robot}_Groundtruth.dat")

    subjects = np.array([barcode_to_subject.get(int(b), -1) for b in meas[:, 1]])
    keep = np.isin(subjects, lm_subjects)
    meas = meas[keep]
    lm_index = np.array([subject_to_index[s] for s in subjects[keep]], dtype=float)
    meas = np.column_stack([meas[:, 0], lm_index, meas[:, 2], meas[:, 3]])

    t0 = odo[0, 0]
    meas = meas[(meas[:, 0] >= t0) & (meas[:, 0] <= odo[-1, 0])]
    return RobotData(robot, odo, meas[np.argsort(meas[:, 0], kind="stable")], gt, lm_gt[:, 1:3], t0)


def measurement_residuals(data: RobotData):
    """Range and bearing errors of every landmark measurement against ground truth."""
    pose = data.gt_at(data.measurements[:, 0])
    lm = data.landmarks[data.measurements[:, 1].astype(int)]
    d = lm - pose[:, :2]
    r_true = np.hypot(d[:, 0], d[:, 1])
    b_true = wrap(np.arctan2(d[:, 1], d[:, 0]) - pose[:, 2])
    return data.measurements[:, 2] - r_true, wrap(data.measurements[:, 3] - b_true), r_true


def velocity_residuals(data: RobotData, window=0.5):
    """Odometry velocity errors against velocities differentiated from ground truth.

    Ground truth is averaged over `window` seconds so the Vicon jitter does not
    dominate. Returns v and w errors, sampled every `window` seconds.
    """
    t = np.arange(data.odometry[0, 0] + window, data.odometry[-1, 0] - window, window)
    a, b = data.gt_at(t - window / 2), data.gt_at(t + window / 2)
    heading = wrap(a[:, 2] + 0.5 * wrap(b[:, 2] - a[:, 2]))
    v_gt = ((b[:, 0] - a[:, 0]) * np.cos(heading) + (b[:, 1] - a[:, 1]) * np.sin(heading)) / window
    w_gt = wrap(b[:, 2] - a[:, 2]) / window
    odo = data.odometry
    # average commanded odometry over the same window
    cum_v = np.concatenate([[0], np.cumsum(odo[:-1, 1] * np.diff(odo[:, 0]))])
    cum_w = np.concatenate([[0], np.cumsum(odo[:-1, 2] * np.diff(odo[:, 0]))])
    v_odo = (np.interp(t + window / 2, odo[:, 0], cum_v) - np.interp(t - window / 2, odo[:, 0], cum_v)) / window
    w_odo = (np.interp(t + window / 2, odo[:, 0], cum_w) - np.interp(t - window / 2, odo[:, 0], cum_w)) / window
    return v_odo - v_gt, w_odo - w_gt, v_odo, w_odo
