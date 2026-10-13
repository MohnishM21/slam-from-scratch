"""Accuracy and consistency metrics against ground truth."""

import numpy as np
from scipy.stats import chi2

from .se2 import wrap


def robust_std(x):
    """Standard deviation estimated from the median absolute deviation (insensitive to outliers)."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def ate(est_poses, gt_poses):
    """Absolute trajectory error: RMSE of position (m) and of heading (rad), no alignment.

    All estimators here start from the same ground-truth pose, so their frames
    already agree and no alignment step is needed.
    """
    dp = est_poses[:, :2] - gt_poses[:, :2]
    pos = np.sqrt(np.mean((dp ** 2).sum(-1)))
    head = np.sqrt(np.mean(wrap(est_poses[:, 2] - gt_poses[:, 2]) ** 2))
    return pos, head


def align_se2(est_xy, gt_xy):
    """Rotation R and translation t minimizing |R est + t - gt| (Umeyama, no scale)."""
    mu_e, mu_g = est_xy.mean(0), gt_xy.mean(0)
    U, _, Vt = np.linalg.svd((gt_xy - mu_g).T @ (est_xy - mu_e))
    R = U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, mu_g - R @ mu_e


def ate_aligned(est_poses, gt_poses, est_landmarks=None, gt_landmarks=None):
    """Position RMSE after the best rigid alignment of the trajectory, as in the TUM benchmark.

    SLAM cannot observe the global frame beyond the first pose: a small heading error
    in the first seconds rotates the whole map. Aligning first separates "is the
    map the right shape" from "is it pointing the right way". The same transform is
    applied to the landmarks. Returns (trajectory RMSE, landmark RMSE, rotation in degrees).
    """
    R, t = align_se2(est_poses[:, :2], gt_poses[:, :2])
    traj = (R @ est_poses[:, :2].T).T + t
    pos = np.sqrt(np.mean(((traj - gt_poses[:, :2]) ** 2).sum(-1)))
    lm = float("nan")
    if est_landmarks is not None:
        lm = np.sqrt(np.mean(((((R @ est_landmarks.T).T + t) - gt_landmarks) ** 2).sum(-1)))
    return pos, lm, float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def nees(est_poses, covs, gt_poses):
    """Normalized estimation error squared of the 3-DoF pose, per sample.

    A consistent filter has mean NEES equal to 3. Larger means the filter is
    overconfident: its covariance is smaller than its real error.
    """
    e = est_poses - gt_poses
    e[:, 2] = wrap(e[:, 2])
    return np.einsum("ni,ni->n", e, np.linalg.solve(covs, e[..., None])[..., 0])


def nees_bounds(n_samples, dof=3, confidence=0.95):
    """Two-sided bounds for the average NEES of n independent samples."""
    lo, hi = chi2.ppf([(1 - confidence) / 2, (1 + confidence) / 2], dof * n_samples)
    return lo / n_samples, hi / n_samples


def landmark_rmse(est, gt, mask=None):
    mask = np.ones(len(gt), bool) if mask is None else mask
    return np.sqrt(np.mean(((est[mask] - gt[mask]) ** 2).sum(-1)))
