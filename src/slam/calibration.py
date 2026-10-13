"""Noise models estimated from a calibration robot's data and ground truth.

Measurement noise: robust standard deviations of range and bearing errors.

Odometry noise: the velocity errors over short windows give a white-noise
spectral density, but real odometry errors are correlated (wheel slip, scale
error), so that model badly underestimates long-run drift: dead reckoning had
a mean NEES of 93 instead of 3. The densities are therefore scaled until dead
reckoning on the calibration robot is consistent (mean NEES = 3).
"""

import numpy as np

from .ekf_slam import EKFConfig, run
from .metrics import nees, robust_std
from .mrclam import measurement_residuals, velocity_residuals


def dead_reckoning_nees(data, q_v, q_w):
    _, rec = run(data, EKFConfig(q_v, q_w, 1.0, 1.0, None), use_measurements=False)
    return float(np.mean(nees(rec["pose"], rec["cov"], data.gt_at(rec["t"]))))


def calibrate(data, window=0.5, target_nees=3.0, gate=9.21):
    er, eb, _ = measurement_residuals(data)
    ev, ew, _, _ = velocity_residuals(data, window)
    q_v, q_w = robust_std(ev) ** 2 * window, robust_std(ew) ** 2 * window
    # NEES is close to inversely proportional to the noise scale: bisect in log space.
    lo, hi = 0.0, np.log(1e4)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if dead_reckoning_nees(data, q_v * np.exp(mid), q_w * np.exp(mid)) > target_nees:
            lo = mid
        else:
            hi = mid
    scale = float(np.exp(0.5 * (lo + hi)))
    cfg = EKFConfig(q_v * scale, q_w * scale, robust_std(er), robust_std(eb), gate)
    return cfg, {"white_noise_q_v": q_v, "white_noise_q_w": q_w, "consistency_scale": scale}
