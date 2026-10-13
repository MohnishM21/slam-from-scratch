"""EKF-SLAM with range-bearing landmarks and known data association.

State: robot pose (x, y, theta) followed by (x, y) of every landmark. Landmarks
enter the state the first time they are seen, initialized from that measurement
with a covariance (and cross-covariances) propagated through the inverse
measurement model, rather than an arbitrary large value.

A landmark's first sighting is often an outlier in real data (on MRCLAM robot 5,
four of the first fifteen are off by more than 0.5 m in range), and gating cannot
protect it because there is nothing to compare it with yet. With delayed
initialization, sightings of an unmapped landmark are buffered as world-frame
candidates, and the landmark is created from the newest sighting only once it
lies within `init_consensus` of the median of the last `init_sightings` candidates.

Updates use the Joseph form, which keeps the covariance symmetric and positive
definite. An optional chi-square gate on the innovation rejects outliers, which
are common in real data (about 10% of MRCLAM landmark measurements are more
than five robust standard deviations off).
"""

from dataclasses import dataclass

import numpy as np

from .motion import noise, step
from .se2 import wrap

CHI2_2DOF_99 = 9.21


@dataclass
class EKFConfig:
    q_v: float                  # odometry velocity noise spectral density, m^2/s
    q_w: float                  # rad^2/s
    sigma_range: float          # m
    sigma_bearing: float        # rad
    gate: float = CHI2_2DOF_99  # None disables gating
    init_sightings: int = 1     # 1 = initialize from the first sighting
    init_consensus: float = 0.3  # m
    estimate_bias: bool = False  # add a turn-rate bias to the state
    bias_sigma0: float = 0.02   # rad/s, initial uncertainty of that bias
    q_bias: float = 1e-6        # (rad/s)^2 per second, how fast the bias may drift


class EKFSLAM:
    """State: x, y, theta, [turn-rate bias b], then (x, y) per landmark.

    With estimate_bias the motion model integrates (v, w - b) and b follows a
    slow random walk. Landmark sightings then correct the heading drift that a
    biased gyro or wheel encoder causes, instead of the filter treating it as noise.
    """

    def __init__(self, x0, P0, n_landmarks, cfg: EKFConfig):
        self.cfg = cfg
        self.nr = 4 if cfg.estimate_bias else 3          # robot block size
        n = self.nr + 2 * n_landmarks
        self.mu = np.zeros(n)
        self.mu[:3] = x0
        self.P = np.zeros((n, n))
        self.P[:3, :3] = P0
        if cfg.estimate_bias:
            self.P[3, 3] = cfg.bias_sigma0 ** 2
        self.seen = np.zeros(n_landmarks, dtype=bool)
        self.pending = {}
        self.R = np.diag([cfg.sigma_range ** 2, cfg.sigma_bearing ** 2])
        self.stats = {"updates": 0, "rejected": 0, "initialized": 0}

    @property
    def pose(self):
        return self.mu[:3]

    @property
    def bias(self):
        return self.mu[3] if self.cfg.estimate_bias else 0.0

    def landmarks(self):
        return self.mu[self.nr:].reshape(-1, 2)

    def predict(self, v, w, dt):
        if dt <= 0:
            return
        nr = self.nr
        new, G, V = step(self.mu[:3], v, w - self.bias, dt)
        F = np.eye(nr)
        F[:3, :3] = G
        if self.cfg.estimate_bias:
            F[:3, 3] = -V[:, 1]                           # d pose / d bias
        self.mu[:3] = new
        P = self.P
        P[:nr, :] = F @ P[:nr, :]
        P[:, :nr] = P[:, :nr] @ F.T
        P[:3, :3] += noise(V, dt, self.cfg.q_v, self.cfg.q_w)
        if self.cfg.estimate_bias:
            P[3, 3] += self.cfg.q_bias * dt

    def _landmark_slice(self, j):
        lj = self.nr + 2 * j
        return slice(lj, lj + 2)

    def _initialize(self, j, r, b):
        x, y, th = self.mu[:3]
        a = th + b
        lj = self._landmark_slice(j)
        self.mu[lj] = [x + r * np.cos(a), y + r * np.sin(a)]
        Gr = np.zeros((2, self.nr))
        Gr[:, :3] = [[1, 0, -r * np.sin(a)], [0, 1, r * np.cos(a)]]
        Gz = np.array([[np.cos(a), -r * np.sin(a)], [np.sin(a), r * np.cos(a)]])
        P = self.P
        P[lj, :] = Gr @ P[:self.nr, :]
        P[:, lj] = P[lj, :].T
        P[lj, lj] = Gr @ P[:self.nr, :self.nr] @ Gr.T + Gz @ self.R @ Gz.T
        self.seen[j] = True
        self.stats["initialized"] += 1

    def _ready_to_initialize(self, j, r, b):
        """Delayed initialization: wait for agreeing sightings (see module docstring)."""
        if self.cfg.init_sightings <= 1:
            return True
        x, y, th = self.mu[:3]
        candidate = np.array([x + r * np.cos(th + b), y + r * np.sin(th + b)])
        buf = self.pending.setdefault(j, [])
        buf.append(candidate)
        del buf[:-self.cfg.init_sightings]
        agree = np.linalg.norm(candidate - np.median(buf, axis=0)) <= self.cfg.init_consensus
        return len(buf) >= self.cfg.init_sightings and agree

    def update(self, j, r, b):
        """Returns 'init', 'pending', 'accepted' or 'rejected'."""
        if not self.seen[j]:
            if self._ready_to_initialize(j, r, b):
                self._initialize(j, r, b)
                self.pending.pop(j, None)
                return "init"
            return "pending"
        lj = self._landmark_slice(j)
        d = self.mu[lj] - self.mu[:2]
        q = d @ d
        rr = np.sqrt(q)
        z_hat = np.array([rr, wrap(np.arctan2(d[1], d[0]) - self.mu[2])])
        n = len(self.mu)
        H = np.zeros((2, n))
        H[:, :3] = [[-d[0] / rr, -d[1] / rr, 0.0], [d[1] / q, -d[0] / q, -1.0]]
        H[:, lj] = [[d[0] / rr, d[1] / rr], [-d[1] / q, d[0] / q]]
        PHt = self.P @ H.T
        S = H @ PHt + self.R
        nu = np.array([r - z_hat[0], wrap(b - z_hat[1])])
        if self.cfg.gate is not None and nu @ np.linalg.solve(S, nu) > self.cfg.gate:
            self.stats["rejected"] += 1
            return "rejected"
        K = np.linalg.solve(S, PHt.T).T
        self.mu += K @ nu
        self.mu[2] = wrap(self.mu[2])
        IKH = np.eye(n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.stats["updates"] += 1
        return "accepted"


def run(data, cfg: EKFConfig, use_measurements=True, record_every=0.1, P0=None):
    """Run EKF-SLAM (or dead reckoning if use_measurements is False) over one robot's log.

    The filter starts at the ground-truth pose at the first odometry time, the
    usual convention for MRCLAM. Returns the filter and a record of (t, pose,
    pose covariance) every `record_every` seconds.
    """
    odo, meas = data.odometry, data.measurements
    x0 = data.gt_at(np.array([data.t0]))[0]
    ekf = EKFSLAM(x0, np.diag([1e-6, 1e-6, 1e-6]) if P0 is None else P0, len(data.landmarks), cfg)
    times, poses, covs = [], [], []
    next_record = data.t0
    m = 0
    t = odo[0, 0]
    for k in range(len(odo)):
        t_end = odo[k + 1, 0] if k + 1 < len(odo) else odo[k, 0]
        v, w = odo[k, 1], odo[k, 2]
        while use_measurements and m < len(meas) and meas[m, 0] <= t_end:
            ekf.predict(v, w, meas[m, 0] - t)
            t = meas[m, 0]
            ekf.update(int(meas[m, 1]), meas[m, 2], meas[m, 3])
            m += 1
        while next_record <= t_end:
            ekf.predict(v, w, next_record - t)
            t = next_record
            times.append(t)
            poses.append(ekf.pose.copy())
            covs.append(ekf.P[:3, :3].copy())
            next_record += record_every
        ekf.predict(v, w, t_end - t)
        t = t_end
    return ekf, {"t": np.array(times), "pose": np.array(poses), "cov": np.array(covs)}
