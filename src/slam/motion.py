"""Unicycle motion model driven by measured odometry velocities, with noise propagation.

Velocity errors are modelled as continuous white noise with spectral densities
q_v (m^2/s) and q_w (rad^2/s). Over a step of length dt the averaged velocity
error then has covariance diag(q_v, q_w) / dt, so the pose uncertainty grows
linearly with time however finely the odometry is sampled.
"""

import numpy as np

from .se2 import wrap


def step(pose, v, w, dt):
    """Exact unicycle integration over dt with constant v and w. Returns new pose,
    the Jacobian G with respect to the pose and V with respect to (v, w)."""
    x, y, th = pose
    if abs(w) > 1e-9:
        th2 = th + w * dt
        s1, c1, s2, c2 = np.sin(th), np.cos(th), np.sin(th2), np.cos(th2)
        r = v / w
        new = np.array([x + r * (s2 - s1), y - r * (c2 - c1), wrap(th2)])
        G = np.array([[1, 0, r * (c2 - c1)], [0, 1, r * (s2 - s1)], [0, 0, 1]])
        V = np.array([[(s2 - s1) / w, r * (c2 * dt - (s2 - s1) / w)],
                      [-(c2 - c1) / w, r * (s2 * dt + (c2 - c1) / w)],
                      [0, dt]])
    else:
        c, s = np.cos(th), np.sin(th)
        new = np.array([x + v * dt * c, y + v * dt * s, th])
        G = np.array([[1, 0, -v * dt * s], [0, 1, v * dt * c], [0, 0, 1]])
        V = np.array([[dt * c, -0.5 * v * dt * dt * s], [dt * s, 0.5 * v * dt * dt * c], [0, dt]])
    return new, G, V


def noise(V, dt, q_v, q_w):
    return V @ np.diag([q_v / dt, q_w / dt]) @ V.T


def integrate_segments(odometry, times, q_v, q_w):
    """Relative motion and its covariance between consecutive `times`.

    odometry: (n, 3) rows of t, v, w; each command holds until the next sample.
    Returns z (k-1, 3) relative poses in the frame of the earlier time and cov (k-1, 3, 3).
    """
    t_odo, v_odo, w_odo = odometry[:, 0], odometry[:, 1], odometry[:, 2]
    k = len(times)
    z = np.zeros((k - 1, 3))
    cov = np.zeros((k - 1, 3, 3))
    idx = max(np.searchsorted(t_odo, times[0], side="right") - 1, 0)
    for seg in range(k - 1):
        t, t_end = times[seg], times[seg + 1]
        pose, P = np.zeros(3), np.zeros((3, 3))
        while t < t_end - 1e-12:
            nxt = t_odo[idx + 1] if idx + 1 < len(t_odo) else np.inf
            dt = min(nxt, t_end) - t
            if dt > 0:
                pose, G, V = step(pose, v_odo[idx], w_odo[idx], dt)
                P = G @ P @ G.T + noise(V, dt, q_v, q_w)
            t += dt
            if t >= nxt - 1e-12:
                idx += 1
        z[seg], cov[seg] = pose, P
    return z, cov
