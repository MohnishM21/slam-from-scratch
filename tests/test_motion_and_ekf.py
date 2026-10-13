import numpy as np
import pytest

from slam.ekf_slam import EKFConfig, EKFSLAM
from slam.motion import integrate_segments, step
from slam.se2 import compose, wrap


@pytest.mark.parametrize("v, w", [(0.3, 0.4), (0.2, 0.0), (-0.1, -1.2)])
def test_motion_jacobians_match_finite_differences(v, w):
    pose, dt, eps = np.array([1.0, -2.0, 0.7]), 0.2, 1e-7
    _, G, V = step(pose, v, w, dt)
    G_num = np.column_stack([(step(pose + eps * e, v, w, dt)[0] - step(pose - eps * e, v, w, dt)[0]) / (2 * eps)
                             for e in np.eye(3)])
    ew = 1e-4  # the arc formula divides by w, so a tiny step near w = 0 loses precision
    V_num = np.column_stack([(step(pose, v + eps, w, dt)[0] - step(pose, v - eps, w, dt)[0]) / (2 * eps),
                             (step(pose, v, w + ew, dt)[0] - step(pose, v, w - ew, dt)[0]) / (2 * ew)])
    np.testing.assert_allclose(G, G_num, atol=1e-6)
    np.testing.assert_allclose(V, V_num, atol=1e-5)


def test_segments_compose_to_the_whole_trajectory():
    t = np.arange(0, 10, 0.013)
    odo = np.column_stack([t, 0.2 + 0.05 * np.sin(t), 0.3 * np.cos(0.5 * t)])
    z_whole, _ = integrate_segments(odo, np.array([0.0, 9.0]), 1e-4, 1e-3)
    z_parts, _ = integrate_segments(odo, np.linspace(0, 9, 31), 1e-4, 1e-3)
    pose = np.zeros(3)
    for z in z_parts:
        pose = compose(pose, z)
    np.testing.assert_allclose(pose, z_whole[0], atol=1e-9)


def test_heading_variance_grows_linearly_with_time():
    t = np.arange(0, 20, 0.01)
    odo = np.column_stack([t, np.full_like(t, 0.1), np.zeros_like(t)])
    q_w = 1e-3
    for T in (5.0, 10.0):
        _, cov = integrate_segments(odo, np.array([0.0, T]), 1e-5, q_w)
        assert cov[0, 2, 2] == pytest.approx(q_w * T, rel=1e-6)


def simulate(seconds=120, dt=0.02, bias=0.0, seed=0, q_v=1e-4, q_w=1e-3, sr=0.05, sb=0.01):
    """A robot circling among 8 landmarks, with odometry noise drawn from the EKF's own model."""
    rng = np.random.default_rng(seed)
    lms = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]) * 4.0
    pose = np.array([2.0, 0.0, np.pi / 2])
    truth, odo, meas = [], [], []
    for k in range(int(seconds / dt)):
        t = k * dt
        v_true, w_true = 0.3, 0.15
        odo.append([t, v_true + rng.normal(0, np.sqrt(q_v / dt)), w_true + bias + rng.normal(0, np.sqrt(q_w / dt))])
        truth.append([t, *pose])
        if k % 10 == 0:
            for j, l in enumerate(lms):
                d = l - pose[:2]
                r = np.hypot(*d)
                if r < 5:
                    meas.append([t, j, r + rng.normal(0, sr), wrap(np.arctan2(d[1], d[0]) - pose[2] + rng.normal(0, sb))])
        pose = step(pose, v_true, w_true, dt)[0]
    return np.array(truth), np.array(odo), np.array(meas), lms


def run_sim(cfg, odo, meas, x0, n_lm):
    ekf = EKFSLAM(x0, np.diag([1e-6] * 3), n_lm, cfg)
    m = 0
    for k in range(len(odo) - 1):
        while m < len(meas) and meas[m, 0] <= odo[k, 0] + 1e-9:
            ekf.update(int(meas[m, 1]), meas[m, 2], meas[m, 3])
            m += 1
        ekf.predict(odo[k, 1], odo[k, 2], odo[k + 1, 0] - odo[k, 0])
    return ekf


def test_ekf_maps_simulated_world():
    truth, odo, meas, lms = simulate()
    cfg = EKFConfig(1e-4, 1e-3, 0.05, 0.01)
    ekf = run_sim(cfg, odo, meas, truth[0, 1:], len(lms))
    assert ekf.seen.all()
    assert np.abs(ekf.landmarks() - lms).max() < 0.2
    assert np.linalg.norm(ekf.pose[:2] - step(truth[-2, 1:], 0.3, 0.15, 0.02)[0][:2]) < 0.2


def test_ekf_learns_a_turn_rate_bias():
    truth, odo, meas, lms = simulate(bias=0.03, seconds=200, seed=1)
    cfg = EKFConfig(1e-4, 1e-3, 0.05, 0.01, estimate_bias=True)
    ekf = run_sim(cfg, odo, meas, truth[0, 1:], len(lms))
    assert ekf.bias == pytest.approx(0.03, abs=0.005)


def test_gate_rejects_a_gross_outlier():
    cfg = EKFConfig(1e-4, 1e-3, 0.05, 0.01)
    ekf = EKFSLAM(np.zeros(3), np.diag([1e-4] * 3), 1, cfg)
    ekf.update(0, 2.0, 0.0)                      # initializes the landmark at (2, 0)
    assert ekf.update(0, 2.01, 0.005) == "accepted"
    assert ekf.update(0, 5.0, 1.0) == "rejected"
