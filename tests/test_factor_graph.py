"""Solver correctness on synthetic problems with known answers."""

import numpy as np
import pytest

from slam.factor_graph import (Between, Graph, Prior, RangeBearing, _between_terms, _range_bearing_terms,
                               between_error, linearize, optimize, range_bearing_error)
from slam.initialization import chordal, odometry_chain
from slam.se2 import between, compose, wrap

rng = np.random.default_rng(0)


def random_poses(n, spread=5.0):
    return np.column_stack([rng.uniform(-spread, spread, (n, 2)), rng.uniform(-np.pi, np.pi, n)])


def numeric_jacobian(f, x, eps=1e-7):
    x = x.astype(float)
    cols = []
    for k in range(x.size):
        d = np.zeros_like(x)
        d.flat[k] = eps
        diff = f(x + d) - f(x - d)
        diff[..., -1] = wrap(diff[..., -1]) if diff.shape[-1] == 3 else diff[..., -1]
        cols.append(diff / (2 * eps))
    return np.stack(cols, -1)


def test_between_jacobians_match_finite_differences():
    poses = random_poses(2)
    z = random_poses(1, 2.0)
    f = Between(np.array([0]), np.array([1]), z, np.eye(3)[None])
    _, [(_, _, A), (_, _, B)] = _between_terms(poses, f)

    def err_i(xi):
        p = poses.copy(); p[0] = xi
        return between_error(p, f)[0]

    def err_j(xj):
        p = poses.copy(); p[1] = xj
        return between_error(p, f)[0]

    np.testing.assert_allclose(A[0], numeric_jacobian(err_i, poses[0]), atol=1e-6)
    np.testing.assert_allclose(B[0], numeric_jacobian(err_j, poses[1]), atol=1e-6)


def test_range_bearing_jacobians_match_finite_differences():
    poses, lms = random_poses(1), rng.uniform(-5, 5, (1, 2))
    f = RangeBearing(np.array([0]), np.array([0]), np.array([[3.0, 0.2]]), np.eye(2)[None])
    _, [(_, _, Jp), (_, _, Jl)] = _range_bearing_terms(poses, lms, f)

    def err_p(x):
        return range_bearing_error(x[None], lms, f)[0]

    def err_l(l):
        return range_bearing_error(poses, l[None], f)[0]

    np.testing.assert_allclose(Jp[0], numeric_jacobian(err_p, poses[0]), atol=1e-6)
    np.testing.assert_allclose(Jl[0], numeric_jacobian(err_l, lms[0]), atol=1e-6)


def synthetic_pose_graph(n=60, loops=30, noise=(0.02, 0.01), seed=1):
    """A wandering trajectory with noisy odometry and noisy loop closures."""
    rng = np.random.default_rng(seed)
    truth = np.zeros((n, 3))
    for k in range(n - 1):
        truth[k + 1] = compose(truth[k], [1.0, 0.0, rng.uniform(-0.6, 0.6)])
    pairs = [(k, k + 1) for k in range(n - 1)]
    pairs += [tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(loops)]
    i, j = np.array(pairs).T
    z = between(truth[i], truth[j])
    z[:, :2] += rng.normal(0, noise[0], (len(i), 2))
    z[:, 2] = wrap(z[:, 2] + rng.normal(0, noise[1], len(i)))
    info = np.diag([1 / noise[0] ** 2] * 2 + [1 / noise[1] ** 2])
    edges = Between(i, j, z, np.repeat(info[None], len(i), 0))
    return truth, edges


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pose_graph_gn_and_lm_reach_the_same_optimum(seed):
    truth, edges = synthetic_pose_graph(seed=seed)
    init = odometry_chain(len(truth), edges)
    g = Graph(poses=init, between=[edges], priors=[Prior(0, truth[0], 1e8 * np.eye(3))])
    p_gn, _, h_gn = optimize(g, method="gn", max_iter=100)
    p_lm, _, h_lm = optimize(g, method="lm", max_iter=100)
    assert h_lm[-1]["chi2"] == pytest.approx(h_gn[-1]["chi2"], rel=1e-6)
    np.testing.assert_allclose(p_lm, p_gn, atol=1e-5)
    # With 60 poses and 30 loop closures, chi^2 at the optimum is about its degrees of freedom.
    dof = 3 * len(edges.i) - 3 * (len(truth) - 1)
    assert h_gn[-1]["chi2"] < 2 * dof
    # And it removes most of the odometry drift.
    drift = np.abs(init[:, :2] - truth[:, :2]).max()
    assert np.abs(p_gn[:, :2] - truth[:, :2]).max() < 0.5 * drift


@pytest.mark.parametrize("seed", [4, 6, 7])
def test_chordal_initialization_starts_near_the_optimum(seed):
    truth, edges = synthetic_pose_graph(seed=seed)
    prior = [Prior(0, truth[0], 1e8 * np.eye(3))]
    odo = odometry_chain(len(truth), edges)
    opt, _, _ = optimize(Graph(poses=odo, between=[edges], priors=prior), method="gn", max_iter=100)
    ch = chordal(len(truth), [edges])

    def chi2(p):
        return linearize(Graph(poses=p, between=[edges], priors=prior), p, np.zeros((0, 2)))[3]

    assert chi2(ch) < chi2(odo) / 10
    assert np.abs(wrap(ch[:, 2] - opt[:, 2])).max() < 0.05


def test_landmark_slam_recovers_map():
    truth = np.zeros((40, 3))
    for k in range(39):
        truth[k + 1] = compose(truth[k], [0.5, 0.0, 0.15])
    lms = rng.uniform(-4, 6, (6, 2))
    odo_z = between(truth[:-1], truth[1:]) + rng.normal(0, [0.01, 0.01, 0.005], (39, 3))
    odo = Between(np.arange(39), np.arange(1, 40), odo_z, np.repeat(np.diag([1e4, 1e4, 4e4])[None], 39, 0))
    pi, li = np.meshgrid(np.arange(40), np.arange(6), indexing="ij")
    pi, li = pi.ravel(), li.ravel()
    d = lms[li] - truth[pi, :2]
    z = np.column_stack([np.hypot(d[:, 0], d[:, 1]), wrap(np.arctan2(d[:, 1], d[:, 0]) - truth[pi, 2])])
    z += rng.normal(0, [0.02, 0.005], z.shape)
    rb = RangeBearing(pi, li, z, np.repeat(np.diag([2500, 40000])[None], len(pi), 0))
    g = Graph(poses=odometry_chain(40, odo), landmarks=lms + rng.normal(0, 0.3, lms.shape),
              between=[odo], range_bearing=[rb], priors=[Prior(0, truth[0], 1e8 * np.eye(3))])
    p, l, _ = optimize(g, method="lm", max_iter=50)
    assert np.abs(l - lms).max() < 0.05
    assert np.abs(p[:, :2] - truth[:, :2]).max() < 0.05


def test_cauchy_kernel_ignores_false_loop_closures():
    truth, edges = synthetic_pose_graph(n=80, loops=40, seed=5)
    bad_i = rng.choice(80, 10)
    bad_j = (bad_i + rng.integers(10, 60, 10)) % 80
    bad = Between(bad_i, bad_j, random_poses(10, 3.0), np.repeat(edges.info[:1], 10, 0))
    init = odometry_chain(80, edges)
    prior = [Prior(0, truth[0], 1e8 * np.eye(3))]
    plain, _, _ = optimize(Graph(poses=init, between=[edges, bad], priors=prior), method="gn", max_iter=50)
    robust, _, _ = optimize(Graph(poses=init, between=[edges, bad], priors=prior), method="gn", max_iter=50,
                            robust="cauchy", robust_k=1.0)
    err = lambda p: np.sqrt(np.mean(((p[:, :2] - truth[:, :2]) ** 2).sum(-1)))
    assert err(robust) < 0.2
    assert err(plain) > 5 * err(robust)
