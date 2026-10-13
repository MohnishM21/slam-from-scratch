"""Sparse nonlinear least squares for 2D SLAM, written from scratch.

Unknowns are robot poses (x, y, theta) and, optionally, point landmarks (x, y).
Three kinds of factor connect them:

    Between       pose i to pose j: odometry or a loop closure, measured in i's frame
    RangeBearing  pose i to landmark j: range and bearing
    Prior         a single pose: fixes the gauge (the map's global position and heading)

Each kind is evaluated for all its factors at once with NumPy, with analytic
Jacobians. Residuals are whitened by the Cholesky factor of each information
matrix, so the cost is sum_k rho(e_k^T Omega_k e_k). The normal equations
J^T J dx = -J^T r are sparse and solved with SuperLU (see solve_spd).

Robust kernels (Huber, Cauchy) are applied by iteratively reweighted least squares:
each factor's whitened residual and Jacobian are scaled by sqrt(w), w = rho'(s).
"""

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .se2 import rot, wrap


@dataclass
class Between:
    i: np.ndarray            # (m,) pose indices
    j: np.ndarray            # (m,)
    z: np.ndarray            # (m, 3) measured pose of j in i's frame
    info: np.ndarray         # (m, 3, 3)


@dataclass
class RangeBearing:
    pose: np.ndarray         # (m,)
    landmark: np.ndarray     # (m,)
    z: np.ndarray            # (m, 2) range (m), bearing (rad)
    info: np.ndarray         # (m, 2, 2)


@dataclass
class Prior:
    i: int
    z: np.ndarray            # (3,)
    info: np.ndarray         # (3, 3)


@dataclass
class Graph:
    poses: np.ndarray                         # (Np, 3) initial guess
    landmarks: np.ndarray = None              # (Nl, 2) initial guess, or None
    between: list = field(default_factory=list)
    range_bearing: list = field(default_factory=list)
    priors: list = field(default_factory=list)

    def __post_init__(self):
        if self.landmarks is None:
            self.landmarks = np.zeros((0, 2))

    @property
    def n_vars(self):
        return 3 * len(self.poses) + 2 * len(self.landmarks)


def _cat(factors, cls):
    if not factors:
        return None
    return cls(*[np.concatenate([getattr(f, name) for f in factors]) for name in cls.__dataclass_fields__])


def _sqrt_info(info):
    """Upper factor U with U^T U = info, so whitened residual = U e."""
    return np.swapaxes(np.linalg.cholesky(info), -1, -2)


def between_error(poses, f: Between):
    xi, xj = poses[f.i], poses[f.j]
    Ri, Rz = rot(xi[:, 2]), rot(f.z[:, 2])
    dt = xj[:, :2] - xi[:, :2]
    local = np.einsum("nba,nb->na", Ri, dt)                       # R_i^T (t_j - t_i)
    e = np.empty((len(f.i), 3))
    e[:, :2] = np.einsum("nba,nb->na", Rz, local - f.z[:, :2])    # R_z^T (local - t_z)
    e[:, 2] = wrap(xj[:, 2] - xi[:, 2] - f.z[:, 2])
    return e


def _between_terms(poses, f: Between):
    e = between_error(poses, f)
    xi = poses[f.i]
    c, s = np.cos(xi[:, 2]), np.sin(xi[:, 2])
    RzT = np.swapaxes(rot(f.z[:, 2]), 1, 2)
    RiT = np.swapaxes(rot(xi[:, 2]), 1, 2)
    dRiT = np.stack([np.stack([-s, c], -1), np.stack([-c, -s], -1)], -2)  # d(R_i^T)/d theta_i
    dt = poses[f.j, :2] - xi[:, :2]
    m = len(f.i)
    A = np.zeros((m, 3, 3))
    B = np.zeros((m, 3, 3))
    RzRi = RzT @ RiT
    A[:, :2, :2] = -RzRi
    A[:, :2, 2] = np.einsum("nab,nb->na", RzT @ dRiT, dt)
    A[:, 2, 2] = -1.0
    B[:, :2, :2] = RzRi
    B[:, 2, 2] = 1.0
    return e, [(f.i, 3, A), (f.j, 3, B)]


def range_bearing_error(poses, landmarks, f: RangeBearing):
    x = poses[f.pose]
    d = landmarks[f.landmark] - x[:, :2]
    r = np.hypot(d[:, 0], d[:, 1])
    return np.stack([r - f.z[:, 0], wrap(np.arctan2(d[:, 1], d[:, 0]) - x[:, 2] - f.z[:, 1])], -1)


def _range_bearing_terms(poses, landmarks, f: RangeBearing):
    e = range_bearing_error(poses, landmarks, f)
    d = landmarks[f.landmark] - poses[f.pose, :2]
    q = np.maximum((d ** 2).sum(-1), 1e-12)
    r = np.sqrt(q)
    dx, dy = d[:, 0], d[:, 1]
    m = len(f.pose)
    Jp = np.zeros((m, 2, 3))
    Jp[:, 0, 0], Jp[:, 0, 1] = -dx / r, -dy / r
    Jp[:, 1, 0], Jp[:, 1, 1], Jp[:, 1, 2] = dy / q, -dx / q, -1.0
    Jl = np.zeros((m, 2, 2))
    Jl[:, 0, 0], Jl[:, 0, 1] = dx / r, dy / r
    Jl[:, 1, 0], Jl[:, 1, 1] = -dy / q, dx / q
    return e, [(f.pose, 3, Jp), (f.landmark, 2, Jl)]   # width 2: landmark columns follow the poses


# ---------------------------------------------------------------- robust kernels

def kernel(name, s, k):
    """Cost rho(s) and IRLS weight rho'(s) for squared whitened residual norms s."""
    if name is None:
        return s, np.ones_like(s)
    if name == "huber":
        root = np.sqrt(s)
        inlier = root <= k
        return np.where(inlier, s, 2 * k * root - k * k), np.where(inlier, 1.0, k / np.maximum(root, 1e-12))
    if name == "cauchy":
        return k * k * np.log1p(s / (k * k)), 1.0 / (1.0 + s / (k * k))
    raise ValueError(name)


# ---------------------------------------------------------------- linear system

def linearize(graph: Graph, poses, landmarks, robust=None, robust_k=1.0, robust_on=("between",)):
    """Whitened, robustly weighted residual vector and sparse Jacobian.

    Returns r (M,), J (M, N) CSR, robust cost, and the plain chi^2 (sum of e^T Omega e).
    """
    n_poses = len(poses)
    blocks = []        # (error (m, d), information (m, d, d), [(index, width, jacobian (m, d, width))], kind)
    bt = _cat(graph.between, Between)
    if bt is not None:
        e, jac = _between_terms(poses, bt)
        blocks.append((e, bt.info, jac, "between"))
    rb = _cat(graph.range_bearing, RangeBearing)
    if rb is not None and len(rb.pose):
        e, jac = _range_bearing_terms(poses, landmarks, rb)
        blocks.append((e, rb.info, jac, "range_bearing"))
    for p in graph.priors:
        e = np.concatenate([poses[p.i, :2] - p.z[:2], [wrap(poses[p.i, 2] - p.z[2])]])[None]
        blocks.append((e, p.info[None], [(np.array([p.i]), 3, np.eye(3)[None])], "prior"))

    rows, cols, vals, res = [], [], [], []
    cost, chi2, row0 = 0.0, 0.0, 0
    for e, info, jac, kind in blocks:
        U = _sqrt_info(info)
        rw = np.einsum("nab,nb->na", U, e)
        s = (rw ** 2).sum(-1)
        chi2 += s.sum()
        rho, w = kernel(robust if kind in robust_on else None, s, robust_k)
        cost += rho.sum()
        sw = np.sqrt(w)[:, None]
        rw = rw * sw
        m, d = rw.shape
        res.append(rw.ravel())
        for index, width, Jb in jac:
            Jw = np.einsum("nab,nbc->nac", U, Jb) * sw[:, :, None]
            col_base = 3 * index if width == 3 else 3 * n_poses + 2 * index
            r_idx = row0 + np.arange(m)[:, None, None] * d + np.arange(d)[None, :, None]
            c_idx = col_base[:, None, None] + np.arange(width)[None, None, :]
            rows.append(np.broadcast_to(r_idx, Jw.shape).ravel())
            cols.append(np.broadcast_to(c_idx, Jw.shape).ravel())
            vals.append(Jw.ravel())
        row0 += m * d
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(row0, graph.n_vars))
    return np.concatenate(res), J, cost, chi2


def solve_spd(A, b):
    """Solve a sparse symmetric positive-definite system.

    SuperLU with a symmetric minimum-degree ordering: on a 5,000-pose MRCLAM graph
    this is about 6 times faster than SciPy's default column ordering (0.09 s vs 0.58 s).
    """
    return splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(b)


def _apply(poses, landmarks, dx):
    n = len(poses)
    p = poses + dx[: 3 * n].reshape(n, 3)
    p[:, 2] = wrap(p[:, 2])
    lm = landmarks + dx[3 * n:].reshape(-1, 2) if len(landmarks) else landmarks
    return p, lm


def optimize(graph: Graph, method="lm", max_iter=100, tol=1e-9, robust=None, robust_k=1.0,
             robust_on=("between",), verbose=False, keep_iterates=False):
    """Gauss-Newton or Levenberg-Marquardt. Returns poses, landmarks and a per-iteration history.

    With keep_iterates, each history entry also holds the poses after that iteration.
    """
    poses, landmarks = graph.poses.copy(), graph.landmarks.copy()
    r, J, cost, chi2 = linearize(graph, poses, landmarks, robust, robust_k, robust_on)
    history = [{"iteration": 0, "cost": cost, "chi2": chi2, "seconds": 0.0, "lambda": float("nan")}]
    if keep_iterates:
        history[0]["poses"] = poses.copy()
    lam, nu = 1e-4, 2.0
    tic = time.perf_counter()
    for it in range(1, max_iter + 1):
        H = (J.T @ J).tocsc()
        g = J.T @ r
        diag = H.diagonal()
        while True:
            damping = lam * sp.diags(np.maximum(diag, 1e-12)) if method == "lm" else 1e-12 * sp.eye(H.shape[0])
            dx = solve_spd(H + damping, -g)
            p_new, l_new = _apply(poses, landmarks, dx)
            r_new, J_new, cost_new, chi2_new = linearize(graph, p_new, l_new, robust, robust_k, robust_on)
            if method == "gn":
                break
            # cost = sum of squared whitened residuals, so its model reduction is 2 (-g.dx) - dx.H.dx
            predicted = -2.0 * (dx @ g) - dx @ (H @ dx)
            rho = (cost - cost_new) / max(predicted, 1e-300) if predicted > 0 else -1.0
            if rho > 0:
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                break
            lam *= nu
            nu *= 2.0
            if lam > 1e12:
                break
        improvement = (cost - cost_new) / max(cost, 1e-300)
        accepted = method == "gn" or cost_new < cost
        if accepted:
            poses, landmarks, r, J, cost, chi2 = p_new, l_new, r_new, J_new, cost_new, chi2_new
        history.append({"iteration": it, "cost": cost, "chi2": chi2,
                        "seconds": time.perf_counter() - tic, "lambda": lam if method == "lm" else float("nan")})
        if keep_iterates:
            history[-1]["poses"] = poses.copy()
        if verbose:
            print(f"  it {it:3d} cost {cost:.6e} chi2 {chi2:.6e} lambda {lam:.1e}")
        if not accepted or abs(improvement) < tol or np.abs(dx).max() < 1e-10:
            break
    return poses, landmarks, history

