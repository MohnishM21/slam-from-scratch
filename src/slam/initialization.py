"""Initial guesses for pose-graph optimization.

Odometry chaining accumulates drift, and Gauss-Newton started from it can stop in
a local minimum (it does on the MIT Killian Court graph). Chordal initialization
(Martinec and Pajdla 2007; Carlone et al. 2015) avoids the nonconvexity in two
linear steps:

1. Rotations. Write each orientation as an unconstrained 2-vector (c, s). Every
   edge says R_j = R_i R_ij, which is linear in (c_i, s_i, c_j, s_j). Solve the
   weighted linear least-squares problem, then project each 2-vector back onto
   the unit circle.
2. Positions. With rotations fixed, every edge says t_j - t_i = R_i t_ij, which
   is linear in the positions. Solve again.

Both steps fix pose 0 to remove the gauge freedom.
"""

import numpy as np
import scipy.sparse as sp

from .factor_graph import Between, _cat, solve_spd
from .se2 import compose, rot


def odometry_chain(n_poses, between: Between):
    """Compose the consecutive-pose edges, starting from the origin."""
    poses = np.zeros((n_poses, 3))
    odo = {int(i): k for k, (i, j) in enumerate(zip(between.i, between.j)) if j == i + 1}
    for i in range(n_poses - 1):
        poses[i + 1] = compose(poses[i], between.z[odo[i]])
    return poses


def _solve_anchored(A, b, block, anchor_values):
    """Least squares with the first block of unknowns fixed to anchor_values."""
    A = A.tocsc()
    free = A[:, block:]
    rhs = b - A[:, :block] @ anchor_values
    AtA = (free.T @ free).tocsc()
    return np.concatenate([anchor_values, solve_spd(AtA, free.T @ rhs)])


def chordal(n_poses, between_list, anchor=(0.0, 0.0, 0.0)):
    e = _cat(between_list, Between)
    m = len(e.i)
    w_rot = np.sqrt(np.maximum(e.info[:, 2, 2], 1e-12))
    w_pos = np.sqrt(np.maximum(0.5 * (e.info[:, 0, 0] + e.info[:, 1, 1]), 1e-12))

    # 1. rotations as 2-vectors v = (c, s). theta_j = theta_i + theta_z gives two linear rows:
    #    c_j - (cz c_i - sz s_i) = 0
    #    s_j - (sz c_i + cz s_i) = 0
    cz, sz = np.cos(e.z[:, 2]), np.sin(e.z[:, 2])
    one = np.ones(m)
    rows, cols, vals = [], [], []
    for r_off, coef_ci, coef_si in ((0, cz, -sz), (1, sz, cz)):
        row = 2 * np.arange(m) + r_off
        rows += [row, row, row]
        cols += [2 * e.i, 2 * e.i + 1, 2 * e.j + r_off]
        vals += [-coef_ci * w_rot, -coef_si * w_rot, one * w_rot]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * m, 2 * n_poses))
    v = _solve_anchored(A, np.zeros(2 * m), 2, np.array([np.cos(anchor[2]), np.sin(anchor[2])]))
    v = v.reshape(n_poses, 2)
    theta = np.arctan2(v[:, 1], v[:, 0])

    # 2. positions: t_j - t_i = R_i t_z
    rhs = np.einsum("nab,nb->na", rot(theta[e.i]), e.z[:, :2]) * w_pos[:, None]
    rows, cols, vals = [], [], []
    for d in range(2):
        row = 2 * np.arange(m) + d
        rows += [row, row]
        cols += [2 * e.i + d, 2 * e.j + d]
        vals += [-w_pos, w_pos]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * m, 2 * n_poses))
    t = _solve_anchored(A, rhs.ravel(), 2, np.array(anchor[:2], float)).reshape(n_poses, 2)
    return np.column_stack([t, theta])
