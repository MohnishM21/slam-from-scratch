"""Read and write 2D pose graphs in g2o format (VERTEX_SE2 / EDGE_SE2)."""

import numpy as np

from .factor_graph import Between, Graph, Prior

UPPER = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def read_g2o(path, anchor_info=1e8):
    """Returns a Graph with the file's vertex estimates as the initial guess and pose 0 anchored."""
    ids, poses, edges = [], [], []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "VERTEX_SE2":
                ids.append(int(tok[1]))
                poses.append([float(v) for v in tok[2:5]])
            elif tok[0] == "EDGE_SE2":
                edges.append([float(v) for v in tok[1:12]])
    order = np.argsort(ids)
    index = {ids[k]: n for n, k in enumerate(order)}
    poses = np.array(poses)[order]
    edges = np.array(edges)
    info = np.zeros((len(edges), 3, 3))
    for k, (a, b) in enumerate(UPPER):
        info[:, a, b] = info[:, b, a] = edges[:, 5 + k]
    between = Between(
        i=np.array([index[int(v)] for v in edges[:, 0]]),
        j=np.array([index[int(v)] for v in edges[:, 1]]),
        z=edges[:, 2:5].copy(),
        info=info,
    )
    prior = Prior(0, poses[0].copy(), anchor_info * np.eye(3))
    return Graph(poses=poses, between=[between], priors=[prior])


def write_g2o(path, poses, between: Between):
    with open(path, "w") as fh:
        for n, p in enumerate(poses):
            fh.write(f"VERTEX_SE2 {n} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for i, j, z, info in zip(between.i, between.j, between.z, between.info):
            upper = " ".join(f"{info[a, b]:.6f}" for a, b in UPPER)
            fh.write(f"EDGE_SE2 {i} {j} {z[0]:.6f} {z[1]:.6f} {z[2]:.6f} {upper}\n")


def is_odometry(between: Between):
    """Edges between consecutive poses are odometry; the rest are loop closures."""
    return np.abs(between.j - between.i) == 1
