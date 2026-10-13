"""SE(2) helpers. Poses are (x, y, theta) arrays; every function also works on stacks of shape (n, 3)."""

import numpy as np


def wrap(a):
    """Wrap angles to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rot(theta):
    """Rotation matrices, shape (..., 2, 2)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def compose(a, b):
    """a (+) b: apply b in the frame of a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    x = a[..., 0] + c * b[..., 0] - s * b[..., 1]
    y = a[..., 1] + s * b[..., 0] + c * b[..., 1]
    return np.stack([x, y, wrap(a[..., 2] + b[..., 2])], -1)


def inverse(a):
    a = np.asarray(a, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    x = -c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] - c * a[..., 1]
    return np.stack([x, y, wrap(-a[..., 2])], -1)


def between(a, b):
    """The pose of b expressed in the frame of a: a^-1 (+) b."""
    return compose(inverse(a), b)
