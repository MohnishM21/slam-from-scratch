"""Checks on the real dataset. Skipped when data/ has not been downloaded (as in CI)."""

from pathlib import Path

import numpy as np
import pytest

from slam.mrclam import load, measurement_residuals

DATA = Path(__file__).resolve().parents[1] / "data" / "MRCLAM1"
pytestmark = pytest.mark.skipif(not (DATA / "Robot1_Odometry.dat").exists(),
                                reason="run scripts/download_data.py first")


def per_landmark_bearing_bias(robot, fix):
    d = load(DATA, robot, fix_known_swaps=fix)
    _, eb, _ = measurement_residuals(d)
    j = d.measurements[:, 1].astype(int)
    return np.array([np.median(np.abs(eb[j == k])) for k in range(15)])


@pytest.mark.parametrize("robot", [1, 2, 3, 4, 5])
def test_landmark_swap_evidence(robot):
    """Without the fix, landmarks 5 and 11 (subjects 11, 17) are off by about 1.5 rad in bearing."""
    raw = per_landmark_bearing_bias(robot, fix=False)
    fixed = per_landmark_bearing_bias(robot, fix=True)
    assert raw[5] > 0.8 and raw[11] > 0.8
    assert fixed.max() < 0.05


def test_loader_keeps_only_landmark_sightings():
    d = load(DATA, 1)
    assert d.landmarks.shape == (15, 2)
    assert set(np.unique(d.measurements[:, 1]).astype(int)) <= set(range(15))
    assert np.all(np.diff(d.measurements[:, 0]) >= 0)
    assert d.measurements[0, 0] >= d.t0
