"""Reproduce the evidence that MRCLAM dataset 1 has two landmark positions swapped.

    python scripts/check_landmark_swap.py

For every landmark, compares each sighting with the range and bearing predicted
from the Vicon ground truth (robot pose and surveyed landmark position).
"""

from pathlib import Path

import numpy as np

from slam.mrclam import load, measurement_residuals

DATA = Path(__file__).resolve().parents[1] / "data" / "MRCLAM1"


def main():
    print("median |bearing error| (rad) per landmark; subjects 6-20")
    print("robot  fix   " + " ".join(f"{s:>5d}" for s in range(6, 21)))
    for robot in range(1, 6):
        for fix in (False, True):
            d = load(DATA, robot, fix_known_swaps=fix)
            _, eb, _ = measurement_residuals(d)
            j = d.measurements[:, 1].astype(int)
            cells = " ".join(f"{np.median(np.abs(eb[j == k])):5.2f}" for k in range(15))
            print(f"  {robot}    {'yes' if fix else 'no ':3s}  {cells}")


if __name__ == "__main__":
    main()
