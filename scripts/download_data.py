"""Download the datasets from their original sources into data/.

    python scripts/download_data.py

UTIAS MRCLAM (Leung, Halpern, Barfoot, Liu, IJRR 2011)
    https://asrl.utias.utoronto.ca/datasets/mrclam/
2D pose-graph benchmarks (Carlone and Censi, IEEE T-RO 2014)
    https://lucacarlone.mit.edu/datasets/
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MRCLAM = {"MRCLAM1": "ftp://asrl3.utias.utoronto.ca/MRCLAM/MRCLAM1.zip"}
POSE_GRAPHS = {
    "input_INTEL_g2o.g2o": "https://www.dropbox.com/s/vcz8cag7bo0zlaj/input_INTEL_g2o.g2o?dl=1",
    "input_MITb_g2o.g2o": "https://www.dropbox.com/s/d8fcn1jg1mebx8f/input_MITb_g2o.g2o?dl=1",
    "input_M3500_g2o.g2o": "https://www.dropbox.com/s/gmdzo74b3tzvbrw/input_M3500_g2o.g2o?dl=1",
}


def fetch(url, dest):
    if dest.exists():
        print(f"have  {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(dest)


def safe_extract(zip_path, out_dir):
    """Extract only plain data files, refusing paths that would escape out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename)
            if info.is_dir() or name.suffix.lower() != ".dat":
                continue
            target = (out_dir / name.name).resolve()
            if out_dir.resolve() not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {info.filename}")
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main():
    for name, url in MRCLAM.items():
        archive = DATA / "downloads" / f"{name}.zip"
        fetch(url, archive)
        out = DATA / name
        if not (out / "Robot1_Groundtruth.dat").exists():
            safe_extract(archive, out)
            print(f"extracted {len(list(out.glob('*.dat')))} files to {out.relative_to(ROOT)}")
    for name, url in POSE_GRAPHS.items():
        fetch(url, DATA / "pose_graphs" / name)


if __name__ == "__main__":
    main()
