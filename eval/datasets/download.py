"""Download all evaluation datasets into data/.

Usage: python -m eval.datasets.download
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASETS = {
    "clamber": {
        "url": "https://github.com/zt991211/CLAMBER.git",
        "dir": "data/clamber",
    },
    "qulac": {
        "url": "https://github.com/aliannejadi/qulac.git",
        "dir": "data/qulac",
    },
    "clariq": {
        "url": "https://github.com/aliannejadi/ClariQ.git",
        "dir": "data/clariq",
    },
}


def download_dataset(name: str, url: str, target_dir: str) -> bool:
    """Clone a dataset repo. Returns True if newly downloaded."""
    target = Path(target_dir)
    if target.exists() and any(target.iterdir()):
        print(f"  [{name}] Already exists at {target_dir}, skipping.")
        return False

    print(f"  [{name}] Cloning {url} -> {target_dir} ...")
    target.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, target_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [{name}] ERROR: {result.stderr.strip()}")
        return False

    print(f"  [{name}] Done.")
    return True


def print_summary() -> None:
    """Print dataset file counts after download."""
    print("\n--- Download Summary ---")
    for name, info in DATASETS.items():
        target = Path(info["dir"])
        if not target.exists():
            print(f"  {name}: NOT DOWNLOADED")
            continue

        # Count data files
        jsonl_files = list(target.rglob("*.jsonl"))
        json_files = list(target.rglob("*.json"))
        tsv_files = list(target.rglob("*.tsv"))
        data_files = jsonl_files + json_files + tsv_files
        print(f"  {name}: {len(data_files)} data files found")
        for f in data_files[:5]:
            print(f"    - {f.relative_to(target)}")


def main() -> None:
    print("Downloading D2C evaluation datasets...\n")
    for name, info in DATASETS.items():
        download_dataset(name, info["url"], info["dir"])
    print_summary()


if __name__ == "__main__":
    main()
