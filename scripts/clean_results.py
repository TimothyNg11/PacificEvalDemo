"""Remove all benchmark results and test artifacts."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = ROOT / "data" / "chroma_indexes"


def clean():
    removed = []

    # Remove benchmark results
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
        removed.append(str(RESULTS_DIR))

    # Remove leftover test index directories
    if CHROMA_DIR.exists():
        for d in CHROMA_DIR.iterdir():
            if d.is_dir() and d.name.startswith("test_retriever"):
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))

    # Remove __pycache__ and .pytest_cache directories
    for pattern in ("__pycache__", ".pytest_cache"):
        for d in ROOT.rglob(pattern):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))

    if removed:
        for path in removed:
            print(f"Removed {path}")
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    clean()
