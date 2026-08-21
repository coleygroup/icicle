"""Convert numpy arrays in candidates pickle to plain Python lists.

Pickles created with NumPy 2.x store arrays using numpy._core internals
which are unreadable in older NumPy environments (e.g. MF-GPU conda env).
This script re-saves the pickle with Python lists for cross-version compatibility.

Run this once in the main uv environment (NumPy 2.x) before running inference for baselines (MF-GPU).

Usage
-----
uv run examples/scripts/data_processing/retrieval_candidate_set/convert_candidates_pickle.py \
    data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_50.pkl
"""

import argparse
import pickle
from pathlib import Path


def convert(input_path: str) -> None:
    """Load pickle, convert numpy arrays to lists, re-save."""
    import numpy as np

    path = Path(input_path)
    out_path = path.with_name(path.stem + "_compat.pkl")

    with open(path, "rb") as f:
        data = pickle.load(f)

    for entry in data.values():
        for key in ("cands", "tani_sims"):
            if key in entry and isinstance(entry[key], np.ndarray):
                entry[key] = entry[key].tolist()

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved compatible pickle to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle", help="Path to cands_pickled_*.pkl")
    args = parser.parse_args()
    convert(args.pickle)
