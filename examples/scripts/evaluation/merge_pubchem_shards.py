"""Merge per-GPU PubChem inference shards into the final output HDF5.

Run once after all 8 independent per-GPU batch_infer.py processes (launched
via examples/scripts/inference/batch_infer_pubchem_workstation.sh) have
finished -- i.e. no results/inference/_shard_*.hdf5.progress.json files
remain.

Usage
-----
uv run examples/scripts/evaluation/merge_pubchem_shards.py \
    --output results/inference/pubchem_predictions_rerun_260630.hdf5 \
    --num-shards 8
"""

import argparse
import glob
from pathlib import Path

from icicle.batch_infer import _merge_failure_logs, _merge_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="Final merged HDF5 path."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shard files to merge.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    shard_paths = [
        str(output_dir / f"_shard_{i}.hdf5") for i in range(args.num_shards)
    ]

    incomplete = [p for p in shard_paths if glob.glob(p + ".progress.json")]
    if incomplete:
        raise SystemExit(
            f"Shards still in progress, aborting merge: {incomplete}"
        )
    missing = [p for p in shard_paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing shard files, aborting merge: {missing}")

    _merge_shards(shard_paths, args.output)
    _merge_failure_logs(shard_paths, args.output)
    print(f"Merged {args.num_shards} shards -> {args.output}")


if __name__ == "__main__":
    main()
