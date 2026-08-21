"""Build NIST RI eval TSV for pubchem_retrieval_eval.py.

Merges experimental RI from per-column-type AIRI parquets (test split of
random_no_xeno_aas_deduplicated) into a single TSV with columns:
    mol_id, inchi_key, split, ri_StdNP, ri_SemiStdNP, ri_StdPolar

Output: data/NIST2023_GCMS_main/retention_index_airi/ri_dataset_random_split_no_xeno_aas.tsv
"""

import argparse
from pathlib import Path

import pandas as pd


AIRI_DIRS = {
    "StdNP": "airi_data_stdnp_random",
    "SemiStdNP": "airi_data_semistdnp_random",
    "StdPolar": "airi_data_stdpolar_random",
}
SPLIT_FILE = "splits/random_no_xeno_aas_deduplicated.tsv"
OUTPUT_SUBDIR = "retention_index_airi"
OUTPUT_FILE = "ri_dataset_random_split_no_xeno_aas.tsv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/NIST2023_GCMS_main"),
        help="Root NIST dataset directory",
    )
    return p.parse_args()


def load_ri_parquet(
    data_dir: Path, col_type: str, subdir: str
) -> pd.DataFrame:
    """Load test-split parquet for one column type, return mol_id + ri
    column."""
    path = data_dir / subdir / "airi_test.parquet"
    df = pd.read_parquet(path)[["id", "experimental_ri"]]
    return df.rename(
        columns={"id": "mol_id", "experimental_ri": f"ri_{col_type}"}
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    meta = pd.read_csv(data_dir / "metadata.tsv", sep="\t")[
        ["mol_id", "inchi_key"]
    ]
    splits = pd.read_csv(data_dir / SPLIT_FILE, sep="\t")[["mol_id", "split"]]

    ri_parts = [
        load_ri_parquet(data_dir, col_type, subdir)
        for col_type, subdir in AIRI_DIRS.items()
    ]

    # outer merge: keep all molecules with at least one RI measurement
    ri = ri_parts[0]
    for part in ri_parts[1:]:
        ri = ri.merge(part, on="mol_id", how="outer")

    ri = ri.merge(meta, on="mol_id", how="left")
    ri = ri.merge(splits, on="mol_id", how="left")

    # all airi_test parquets already contain only test-split molecules, but filter
    # explicitly to be safe
    ri = ri[ri["split"] == "test"].reset_index(drop=True)

    out_dir = data_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILE
    ri.to_csv(out_path, sep="\t", index=False)

    print(f"Wrote {len(ri)} rows → {out_path}")
    for col_type in AIRI_DIRS:
        col = f"ri_{col_type}"
        print(f"  {col}: {ri[col].notna().sum()} non-null")


if __name__ == "__main__":
    main()
