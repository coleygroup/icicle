"""Compute molecular complexity scores (SAScore, NPScore, SPScore, Boettcher)
for the NIST test queries used in the global PubChem retrieval structural
analysis, and merge them into the existing merged_top1_structural_global.csv.

Usage:
    uv run examples/scripts/evaluation/compute_molecular_complexity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from icicle.utils.chem.complexity.product_complexity import (
    get_BoettcherScore,
    get_NPScore,
    get_SAScore,
    get_SPScore,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def compute_scores(mol_id: int, smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    return {
        "mol_id": mol_id,
        "sascore": _safe(get_SAScore, mol) if mol is not None else None,
        "npscore": _safe(get_NPScore, mol) if mol is not None else None,
        "spscore": _safe(get_SPScore, mol) if mol is not None else None,
        "boettcher": _safe(get_BoettcherScore, smiles),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=REPO_ROOT
        / "figures"
        / "retrieval_structural_analysis"
        / "merged_top1_structural_global.csv",
    )
    parser.add_argument(
        "--metadata-tsv",
        type=Path,
        default=REPO_ROOT / "data" / "NIST2023_GCMS_main" / "metadata.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "figures"
        / "retrieval_structural_analysis"
        / "merged_top1_structural_global_with_complexity.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged = pd.read_csv(args.merged_csv)
    metadata = pd.read_csv(
        args.metadata_tsv, sep="\t", usecols=["mol_id", "standardized_smiles"]
    )

    df = merged.merge(metadata, on="mol_id", how="left")
    missing_smiles = df["standardized_smiles"].isna().sum()
    if missing_smiles:
        print(
            f"Warning: {missing_smiles} mol_ids have no SMILES in metadata.tsv"
        )

    scores = [
        compute_scores(row.mol_id, row.standardized_smiles)
        for row in tqdm(
            df.dropna(subset=["standardized_smiles"]).itertuples(),
            total=df["standardized_smiles"].notna().sum(),
            desc="Complexity scores",
        )
    ]
    scores_df = pd.DataFrame(scores)

    result = merged.merge(scores_df, on="mol_id", how="left")
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
