"""Filter a splits file to only include molecules valid for RASSP inference.

RASSP can only handle molecules that satisfy:
  1. Contain only allowed elements: H, C, N, O, F, P, S, Cl
  2. Have ≤ max_n_atoms (default 48) atoms after adding explicit Hs
  3. Per-element count limits (H<50, C<46, N/O/F/P/S/Cl<30)
  4. Are a single connected fragment (no salts / mixtures)

Note: We skip the num_unique_frag_formulae check (max_n_formula ≤ 4096)
because it's expensive and rarely the binding constraint.

Usage:
    python filter_splits_for_rassp.py \
        --splits-path data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
        --metadata-path data/NIST2023_GCMS_main/metadata.tsv \
        --output-path data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated_rassp.tsv \
        --max-n-atoms 48
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# RASSP allowed elements (atomic numbers)
ALLOWED_ATOM_NUMS = {1, 6, 7, 8, 9, 15, 16, 17}
ALLOWED_ATOM_NUMS_LIST = [1, 6, 7, 8, 9, 15, 16, 17]
# Per-element formula limits (same order as ALLOWED_ATOM_NUMS_LIST)
FORMULA_LIMITS = np.array([50, 46, 30, 30, 30, 30, 30, 30])
IND_MAP = {anum: i for i, anum in enumerate(ALLOWED_ATOM_NUMS_LIST)}


def is_valid_for_rassp(smiles: str, max_n_atoms: int = 48) -> tuple[bool, str]:
    """Check if a SMILES string is valid for RASSP inference.

    Parameters
    ----------
    smiles : str
        SMILES string to validate.
    max_n_atoms : int
        Maximum number of atoms (including explicit Hs).

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason) where reason is empty if valid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid_smiles"

    # Check allowed elements (before adding Hs)
    atom_nums = {
        mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())
    }
    if not atom_nums.issubset(ALLOWED_ATOM_NUMS):
        disallowed = atom_nums - ALLOWED_ATOM_NUMS
        return False, f"disallowed_elements:{disallowed}"

    # Add explicit Hs (RASSP always does this)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    n_atoms = mol.GetNumAtoms()

    # Check atom count
    if n_atoms > max_n_atoms:
        return False, f"too_many_atoms:{n_atoms}>{max_n_atoms}"

    # Check per-element formula limits
    cvec = np.zeros(len(FORMULA_LIMITS), dtype=np.int16)
    for anum, count in zip(
        *np.unique(
            [mol.GetAtomWithIdx(ai).GetAtomicNum() for ai in range(n_atoms)],
            return_counts=True,
        )
    ):
        cvec[IND_MAP[anum]] = count
    if np.any(cvec >= FORMULA_LIMITS):
        return False, f"formula_limits_exceeded:{cvec}"

    # Check single fragment
    if len(Chem.GetMolFrags(mol)) > 1:
        return False, "multiple_fragments"

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Filter splits file to molecules valid for RASSP."
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        required=True,
        help="Path to input splits TSV file (mol_id, inchi_key, split).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Path to metadata TSV file (must contain mol_id and standardized_smiles).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output filtered splits TSV file.",
    )
    parser.add_argument(
        "--max-n-atoms",
        type=int,
        default=48,
        help="Maximum number of atoms including Hs (default: 48).",
    )
    args = parser.parse_args()

    # Load metadata to get mol_id -> SMILES mapping
    logger.info(f"Loading metadata from {args.metadata_path}")
    mol_id_to_smiles = {}
    with open(args.metadata_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mol_id_to_smiles[row["mol_id"]] = row["standardized_smiles"]
    logger.info(f"Loaded {len(mol_id_to_smiles)} molecules from metadata")

    # Load splits
    logger.info(f"Loading splits from {args.splits_path}")
    with open(args.splits_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        splits_rows = list(reader)
    logger.info(f"Loaded {len(splits_rows)} rows from splits")

    # Filter
    kept = 0
    skipped = 0
    skip_reasons = {}
    output_rows = []

    for row in splits_rows:
        mol_id = row["mol_id"]
        smiles = mol_id_to_smiles.get(mol_id)

        if smiles is None:
            skipped += 1
            skip_reasons["missing_smiles"] = (
                skip_reasons.get("missing_smiles", 0) + 1
            )
            continue

        valid, reason = is_valid_for_rassp(
            smiles, max_n_atoms=args.max_n_atoms
        )
        if valid:
            output_rows.append(row)
            kept += 1
        else:
            skipped += 1
            # Group by reason category (before the colon)
            reason_key = reason.split(":")[0]
            skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1

    # Write output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mol_id", "inchi_key", "split"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    # Report
    logger.info(
        f"Results: {kept} kept, {skipped} skipped out of {len(splits_rows)} total"
    )
    logger.info("Skip reasons:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count}")

    # Per-split breakdown
    split_counts = {}
    for row in output_rows:
        s = row["split"]
        split_counts[s] = split_counts.get(s, 0) + 1
    logger.info("Per-split counts in output:")
    for split_name, count in sorted(split_counts.items()):
        logger.info(f"  {split_name}: {count}")

    logger.info(f"Written to {output_path}")


if __name__ == "__main__":
    main()
