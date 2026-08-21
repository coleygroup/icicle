#!/usr/bin/env python
"""Add 2D InChIKeys and molecular descriptors to the PubChem AIRI predictions
TSV.

Computes per-molecule in one SMILES-parsing pass:
  - InChIKey     : 2D (stereochemistry-stripped)
  - mw           : exact monoisotopic mass
  - molecular_formula : e.g. "C10H20O2"
  - dbe          : degree of unsaturation  (1 + (2C + N - H - X) / 2)
  - is_aromatic  : bool, has at least one aromatic ring

Usage:
    python add_inchikeys_descriptors_pubchem.py \
        --input  data/PubChem/260303_full_PubChem_AIRI_with_RI.tsv \
        --output data/PubChem/260303_full_PubChem_AIRI_with_RI_inchikey.tsv \
        --workers 16 \
        --chunksize 50000
"""

import argparse
import multiprocessing as mp

import pandas as pd
from rdkit.Chem import MolFromSmiles, RemoveStereochemistry
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm.auto import tqdm

HALOGEN_SYMBOLS = frozenset(("F", "Cl", "Br", "I"))


def _calc_dbe(mol) -> float:
    """Degree of unsaturation: 1 + (2C + N - H - X) / 2."""
    c = h = n = x = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        h += atom.GetTotalNumHs()
        if sym == "C":
            c += 1
        elif sym == "N":
            n += 1
        elif sym in HALOGEN_SYMBOLS:
            x += 1
        elif sym == "H":
            h += 1  # explicit H atoms (rare after sanitize)
    return 1.0 + (2 * c + n - h - x) / 2.0


def smiles_to_descriptors(
    smiles: str,
) -> tuple[str | None, float | None, str | None, float | None, bool | None]:
    """Return (inchikey_2d, mw, formula, dbe, is_aromatic) for a SMILES."""
    if not smiles or not isinstance(smiles, str):
        return None, None, None, None, None
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None, None

    mw = ExactMolWt(mol)
    formula = CalcMolFormula(mol)
    dbe = _calc_dbe(mol)
    is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    RemoveStereochemistry(mol)
    inchi = MolToInchi(mol)
    inchikey = InchiToInchiKey(inchi) if inchi else None

    return inchikey, mw, formula, dbe, is_aromatic


def _worker_init():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def process_chunk(
    smiles_list: list[str],
) -> list[tuple]:
    return [smiles_to_descriptors(s) for s in smiles_list]


def main():
    parser = argparse.ArgumentParser(
        description="Add InChIKey + descriptors to PubChem AIRI TSV"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/PubChem/260303_full_PubChem_AIRI_with_RI.tsv",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/PubChem/260303_full_PubChem_AIRI_with_RI_inchikey.tsv",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=min(16, mp.cpu_count())
    )
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    for smiles_col in ("input_smiles", "smiles", "SMILES"):
        if smiles_col in df.columns:
            break
    else:
        raise KeyError(f"No SMILES column found. Columns: {list(df.columns)}")
    smiles_list = df[smiles_col].tolist()

    print(
        f"Computing descriptors using {args.workers} workers "
        f"(chunksize={args.chunksize:,})..."
    )

    chunks = [
        smiles_list[i : i + args.chunksize]
        for i in range(0, len(smiles_list), args.chunksize)
    ]

    rows: list[tuple] = []
    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        for chunk_result in tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Chunks",
        ):
            rows.extend(chunk_result)

    df["InChIKey"] = [r[0] for r in rows]
    df["mw"] = [r[1] for r in rows]
    df["molecular_formula"] = [r[2] for r in rows]
    df["dbe"] = [r[3] for r in rows]
    df["is_aromatic"] = [r[4] for r in rows]

    for col in ["InChIKey", "mw", "molecular_formula", "dbe", "is_aromatic"]:
        n_miss = df[col].isna().sum()
        if n_miss:
            print(f"  Warning: {n_miss:,} rows missing {col}")

    print(f"Saving to {args.output}...")
    df.to_csv(args.output, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
