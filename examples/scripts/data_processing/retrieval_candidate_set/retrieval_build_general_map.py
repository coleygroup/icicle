"""Build General PubChem Formula Map.

This script performs the time-consuming, one-time task of processing the entire
PubChem database into a structured formula map. It can be used to create subsets for
several datasets moving forward.

The script saves a single pickled dictionary with the structure:
{
    "Molecular Formula": {
        "Non-Isomeric SMILES": {
            ("Isomeric SMILES_1", "InChIKey_1"),
            ("Isomeric SMILES_2", "InChIKey_2"),
            ...
        }
    }
}
"""

import argparse
import multiprocessing as mp
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import logging
from rdkit import Chem, RDLogger
from rdkit.Chem.AllChem import GetMolFrags
from tqdm import tqdm


# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def process_smiles_chunk(
    smiles_chunk: List[str],
) -> List[Tuple[str, Tuple[str, str, str]]]:
    """Processes a chunk of SMILES strings into key chemical identifiers."""
    results = []
    for smi in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Remove any salts/fragments by taking the largest fragment
                frags = GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if frags:
                    mol = max(frags, key=lambda x: x.GetNumAtoms())

                formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                non_stereo_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                stereo_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                inchi_key = Chem.MolToInchiKey(mol)

                results.append(
                    (formula, (non_stereo_smi, stereo_smi, inchi_key))
                )
            else:
                results.append(("", ("", "", "")))
        except:
            results.append(("", ("", "", "")))
    return results


def create_nested_dict():
    """Helper function to create the nested defaultdict structure."""
    return defaultdict(set)


def build_formula_map(
    smi_file: str,
    dump_file: str = None,
    n_jobs: int = -1,
    chunk_size: int = 5000,
    save_every: int = 500000,
) -> Dict[str, Dict[str, Set[Tuple[str, str]]]]:
    """Builds and saves the nested formula map."""
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count() - 1)

    form_to_mols = defaultdict(create_nested_dict)
    smi_chunks = []
    current_chunk = []
    total_processed = 0

    logging.info("Reading and processing SMILES...")
    with open(smi_file) as fp:
        for line in tqdm(fp):
            if line.strip():
                smi = line.strip().split("\t")[1].strip()
                current_chunk.append(smi)

                if len(current_chunk) >= chunk_size:
                    smi_chunks.append(current_chunk)
                    current_chunk = []

                    if len(smi_chunks) >= n_jobs:
                        with mp.Pool(processes=n_jobs) as pool:
                            for chunk_results in pool.imap(
                                process_smiles_chunk, smi_chunks
                            ):
                                for formula, (
                                    non_stereo_smi,
                                    stereo_smi,
                                    inchi_key,
                                ) in chunk_results:
                                    if formula and non_stereo_smi:
                                        form_to_mols[formula][
                                            non_stereo_smi
                                        ].add((stereo_smi, inchi_key))
                                total_processed += chunk_size

                        if total_processed % save_every == 0:
                            logging.info(
                                f"\nProcessed {total_processed} molecules, saving progress..."
                            )
                            if dump_file:
                                with open(f"{dump_file}.partial", "wb") as f:
                                    pickle.dump(dict(form_to_mols), f)

                        smi_chunks = []

    # Process final remaining chunks
    if current_chunk:
        smi_chunks.append(current_chunk)

    if smi_chunks:
        with mp.Pool(processes=n_jobs) as pool:
            for chunk_results in pool.imap(process_smiles_chunk, smi_chunks):
                for formula, (
                    non_stereo_smi,
                    stereo_smi,
                    inchi_key,
                ) in chunk_results:
                    if formula and non_stereo_smi:
                        form_to_mols[formula][non_stereo_smi].add(
                            (stereo_smi, inchi_key)
                        )

    # Convert defaultdicts to regular dicts for pickling
    final_map = {f: dict(isomers) for f, isomers in form_to_mols.items()}

    if dump_file:
        logging.info(f"\nSaving final map to {dump_file}...")
        with open(dump_file, "wb") as f:
            pickle.dump(final_map, f)

    return final_map


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a general formula map from a large SMILES file."
    )
    parser.add_argument(
        "--pubchem-file",
        type=str,
        required=True,
        help="Path to the input PubChem SMILES file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output pickled map file.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of CPU cores to use."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_formula_map(
        smi_file=args.pubchem_file,
        dump_file=args.output_file,
        n_jobs=args.n_jobs,
        chunk_size=5000,
        save_every=500000,
    )
    logging.info("General formula map building complete.")
