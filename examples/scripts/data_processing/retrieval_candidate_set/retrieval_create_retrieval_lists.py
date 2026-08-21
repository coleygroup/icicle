"""Create and Standardize Retrieval Lists.

This script is the main workflow for a specific dataset. It performs three tasks:
1.  Standardizes Molecules: It loads the dataset labels and the formula map and
    creates a single, authoritative mapping from InChIKey to a canonical SMILES
    string. This resolves inconsistencies like different tautomers or stereoisomers upfront.
2.  Generates Candidate Lists: For each target molecule, it finds all isomeric
    non-stereo SMILES from the map to serve as candidates.
3.  Ranks Candidates: It ranks the non-stereo candidates by calculating the
    Tanimoto similarity of a representative stereoisomer against the target.
    The final output is a ranked list of distinct non-stereo SMILES.
"""

import argparse
import pickle
from pathlib import Path
from functools import partial
import logging
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from icicle.utils import chunked_parallel, get_morgan_fp_from_smi


def standardize_and_get_map(
    labels_df: pd.DataFrame, formula_map: dict
) -> dict:
    """Creates a definitive InChIKey -> SMILES map to resolve
    inconsistencies."""
    inchi_to_smi_map = {}
    for _, row in labels_df.iterrows():
        inchi_to_smi_map[row["inchikey"]] = row["standardized_smiles"]

    for formula, isomers in formula_map.items():
        for non_stereo_smi, stereo_set in isomers.items():
            for stereo_smi, inchi in stereo_set:
                if inchi not in inchi_to_smi_map:
                    inchi_to_smi_map[inchi] = stereo_smi
    return inchi_to_smi_map


def process_example(obj: dict, max_k: int = 50) -> dict:
    """Ranks non-stereo SMILES candidates by similarity."""
    if max_k is None:
        max_k = int(1e10)

    candidate_groups = obj.get("cands", {})
    ranked_candidates = []

    target_fp = get_morgan_fp_from_smi(obj["non_stereo_smiles"])

    for non_stereo_smi, stereo_set in candidate_groups.items():
        if stereo_set:
            try:
                cand_fp = get_morgan_fp_from_smi(non_stereo_smi)
            except ValueError:
                logging.warning(
                    f"Skipping unparseable candidate SMILES: {non_stereo_smi!r}"
                )
                continue

            intersect = np.dot(target_fp, cand_fp)
            union = target_fp.sum() + cand_fp.sum() - intersect
            sim = intersect / (union + 1e-22)
            ranked_candidates.append((sim, non_stereo_smi))

    ranked_candidates.sort(key=lambda x: x[0], reverse=True)
    top_k = ranked_candidates[:max_k]
    obj["tani_sims"] = np.array([sim for sim, _ in top_k] if top_k else [])
    obj["cands"] = np.array([ns_smi for _, ns_smi in top_k] if top_k else [])

    return obj


def none_or_int(value):
    """Helper type for argparse to allow 'None' as a value."""
    if value.lower() == "none":
        return None
    return int(value)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and standardize retrieval lists for a dataset."
    )
    parser.add_argument(
        "--input-map",
        type=str,
        required=True,
        help="Path to the subset formula map pickle file.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        required=True,
        help="Path to the dataset-specific labels TSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        help="Optional path to a data split file (e.g., train/test).",
    )
    parser.add_argument(
        "--max-k",
        type=none_or_int,
        default=50,
        help="Maximum number of candidates to retrieve.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers to use.",
    )
    return parser.parse_args()


def main(args):
    """Main execution function."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_stem = ""
    if args.split_file:
        split_stem = f"_{Path(args.split_file).stem}"

    output_pickle_path = (
        output_dir / f"cands_pickled{split_stem}_{args.max_k}.pkl"
    )
    output_tsv_path = output_dir / f"cands_df{split_stem}_{args.max_k}.tsv"

    logging.info("Loading data files...")
    df_labels = pd.read_csv(args.labels_file, sep="\t")
    with open(args.input_map, "rb") as f:
        form_map = pickle.load(f)

    if args.split_file:
        logging.info(f"Applying split from {args.split_file}...")
        split_df = pd.read_csv(args.split_file, sep="\t")
        name_col, inchikey_col, split_col = split_df.columns

        test_names = set(split_df[split_df[split_col] == "test"][inchikey_col])
        df_labels = df_labels[
            df_labels["inchi_key"].isin(test_names)
        ].reset_index(drop=True)

    logging.info("Standardizing all SMILES representations...")
    inchi_to_smi_map = standardize_and_get_map(df_labels, form_map)
    df_labels["smiles"] = df_labels["inchikey"].map(inchi_to_smi_map)
    df_labels.dropna(subset=["smiles"], inplace=True)
    logging.info(
        f"Standardization complete. Processing {len(df_labels)} entries."
    )

    logging.info("Generating candidate lists...")
    df_labels["non_stereo_smiles"] = df_labels["smiles"].apply(
        lambda smi: Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), isomericSmiles=False
        )
    )
    obj_list = df_labels.to_dict("records")

    for obj in tqdm(obj_list, desc="Preparing candidates"):
        candidate_isomers = form_map.get(obj["formula"], {})
        # Assign all non-stereo isomers as candidates, excluding the target itself
        obj["cands"] = {
            ns_smi: s_set
            for ns_smi, s_set in candidate_isomers.items()
            if ns_smi != obj["non_stereo_smiles"]
        }

    logging.info(
        f"Ranking candidates in parallel with {args.workers} workers..."
    )
    process_fn = partial(process_example, max_k=args.max_k)
    processed_list = chunked_parallel(
        obj_list, process_fn, max_cpu=args.workers
    )
    processed = {p["inchi_key"]: p for p in processed_list}
    logging.info("Ranking complete.")

    logging.info(f"Saving final results to {output_dir}...")
    with open(output_pickle_path, "wb") as f:
        pickle.dump(processed, f)

    entries = []
    for spec, entry in tqdm(processed.items(), desc="Formatting output TSV"):
        # Add the target itself as a candidate with similarity 1.0
        entries.append(
            {
                # "inchi_key": spec,
                "target_inchikey": entry["inchikey"],
                "target_smiles": entry["smiles"],
                "candidate_smiles": entry[
                    "non_stereo_smiles"
                ],  # Candidate is itself (non-stereo)
                "tanimoto_similarity": 1.0,
            }
        )
        # Add the other ranked candidates
        for cand_non_stereo_smi, cand_tani in zip(
            entry["cands"], entry["tani_sims"]
        ):
            entries.append(
                {
                    # "inchi_key": spec,
                    "target_inchikey": entry["inchikey"],
                    "target_smiles": entry["smiles"],
                    "candidate_smiles": cand_non_stereo_smi,
                    "tanimoto_similarity": cand_tani,
                }
            )

    df_out = pd.DataFrame(entries)
    df_out.to_csv(output_tsv_path, sep="\t", index=False)
    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    main(args)
