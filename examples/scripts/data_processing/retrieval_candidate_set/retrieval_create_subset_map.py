"""Create Dataset-Specific Subset Map.

This script takes the large, general formula map created by
retrieval_build_general_map.py and filters it to create a smaller subset
containing only the formulas relevant to a specific dataset (e.g., NIST 2023).
This speeds up downstream processing.
"""

import argparse
import pickle
import logging
import pandas as pd


def create_subset_map(full_map_path: str, labels_path: str, output_path: str):
    """Creates and saves a subset of the formula map."""
    logging.info(f"Loading full map from {full_map_path}...")
    with open(full_map_path, "rb") as f:
        full_map = pickle.load(f)

    logging.info(
        f"Loading labels from {labels_path} to identify required formulas..."
    )
    df = pd.read_csv(labels_path, sep="\t")
    required_formulas = set(df["formula"].unique())

    logging.info("Creating subset map...")
    subset_map = {
        formula: isomers
        for formula, isomers in full_map.items()
        if formula in required_formulas
    }

    logging.info(
        f"Saving subset map with {len(subset_map)} formulas to {output_path}..."
    )
    with open(output_path, "wb") as f:
        pickle.dump(subset_map, f)

    logging.info("Subset map creation complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset-specific subset from the general formula map."
    )
    parser.add_argument(
        "--full-map",
        type=str,
        required=True,
        help="Path to the general formula map pickle file.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        required=True,
        help="Path to the dataset-specific labels TSV file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output subset map pickle file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_subset_map(args.full_map, args.labels_file, args.output_file)
