"""Helper functions for loading data."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def retrieve_ground_truth(
    labels_path: str,
    spectra_path: str,
    mol_ids: List[str],
    min_mz: float,
    max_mz: float,
    bin_width: float,
) -> Tuple[Dict, List[str]]:
    """Retrieve SMILES and experimental spectra for specified mol_ids.

    Parameters
    ----------
    labels_path : str
        Path to labels TSV file
    spectra_path : str
        Path to HDF5 file containing spectra
    mol_ids : List[str]
        List of molecule IDs to retrieve
    min_mz : float
        Minimum m/z value for binning
    max_mz : float
        Maximum m/z value for binning
    bin_width : float
        Width of each bin

    Returns
    -------
    Tuple[Dict, List[str]]
        Dictionary mapping SMILES to spectrum data and list of successfully
        retrieved SMILES
    """
    logging.info(
        f"Loading ground truth spectra from {spectra_path} and {labels_path} for {len(mol_ids)} molecules."
    )
    labels_df = pd.read_csv(labels_path, sep="\t")

    if (
        "mol_id" not in labels_df.columns
        or "standardized_smiles" not in labels_df.columns
        or "inchi_key" not in labels_df.columns
    ):
        raise ValueError(
            "labels_path must contain 'mol_id', 'standardized_smiles' and 'inchi_key' columns."
        )

    # Filter labels_df to only include the mol_ids we are interested in
    filtered_labels_df = labels_df[labels_df["mol_id"].isin(mol_ids)].copy()

    # Create binner for ground truth spectra
    from icicle.data.transforms.spectrum import SpecBinner

    binner = SpecBinner(min_mz=min_mz, max_mz=max_mz, bin_width=bin_width)
    num_bins = int((max_mz - min_mz) / bin_width)
    mz_values = np.linspace(min_mz, max_mz, num_bins, endpoint=False).astype(
        np.float32
    )

    ground_truth_map = {}
    retrieved_smiles_list = []
    with h5py.File(spectra_path, "r") as hf:
        for idx, row in tqdm(
            filtered_labels_df.iterrows(),
            total=len(filtered_labels_df),
            desc="Loading Experimental Spectra",
        ):
            mol_id = str(row["mol_id"])
            smiles = row["standardized_smiles"]
            inchi_key = row["inchi_key"]

            if mol_id in hf or inchi_key in hf:
                group = hf[mol_id]
                raw_mz = group["masses"][:]
                raw_intensities = group["intensities"][:]

                # Bin the spectrum using the same binner as the dataset
                binned_result = binner(raw_mz, raw_intensities)
                binned_intensities = binned_result["spectrum"].numpy()
                # Normalize intensities so that the highest intensity is 1
                binned_intensities = (
                    binned_intensities / binned_intensities.max()
                )

                ground_truth_map[smiles] = {
                    "mol_id": mol_id,
                    "inchi_key": inchi_key,
                    "standardized_smiles": smiles,
                    "mz_bins": mz_values,
                    "intensities": binned_intensities,
                }
                retrieved_smiles_list.append(smiles)
            else:
                logging.warning(
                    f"mol_id '{mol_id}' (SMILES: {smiles}) not found in {spectra_path}. Skipping."
                )
    logging.info(
        f"Successfully loaded ground truth for {len(retrieved_smiles_list)} SMILES."
    )
    return ground_truth_map, retrieved_smiles_list


def get_retrieval_candidates_from_labels(
    test_labels: pd.DataFrame, retrieval_labels: pd.DataFrame
) -> pd.DataFrame:
    """Generate retrieval candidates filtered by molecular formula.

    For each test spectrum, finds all candidates in the retrieval pool that
    share the same molecular formula. Ensures the correct answer is included
    and marks decoys.

    Parameters
    ----------
    test_labels : pd.DataFrame
        DataFrame with test labels containing columns: spec (mol_id),
        inchikey (inchi_key), formula, standardized_smiles
    retrieval_labels : pd.DataFrame
        DataFrame with all potential retrieval candidates

    Returns
    -------
    pd.DataFrame
        DataFrame with test labels and unique retrieval candidates, with
        'is_decoy' column indicating whether each row is a decoy
    """
    # Filter to keep only candidates that share formula with test spectra
    test_formulas = set(test_labels["formula"].dropna().unique())
    logging.info(f"Found {len(test_formulas)} unique formulas in test set")
    candidates = retrieval_labels[
        retrieval_labels["formula"].isin(test_formulas)
    ].copy()
    logging.info(
        f"Retrieved {len(candidates)} candidates matching test formulas"
    )

    # Create mappings from test_labels
    spec_to_inchikey = test_labels.set_index("spec")["inchikey"].to_dict()
    spec_to_formula = test_labels.set_index("spec")["formula"].to_dict()
    logging.info(f"Prepared mappings for {len(spec_to_inchikey)} spectra")

    # Process each spectrum to build candidate sets by formula
    clean_results = []
    duplicates_found = 0

    for spec_name, correct_inchikey in spec_to_inchikey.items():
        formula = spec_to_formula.get(spec_name)
        if formula is None or pd.isna(formula):
            continue

        # Get the test label row for this spectrum
        test_row = (
            test_labels[test_labels["spec"] == spec_name].iloc[0].to_dict()
        )
        test_row["is_decoy"] = False
        clean_results.append(test_row)

        # Get all candidates with the same formula
        formula_candidates = candidates[candidates["formula"] == formula]

        # Check for duplicate inchikeys in the candidates
        if formula_candidates["inchikey"].duplicated().any():
            duplicates_found += 1
            formula_candidates = formula_candidates.drop_duplicates(
                subset=["inchikey"], keep="first"
            )

        # Add candidates that don't match the correct inchikey
        for _, row in formula_candidates.iterrows():
            if row["inchikey"] != correct_inchikey:
                row_dict = row.to_dict()
                row_dict["spec"] = spec_name
                row_dict["is_decoy"] = True
                clean_results.append(row_dict)

    # Convert to DataFrame
    result_df = pd.DataFrame(clean_results)

    # Log statistics
    if len(result_df) == 0:
        logging.warning(
            "No retrieval candidates found - result DataFrame is empty"
        )
        return result_df

    num_specs = len(result_df["spec"].unique())
    num_candidates = len(result_df)

    logging.info(
        f"Found and fixed {duplicates_found} spectrum groups with duplicate inchikeys"
    )
    logging.info(
        f"Created retrieval dataset with {num_specs} spectra and {num_candidates} total candidates"
    )
    if num_specs > 0:
        logging.info(
            f"Average of {num_candidates / num_specs:.2f} candidates per spectrum"
        )

    return result_df
