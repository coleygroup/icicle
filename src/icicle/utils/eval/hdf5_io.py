"""Helper functions for HDF5 I/O operations."""

import logging
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm


def save_predictions_as_hdf5(
    predictions_data: List[Dict], output_path: Path
) -> None:
    """Save predictions, ground truth spectra, and metrics to HDF5 file.

    Parameters
    ----------
    predictions_data : List[Dict]
        List of dictionaries containing prediction data with keys:
        'smiles', 'mol_id', 'inchi_key', 'predicted_mz_bins',
        'predicted_intensities', 'ground_truth_mz_bins',
        'ground_truth_intensities', and 'metrics'
    output_path : Path
        Full path to the HDF5 file to create/overwrite
    """
    logging.info(
        f"Saving evaluation spectra and metrics to HDF5: {output_path}"
    )

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        for entry in tqdm(predictions_data, desc="Saving to HDF5"):
            smiles = entry["smiles"]
            mol_id = entry["mol_id"]
            inchi_key = entry["inchi_key"]

            group_name = str(inchi_key)

            if group_name in hf:
                logging.warning(
                    f"Duplicate inchi_key '{inchi_key}' found. Overwriting existing entry."
                )
                del hf[group_name]

            grp = hf.create_group(group_name)

            # Store basic metadata as attributes
            grp.attrs["smiles"] = smiles
            grp.attrs["inchi_key"] = inchi_key

            # Save predicted spectrum
            grp.create_dataset(
                "predicted_mz_bins",
                data=entry["predicted_mz_bins"],
                compression="gzip",
            )
            grp.create_dataset(
                "mz_bins",
                data=entry["predicted_mz_bins"],
                compression="gzip",
            )
            grp.create_dataset(
                "predicted_intensities",
                data=entry["predicted_intensities"],
                compression="gzip",
            )

            # Save ground truth spectrum
            grp.create_dataset(
                "ground_truth_mz_bins",
                data=entry["ground_truth_mz_bins"],
                compression="gzip",
            )
            grp.create_dataset(
                "ground_truth_intensities",
                data=entry["ground_truth_intensities"],
                compression="gzip",
            )

            # Save metrics as attributes within a 'metrics' subgroup
            metrics_grp = grp.create_group("metrics")
            for metric_name, metric_value in entry["metrics"].items():
                if metric_name not in ["smiles", "mol_id"]:
                    if isinstance(
                        metric_value, (float, int, bool, str, np.number)
                    ):
                        metrics_grp.attrs[metric_name] = metric_value
                    else:
                        logging.warning(
                            f"Skipping non-serializable metric '{metric_name}' for {inchi_key}: {type(metric_value)}"
                        )

            hf.flush()

    logging.info(
        f"Successfully saved {len(predictions_data)} entries to {output_path}"
    )
