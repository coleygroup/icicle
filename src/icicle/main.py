"""Main entry point for the ICICLE CLI."""

import argparse
import os
import time
import pandas as pd
import torch
import csv
import h5py
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from icicle.models.eims_predictor import (
    EIMSPredictorWithFragmentGenerator,
    EIMSPredictorFromFullEnumeration,
)
from icicle.models.intensity_model import IntensityPredictor
from icicle.utils.visualization.mass_spectra import plot_mass_spectrum
from icicle.utils import inchikey_from_smiles

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_smiles_list(args) -> List[str]:
    """Reads SMILES strings from command line or a CSV file."""
    smiles_list = []
    if args.smiles:
        smiles_list.extend(args.smiles)
    elif args.smiles_file:
        try:
            with open(args.smiles_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        smiles_list.append(row[0].strip())
            logging.info(
                f"Loaded {len(smiles_list)} SMILES from {args.smiles_file}"
            )
        except FileNotFoundError:
            logging.error(f"SMILES file not found: {args.smiles_file}")
            exit(1)
    if not smiles_list:
        logging.error("No SMILES provided. Use --smiles or --smiles-file.")
        exit(1)
    return smiles_list


def save_results(
    results: List[Dict[str, Any]], output_format: str, output_folder: str
):
    """Saves prediction results to the specified format."""
    if output_format == "csv":
        data_to_save = []
        for result in results:
            smiles = result["smiles"]
            inchikey = inchikey_from_smiles(smiles) if smiles else ""
            mz_bins = result["mz_bins"]
            intensities = result["intensities"]

            # Flatten mz_bins and intensities into a single string or multiple columns
            mz_str = " ".join(map(str, mz_bins))
            intensity_str = " ".join(map(str, intensities))

            data_to_save.append(
                {
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "mz_bins": mz_str,
                    "intensities": intensity_str,
                    "num_fragments": result.get("num_fragments", "N/A"),
                }
            )
        df = pd.DataFrame(data_to_save)
        df.to_csv(f"{output_folder}/results.csv", index=False)
        logging.info(f"Results saved to CSV: {output_folder}/results.csv")

    elif output_format == "hdf5":
        with h5py.File(f"{output_folder}/results.hdf5", "w") as f:
            for i, result in enumerate(results):
                smiles = result["smiles"]
                inchikey = (
                    inchikey_from_smiles(smiles) if smiles else f"spectrum_{i}"
                )
                group = f.create_group(inchikey)
                group.attrs["smiles"] = smiles
                group.create_dataset("mz_bins", data=result["mz_bins"])
                group.create_dataset("intensities", data=result["intensities"])
                group.attrs["num_fragments"] = result.get("num_fragments", 0)
                # You can also save fragment details if needed (omitted for now)
                # if result.get("fragments"):
                #     fragments_group = group.create_group("fragments")
                #     for frag_id, frag_data in result["fragments"].items():
                #         frag_group = fragments_group.create_group(frag_id.replace("/", "_")) # HDF5 group names cannot contain '/'
                #         for k, v in frag_data.items():
                #             # Convert non-scalar values or complex objects to string or suitable format
                #             if isinstance(v, (list, np.ndarray)):
                #                 frag_group.create_dataset(k, data=np.array(v))
                #             else:
                #                 frag_group.attrs[k] = str(v)

        logging.info(f"Results saved to HDF5: {output_folder}/results.hdf5")

    else:
        logging.warning(
            f"Unsupported output format: {output_format}. Results not saved to file."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Predict mass spectra from SMILES using EIMS models."
    )
    parser.add_argument(
        "--intensity-predictor",
        type=str,
        required=True,
        help="Path to the intensity predictor checkpoint.",
    )
    parser.add_argument(
        "--fragment-generator",
        type=str,
        help="Path to the fragment generator checkpoint. If not provided, full enumeration is used.",
    )
    parser.add_argument(
        "--smiles",
        nargs="+",
        help="List of SMILES strings to process (e.g., 'CCO' 'CCC').",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        help="Path to a CSV file containing SMILES strings (one SMILES per line).",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=".",
        help="Folder where to save the output files.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "hdf5", "none"],
        help="Output format for saving predictions: 'csv', 'hdf5', or 'none'.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot predicted mass spectra."
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Print the time taken for each prediction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for prediction (e.g., 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=50,
        help="Maximum number of nodes for fragment generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for fragment generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for predictions (currently only used for sequential processing).",
    )  # Note: Actual batching for predict_from_smiles is not directly implemented in _BaseEIMSPredictor's predict_from_smiles but rather in batch_predict_from_smiles. For now, it's a conceptual batch size. --> TODO

    parser.add_argument(
        "--top-fragments-to-plot",
        type=int,
        default=0,
        help="Number of top fragments to label on the mass spectrum plot.",
    )

    args = parser.parse_args()

    smiles_list = get_smiles_list(args)
    if not smiles_list:
        logging.error("No SMILES provided for prediction.")
        return

    # Initialize the EIMS Predictor
    model = None
    if args.fragment_generator:
        logging.info("Initializing EIMSPredictorWithFragmentGenerator...")
        model = EIMSPredictorWithFragmentGenerator()
        model.load_from_checkpoint(
            fragment_generator_checkpoint=args.fragment_generator,
            intensity_predictor_checkpoint=args.intensity_predictor,
        )
    else:
        logging.info("Initializing EIMSPredictorFromFullEnumeration...")
        ip = IntensityPredictor.load_from_checkpoint(
            args.intensity_predictor, map_location="cpu"
        )
        model = EIMSPredictorFromFullEnumeration(
            min_mz=ip.min_mz,
            max_mz=ip.max_mz,
            bin_width=ip.bin_width,
            intensity_predictor=ip,
        )

    if model is None:
        logging.error("Failed to initialize EIMS predictor.")
        return

    model.to(args.device)
    model.eval()

    all_results = []
    all_times = []

    logging.info(f"Starting prediction for {len(smiles_list)} SMILES strings.")

    # Process in batches (conceptually, as batch_predict_from_smiles is sequential under the hood for now)
    for i in range(0, len(smiles_list), args.batch_size):
        batch_smiles = smiles_list[i : i + args.batch_size]
        logging.info(
            f"Processing batch {i // args.batch_size + 1}/{(len(smiles_list) + args.batch_size - 1) // args.batch_size} with {len(batch_smiles)} SMILES."
        )
        if args.time:
            batch_start_time = time.time()
        batch_results = model.batch_predict_from_smiles(
            smiles_list=batch_smiles,
            device=args.device,
            max_nodes=args.max_nodes,
            threshold=args.threshold,
        )
        if args.time:
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            logging.info(f"Batch prediction took {batch_time:.2f} seconds.")

        all_results.extend(batch_results)
        if args.time:
            for _ in batch_smiles:
                all_times.append(
                    batch_time / len(batch_smiles)
                )  # Average time per SMILES in this batch

        if args.plot:
            for result in batch_results:
                if result["smiles"]:
                    fig = plot_mass_spectrum(
                        result["mz_bins"],
                        result["intensities"],
                        smiles=result["smiles"],
                        title=f"Predicted Spectrum for {result['smiles']}",
                        figsize=(
                            8,
                            4,
                        ),
                        output_path=f"{args.output_folder}/{result['smiles']}.png",
                        max_fragments=args.top_fragments_to_plot,
                        fragments=result["fragments"],
                    )
                    plt.close(fig)

    if args.time and all_times:
        avg_time_per_spectrum = sum(all_times) / len(all_times)
        logging.info(
            f"Overall average time taken: {avg_time_per_spectrum:.4f} seconds per spectrum"
        )

    # Save results to file
    if args.output_format != "none":
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        save_results(all_results, args.output_format, args.output_folder)
    else:
        logging.info(
            "No output file format specified. Results not saved to file."
        )

    logging.info("Prediction process complete.")


if __name__ == "__main__":
    main()
