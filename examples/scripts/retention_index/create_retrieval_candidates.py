#!/usr/bin/env python
"""Create RI-based candidate sets for retrieval evaluation.

Uses sorted array + binary search for O(log N) lookups instead of
O(N) brute force, providing ~100,000x speedup for large reference databases.

Supports two RI source modes:
1. from_exp: Window centered on experimental query RI; candidates ranked by AIRI-predicted RI.
   Ground truth recall depends on AIRI prediction error — not guaranteed to be in the set.
2. from_pred: Window centered on AIRI-predicted query RI; simulates no experimental RI available.

Two selection modes:
1. Top-N: Select the N molecules with closest RI
2. Range: Select all molecules within a specified RI error margin
"""

import argparse
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

# RI types to consider
RI_TYPES = ["StdNP", "SemiStdNP", "StdPolar"]

# Default model paths (for from_pred mode)
# Can be overridden via environment variable ICICLE_RI_MODEL_DIR
DEFAULT_CHECKPOINT_DIR = Path(
    os.getenv(
        "ICICLE_RI_MODEL_DIR",
        "/home/magled/icicle-dev/data/NIST2023_GCMS_main/retention_index",
    )
)
DEFAULT_CHECKPOINTS = {
    "StdNP": DEFAULT_CHECKPOINT_DIR / "ri_predictor_StdNP.joblib",
    "SemiStdNP": DEFAULT_CHECKPOINT_DIR / "ri_predictor_SemiStdNP.joblib",
    "StdPolar": DEFAULT_CHECKPOINT_DIR / "ri_predictor_StdPolar.joblib",
}


def _process_smiles_chunk(
    args: tuple[list[tuple[int, str]], int, int],
) -> list[tuple[int, np.ndarray]]:
    """Process a chunk of SMILES in a worker process."""
    indexed_smiles, radius, n_bits = args
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    results = []
    for idx, smiles in indexed_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            results.append((idx, np.array(fp)))
    return results


def predict_ri_for_test_molecules(
    smiles_list: list[str],
    models: dict,
    n_workers: int,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, np.ndarray]:
    """Predict RI for all 3 types for test molecules."""
    n_samples = len(smiles_list)
    predictions = {ri_type: np.full(n_samples, np.nan) for ri_type in RI_TYPES}

    print(f"Predicting RI for {n_samples} test molecules...")

    # Create indexed SMILES list
    indexed_smiles = [(i, smi) for i, smi in enumerate(smiles_list)]

    # Split into chunks for workers
    chunk_size = max(1, len(indexed_smiles) // n_workers)
    chunks = [
        indexed_smiles[i : i + chunk_size]
        for i in range(0, len(indexed_smiles), chunk_size)
    ]

    # Prepare arguments
    chunk_args = [(chunk, radius, n_bits) for chunk in chunks]

    # Generate fingerprints in parallel
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_smiles_chunk, chunk_args),
                total=len(chunk_args),
                desc="  Generating fingerprints",
            )
        )

    # Collect results
    fingerprints = []
    valid_indices = []
    for chunk_result in results:
        for idx, fp in chunk_result:
            valid_indices.append(idx)
            fingerprints.append(fp)

    if fingerprints:
        X = np.array(fingerprints)

        # Predict with each model
        for ri_type in RI_TYPES:
            model = models[ri_type]
            preds = model.predict(X)
            predictions[ri_type][valid_indices] = preds
            valid_count = np.sum(~np.isnan(predictions[ri_type]))
            print(
                f"  {ri_type}: {valid_count}/{n_samples} valid predictions, "
                f"range=[{np.nanmin(preds):.0f}, {np.nanmax(preds):.0f}]"
            )

    return predictions


def process_ri_type(
    ri_type: str,
    test_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    mode: str,
    top_n: int,
    ri_margin: float,
    n_jobs: int,
    ri_source: str = "from_exp",
) -> tuple[pd.DataFrame, dict]:
    """Process a single RI type using FAST sorted + binary search algorithm.

    Parameters
    ----------
    ri_source : str
        Either 'from_exp' (use experimental RI) or 'from_pred' (use predicted RI)
        - from_exp: Window centered on experimental query RI; ground truth recall
          depends on AIRI prediction error for the true molecule (not guaranteed)
        - from_pred: Window centered on predicted query RI; simulates no experimental RI

    NO MULTIPROCESSING - binary search is so fast that multiprocessing overhead
    (serializing 94M arrays) is the bottleneck, not computation.
    """
    pred_col = f"ri_{ri_type}"
    exp_col = f"ri_{ri_type}"

    # Filter reference molecules that have predictions for this RI type
    ref_with_ri = ref_df[ref_df[pred_col].notna()].copy()
    if len(ref_with_ri) == 0:
        print(f"  Warning: No reference molecules with {pred_col} predictions")
        return pd.DataFrame(), {}

    if ri_source == "from_exp":
        # Use experimental RI: only process test molecules with experimental RI
        test_with_ri = test_df[test_df[exp_col].notna()].copy()
        if len(test_with_ri) == 0:
            print(f"  Warning: No test molecules with experimental {ri_type}")
            return pd.DataFrame(), {}
        ri_col_name = exp_col
        print(
            f"  {ri_type}: {len(test_with_ri)} test molecules with exp RI, "
            f"{len(ref_with_ri):,} reference molecules"
        )
    else:  # from_pred
        # Use predicted RI: process test molecules with predicted RI
        test_with_ri = test_df[test_df[f"pred_{ri_type}"].notna()].copy()
        if len(test_with_ri) == 0:
            print(f"  Warning: No test molecules with predicted {ri_type}")
            return pd.DataFrame(), {}
        ri_col_name = f"pred_{ri_type}"
        n_with_exp = test_with_ri[exp_col].notna().sum()
        print(
            f"  {ri_type}: {len(test_with_ri)} test molecules with pred RI "
            f"({n_with_exp} have exp RI), {len(ref_with_ri):,} reference molecules"
        )

    # Prepare reference arrays
    ref_ri = ref_with_ri[pred_col].values.astype(np.float32)
    ref_smiles = (
        ref_with_ri["SMILES"].values
        if "SMILES" in ref_with_ri.columns
        else np.array([""] * len(ref_with_ri))
    )
    ref_inchikeys = (
        ref_with_ri["InChIKey"].values
        if "InChIKey" in ref_with_ri.columns
        else np.array([""] * len(ref_with_ri))
    )
    ref_ik_prefixes = np.array(
        [ik[:14] if isinstance(ik, str) and ik else "" for ik in ref_inchikeys]
    )

    # FAST: Pre-sort reference RI values (ONE-TIME cost)
    print(f"  Sorting {len(ref_ri):,} reference RI values...")
    sorted_indices = np.argsort(ref_ri)
    sorted_ri = ref_ri[sorted_indices]
    print(
        f"  Sorting complete. RI range: [{sorted_ri[0]:.0f}, {sorted_ri[-1]:.0f}]"
    )

    n_ref = len(sorted_ri)
    all_results = []

    # Process each query
    for _, row in tqdm(
        test_with_ri.iterrows(),
        total=len(test_with_ri),
        desc=f"  {ri_type}",
    ):
        ik = row["inchi_key"]
        target_smiles = row["standardized_smiles"]
        # Get RI value based on source (experimental or predicted)
        query_ri_val = row[ri_col_name]
        exp_ri_val = row[exp_col] if pd.notna(row[exp_col]) else np.nan
        target_ik_prefix = ik[:14] if isinstance(ik, str) and ik else ""

        # Binary search using query RI to find insertion point
        pos = np.searchsorted(sorted_ri, query_ri_val)

        if mode == "top-n":
            # Take neighbors around the insertion point
            half_n = (
                top_n + 100
            )  # Take extra to ensure we get top_n after filtering
            left = max(0, pos - half_n)
            right = min(n_ref, pos + half_n)

            # Get candidates in this window
            window_ri = sorted_ri[left:right]
            window_diff = np.abs(window_ri - query_ri_val)

            # Get top_n smallest differences
            if len(window_diff) <= top_n:
                top_k_local = np.arange(len(window_diff))
            else:
                top_k_local = np.argpartition(window_diff, top_n)[:top_n]

            # Sort by difference
            order = np.argsort(window_diff[top_k_local])
            top_k_local = top_k_local[order]

            # Map back to original indices
            selected_sorted_idx = left + top_k_local
            selected_orig_idx = sorted_indices[selected_sorted_idx]
            selected_ri_diff = window_diff[top_k_local]

        else:  # range mode
            # Binary search for left and right bounds
            left_pos = np.searchsorted(sorted_ri, query_ri_val - ri_margin)
            right_pos = np.searchsorted(
                sorted_ri, query_ri_val + ri_margin, side="right"
            )

            if left_pos >= right_pos:
                continue

            # Get all candidates in range (no sorting needed)
            selected_sorted_idx = np.arange(left_pos, right_pos)
            selected_orig_idx = sorted_indices[selected_sorted_idx]
            selected_ri_diff = np.abs(
                sorted_ri[left_pos:right_pos] - query_ri_val
            )

        # Check exp_in_set (ground truth in candidate set)
        selected_ik_prefixes = ref_ik_prefixes[selected_orig_idx]
        exp_in_set = (
            target_ik_prefix in selected_ik_prefixes
            if target_ik_prefix
            else False
        )

        # Build results
        for rank, (orig_idx, ri_diff) in enumerate(
            zip(selected_orig_idx, selected_ri_diff), 1
        ):
            all_results.append(
                (
                    ik,
                    target_smiles,
                    query_ri_val,
                    exp_ri_val,
                    ref_smiles[orig_idx],
                    ref_inchikeys[orig_idx],
                    ref_ri[orig_idx],
                    ri_diff,
                    rank,
                    exp_in_set,
                )
            )

    # Convert to DataFrame
    if not all_results:
        return pd.DataFrame(), {}

    result_df = pd.DataFrame(
        all_results,
        columns=[
            "target_inchikey",
            "target_smiles",
            "target_query_ri",  # RI used for matching (exp or pred based on ri_source)
            "target_exp_ri",  # Experimental RI (for validation)
            "candidate_smiles",
            "candidate_inchikey",
            "candidate_ri",
            "ri_diff",
            "candidate_rank",
            "exp_in_set",
        ],
    )

    # Calculate validity stats
    queries_per_target = result_df.groupby("target_inchikey").agg(
        {"exp_in_set": "first", "target_exp_ri": "first"}
    )
    # Count only queries that have experimental RI
    queries_with_exp = queries_per_target[
        queries_per_target["target_exp_ri"].notna()
    ]
    n_queries_total = len(queries_per_target)
    n_queries_with_exp = len(queries_with_exp)
    n_exp_in_set = (
        queries_with_exp["exp_in_set"].sum() if n_queries_with_exp > 0 else 0
    )
    validity_pct = (
        (n_exp_in_set / n_queries_with_exp * 100)
        if n_queries_with_exp > 0
        else 0.0
    )

    stats = {
        "ri_type": ri_type,
        "n_queries_total": len(test_with_ri),
        "n_queries_with_exp": n_queries_with_exp,
        "n_exp_in_set": int(n_exp_in_set),
        "validity_pct": validity_pct,
        "n_ref_molecules": len(ref_with_ri),
    }

    return result_df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Create RI-based candidate sets for retrieval evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ri-predictions-file",
        required=True,
        help="Path to reference database with RI predictions (e.g., PubChem)",
    )
    parser.add_argument(
        "--ri-dataset-file",
        required=True,
        help="Path to RI dataset file with experimental RI values and splits",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--output-prefix",
        default="ri_candidates",
        help="Prefix for output files (default: ri_candidates)",
    )
    parser.add_argument(
        "--ri-source",
        choices=["from_exp", "from_pred"],
        default="from_exp",
        help="RI source: 'from_exp' uses experimental RI (ensures ground truth in set), "
        "'from_pred' uses predicted RI (more realistic scenario)",
    )
    parser.add_argument(
        "--mode",
        choices=["top-n", "range"],
        default="top-n",
        help="Candidate selection mode: 'top-n' for N closest, 'range' for within margin",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Number of top candidates to select per query (for top-n mode)",
    )
    parser.add_argument(
        "--ri-margin",
        type=float,
        default=100.0,
        help="RI error margin for candidate selection (for range mode)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to process (default: test)",
    )
    parser.add_argument(
        "--ri-types",
        nargs="+",
        default=None,
        help="RI types to process (default: all 3)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: all cores)",
    )

    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = cpu_count()

    # Validate input files
    if not Path(args.ri_predictions_file).exists():
        print(
            f"Error: Predictions file not found: {args.ri_predictions_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not Path(args.ri_dataset_file).exists():
        print(
            f"Error: RI dataset file not found: {args.ri_dataset_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model checkpoints if using from_pred mode
    models = None
    if args.ri_source == "from_pred":
        print("Loading RI prediction models for from_pred mode...")
        for ri_type, path in DEFAULT_CHECKPOINTS.items():
            if not path.exists():
                print(
                    f"Error: Model checkpoint not found: {path}",
                    file=sys.stderr,
                )
                sys.exit(1)
        models = {}
        for ri_type, path in DEFAULT_CHECKPOINTS.items():
            models[ri_type] = joblib.load(path)
            print(f"  {ri_type}: {type(models[ri_type]).__name__}")

    # Load reference database (PubChem with predictions)
    print(f"\nLoading reference predictions from: {args.ri_predictions_file}")
    ref_df = pd.read_csv(args.ri_predictions_file, sep="\t")
    print(f"  Loaded {len(ref_df):,} reference molecules")

    # Load RI dataset with experimental values
    print(f"\nLoading RI dataset from: {args.ri_dataset_file}")
    ri_df = pd.read_csv(args.ri_dataset_file, sep="\t")
    print(f"  Loaded {len(ri_df):,} molecules")

    # Filter to test split
    test_df = ri_df[ri_df["split"] == args.split].copy()
    print(f"  {args.split} split: {len(test_df):,} molecules")

    # Filter to molecules with at least one experimental RI
    exp_cols = [f"ri_{rt}" for rt in RI_TYPES]
    has_any_exp_ri = test_df[exp_cols].notna().any(axis=1)
    test_df = test_df[has_any_exp_ri].copy()
    print(f"  With at least one exp RI: {len(test_df):,} molecules")

    # Report experimental RI availability per type
    for rt in RI_TYPES:
        n_with_exp = test_df[f"ri_{rt}"].notna().sum()
        print(f"    {rt}: {n_with_exp} with experimental RI")

    # Predict RI for test molecules if using from_pred mode
    if args.ri_source == "from_pred":
        print("\n" + "=" * 60)
        print("Step 1: Predict RI for test molecules (from_pred mode)")
        print("=" * 60)
        smiles_list = test_df["standardized_smiles"].tolist()
        predictions = predict_ri_for_test_molecules(
            smiles_list, models, args.n_jobs
        )
        # Add predictions to test_df
        for ri_type in RI_TYPES:
            test_df[f"pred_{ri_type}"] = predictions[ri_type]

    # Determine which RI types to process
    ri_types = args.ri_types if args.ri_types else RI_TYPES

    print("\n" + "=" * 60)
    print(f"Create candidate sets using {args.ri_source} RI")
    print("=" * 60)
    print(f"RI source: {args.ri_source}")
    print(f"Processing RI types: {ri_types}")
    print(f"Mode: {args.mode}, top_n={args.top_n}, ri_margin={args.ri_margin}")
    print(f"Using {args.n_jobs} parallel workers\n")

    # Process each RI type separately
    all_stats = []
    for ri_type in ri_types:
        result_df, stats = process_ri_type(
            ri_type,
            test_df,
            ref_df,
            args.mode,
            args.top_n,
            args.ri_margin,
            args.n_jobs,
            args.ri_source,
        )

        if len(result_df) > 0:
            all_stats.append(stats)

            # Save separate file for this RI type
            output_path = (
                output_dir
                / f"{args.output_prefix}_{args.ri_source}_{ri_type}.tsv"
            )
            result_df.to_csv(output_path, sep="\t", index=False)
            print(f"  Saved: {output_path} ({len(result_df):,} rows)")
            print(
                f"    Validity (queries with exp RI): "
                f"{stats['n_exp_in_set']}/{stats['n_queries_with_exp']} = "
                f"{stats['validity_pct']:.2f}%"
            )

    # Write summary log file
    log_path = (
        output_dir / f"{args.output_prefix}_{args.ri_source}_summary.log"
    )
    with open(log_path, "w") as f:
        f.write("RI Candidate Set Generation Log\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"RI source: {args.ri_source}\n")
        f.write(f"RI predictions file: {args.ri_predictions_file}\n")
        f.write(f"RI dataset file: {args.ri_dataset_file}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Mode: {args.mode}\n")
        if args.mode == "top-n":
            f.write(f"Top-N: {args.top_n}\n")
        else:
            f.write(f"RI margin: {args.ri_margin}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"N jobs: {args.n_jobs}\n\n")

        f.write("METHODOLOGY:\n")
        f.write("-" * 60 + "\n")
        if args.ri_source == "from_exp":
            f.write(
                "1. Test molecules: those with experimental RI for each specific type\n"
            )
            f.write("2. Window centered on EXPERIMENTAL query RI\n")
            f.write(
                "3. Candidates selected by proximity of their AIRI-predicted RI to the query window\n"
            )
            f.write(
                "4. Ground truth recall depends on AIRI prediction error for the true molecule;\n"
                "   it is NOT guaranteed to be in the set (see validity stats below)\n"
            )
            f.write(
                "5. FAST algorithm: sorted array + binary search O(log N)\n\n"
            )
        else:
            f.write(
                "1. Test molecules: those with at least one experimental RI\n"
            )
            f.write(
                "2. Predicted RI for test molecules using trained models\n"
            )
            f.write(
                "3. Candidates selected based on PREDICTED RI (realistic scenario)\n"
            )
            f.write("4. Reference database uses predicted RI values\n")
            f.write("5. Experimental RI used only for validity checking\n")
            f.write(
                "6. FAST algorithm: sorted array + binary search O(log N)\n\n"
            )

        f.write(f"Reference database: {len(ref_df):,} molecules\n")
        f.write(f"Test molecules (with any exp RI): {len(test_df):,}\n\n")

        f.write("Results by RI type:\n")
        f.write("-" * 60 + "\n")

        total_queries_with_exp = 0
        total_exp_in_set = 0

        for stats in all_stats:
            ri_type = stats["ri_type"]
            f.write(f"\n{ri_type}:\n")
            f.write(f"  Total queries: {stats['n_queries_total']}\n")
            f.write(f"  Queries with exp RI: {stats['n_queries_with_exp']}\n")
            f.write(
                f"  Reference molecules with predictions: {stats['n_ref_molecules']:,}\n"
            )
            f.write(
                f"  Ground truth in set: {stats['n_exp_in_set']}/{stats['n_queries_with_exp']}\n"
            )
            f.write(f"  Validity: {stats['validity_pct']:.2f}%\n")

            total_queries_with_exp += stats["n_queries_with_exp"]
            total_exp_in_set += stats["n_exp_in_set"]

        overall_validity = (
            (total_exp_in_set / total_queries_with_exp * 100)
            if total_queries_with_exp > 0
            else 0
        )
        f.write("\n" + "=" * 60 + "\n")
        f.write(
            f"OVERALL VALIDITY: {total_exp_in_set}/{total_queries_with_exp} = "
            f"{overall_validity:.2f}%\n"
        )
        f.write(
            "(Across all RI types, counting queries with experimental RI)\n"
        )
        f.write("=" * 60 + "\n")

    print(f"\nSaved summary log: {log_path}")
    print(f"\n{'=' * 60}")
    print(
        f"OVERALL VALIDITY: {total_exp_in_set}/{total_queries_with_exp} = "
        f"{overall_validity:.2f}%"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
