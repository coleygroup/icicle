#!/usr/bin/env python
"""Merge multiple AIRI prediction files (one per RI column type) into a single
TSV.

Usage:
    python merge_ri_predictions.py \
        --inputs pred_StdNP.csv pred_SemiStdNP.csv pred_StdPolar.csv \
        --output combined_predictions.tsv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-column-type AIRI predictions into one TSV"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV files (one per RI type, with ri_<type> columns)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output TSV file"
    )
    args = parser.parse_args()

    # Read first file as base (has SMILES, InChIKey, MW, etc.)
    print(f"Reading {args.inputs[0]}...")
    base_df = pd.read_csv(args.inputs[0])

    # Identify which ri_ columns are in the base
    ri_cols = [
        c
        for c in base_df.columns
        if c.startswith("ri_") and "_stddev" not in c
    ]
    stddev_cols = [
        c for c in base_df.columns if c.startswith("ri_") and "_stddev" in c
    ]
    print(f"  Found RI columns: {ri_cols}")

    # Merge additional files
    for fpath in args.inputs[1:]:
        print(f"Reading {fpath}...")
        df = pd.read_csv(fpath)
        new_ri_cols = [
            c for c in df.columns if c.startswith("ri_") and "_stddev" not in c
        ]
        new_stddev_cols = [
            c for c in df.columns if c.startswith("ri_") and "_stddev" in c
        ]
        print(f"  Found RI columns: {new_ri_cols}")

        # Determine the SMILES column to join on
        smiles_col = "smiles" if "smiles" in df.columns else "input_smiles"
        base_smiles_col = (
            "smiles" if "smiles" in base_df.columns else "input_smiles"
        )

        # Merge on SMILES (or InChIKey if available)
        if "InChIKey" in df.columns and "InChIKey" in base_df.columns:
            merge_cols = new_ri_cols + new_stddev_cols + ["InChIKey"]
            base_df = base_df.merge(df[merge_cols], on="InChIKey", how="left")
        else:
            merge_cols = new_ri_cols + new_stddev_cols + [smiles_col]
            base_df = base_df.merge(
                df[merge_cols],
                left_on=base_smiles_col,
                right_on=smiles_col,
                how="left",
                suffixes=("", "_drop"),
            )
            # Drop duplicate smiles column from merge
            base_df = base_df.drop(
                columns=[c for c in base_df.columns if c.endswith("_drop")]
            )

        ri_cols.extend(new_ri_cols)
        stddev_cols.extend(new_stddev_cols)

    # Reorder columns to match expected format:
    # SMILES, ID, InChIKey, MW, ri_StdNP, ri_SemiStdNP, ri_StdPolar
    smiles_col = "smiles" if "smiles" in base_df.columns else "input_smiles"
    ordered_cols = [smiles_col]
    for c in ["InChIKey", "MW"]:
        if c in base_df.columns:
            ordered_cols.append(c)
    for c in ["ri_StdNP", "ri_SemiStdNP", "ri_StdPolar"]:
        if c in base_df.columns:
            ordered_cols.append(c)
    # Add any remaining ri columns
    for c in ri_cols + stddev_cols:
        if c in base_df.columns and c not in ordered_cols:
            ordered_cols.append(c)

    # Rename smiles column
    base_df = base_df.rename(columns={smiles_col: "SMILES"})
    ordered_cols = ["SMILES" if c == smiles_col else c for c in ordered_cols]

    output_df = base_df[ordered_cols]

    # Save as TSV
    output_df.to_csv(args.output, sep="\t", index=True)
    print(f"\nMerged {len(output_df):,} molecules into {args.output}")
    print(f"Columns: {list(output_df.columns)}")


if __name__ == "__main__":
    main()
