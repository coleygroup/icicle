"""Deduplicate stereoisomers from split files.

Stereoisomers share the same InChI skeleton (first part of InChI key before the
dash). This script keeps only one representative entry per InChI skeleton
GLOBALLY across all splits, preventing data leakage between train/val/test.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_inchi_skeleton(inchi_key: str) -> str | None:
    """Extract the InChI skeleton (first connectivity layer) from an InChI key.

    The InChI key has format: XXXXXXXXXXXXXX-YYYYYYYYYY-Z
    The first part (X's) represents the connectivity layer and is identical
    for stereoisomers.
    """
    if not isinstance(inchi_key, str) or "-" not in inchi_key:
        return None
    return inchi_key.split("-")[0]


def deduplicate_stereoisomers(
    split_file: str,
    metadata_file: str | None = None,
    output_file: str | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Deduplicate stereoisomers from a split file.

    Args:
        split_file: Path to input split TSV file with mol_id and split columns
        metadata_file: Path to metadata TSV file with mol_id and inchi_key columns.
            Required if split file doesn't contain inchi_key column.
        output_file: Path to output deduplicated TSV file. If None, auto-generates
            by appending '_deduplicated' to the input filename.
        keep: Which entry to keep when duplicates are found ('first' or 'last')

    Returns:
        Deduplicated DataFrame
    """
    split_path = Path(split_file)

    # Auto-generate output filename if not provided
    if output_file is None:
        output_file = split_path.parent / f"{split_path.stem}_deduplicated.tsv"

    logger.info(f"Loading split file: {split_file}")
    df = pd.read_csv(split_file, sep="\t", dtype={"mol_id": str})

    original_count = len(df)
    logger.info(f"Original entries: {original_count}")

    # Check if inchi_key column exists, if not load from metadata
    if "inchi_key" not in df.columns:
        if metadata_file is None:
            raise ValueError(
                "Split file doesn't contain 'inchi_key' column. "
                "Please provide --metadata-file to load inchi_key from metadata."
            )
        logger.info(f"Loading inchi_key from metadata: {metadata_file}")
        metadata = pd.read_csv(
            metadata_file,
            sep="\t",
            usecols=["mol_id", "inchi_key"],
            dtype={"mol_id": str},
        )
        df = df.merge(metadata, on="mol_id", how="left")

        # Check for missing inchi_keys
        missing = df["inchi_key"].isna().sum()
        if missing > 0:
            logger.warning(
                f"Found {missing} entries without inchi_key in metadata"
            )
            df = df.dropna(subset=["inchi_key"])

    # Extract InChI skeleton
    df["inchi_skeleton"] = df["inchi_key"].apply(get_inchi_skeleton)

    # Count stereoisomer groups before deduplication (globally)
    skeleton_counts = df.groupby("inchi_skeleton").size()
    stereoisomer_groups = skeleton_counts[skeleton_counts > 1]
    logger.info(
        f"Found {len(stereoisomer_groups)} stereoisomer groups (entries with same skeleton)"
    )

    # Check for cross-split stereoisomers (important for random splits)
    cross_split_skeletons = (
        df.groupby("inchi_skeleton")["split"].nunique().loc[lambda x: x > 1]
    )
    if len(cross_split_skeletons) > 0:
        logger.info(
            f"Found {len(cross_split_skeletons)} stereoisomer groups spanning multiple splits "
            "(these will be deduplicated to prevent data leakage)"
        )

    # Deduplicate GLOBALLY, keeping one entry per InChI skeleton
    # This prevents data leakage from stereoisomers in different splits
    df_deduplicated = (
        df.groupby("inchi_skeleton", as_index=False)
        .agg(
            {
                "mol_id": keep,
                "inchi_key": keep,
                "split": keep,
            }
        )
        .drop(columns=["inchi_skeleton"])
    )

    # Restore original column order
    df_deduplicated = df_deduplicated[["mol_id", "inchi_key", "split"]]

    deduplicated_count = len(df_deduplicated)
    removed_count = original_count - deduplicated_count

    logger.info(f"Deduplicated entries: {deduplicated_count}")
    logger.info(f"Removed {removed_count} duplicate stereoisomers")

    # Print per-split statistics
    logger.info("\nPer-split statistics:")
    for split_name in ["train", "val", "test"]:
        original_split = len(df[df["split"] == split_name])
        dedup_split = len(
            df_deduplicated[df_deduplicated["split"] == split_name]
        )
        removed_split = original_split - dedup_split
        if original_split > 0:
            logger.info(
                f"  {split_name}: {original_split} -> {dedup_split} "
                f"(removed {removed_split}, {removed_split/original_split*100:.1f}%)"
            )

    # Save deduplicated split
    df_deduplicated.to_csv(output_file, sep="\t", index=False)
    logger.info(f"\nSaved deduplicated split to: {output_file}")

    return df_deduplicated


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate stereoisomers from split files"
    )
    parser.add_argument(
        "--split-file",
        required=True,
        help="Path to input split TSV file",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Path to metadata TSV file (required if split file lacks inchi_key column)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to output deduplicated TSV file (default: input_deduplicated.tsv)",
    )
    parser.add_argument(
        "--keep",
        choices=["first", "last"],
        default="first",
        help="Which entry to keep when duplicates are found (default: first)",
    )

    args = parser.parse_args()

    deduplicate_stereoisomers(
        split_file=args.split_file,
        metadata_file=args.metadata_file,
        output_file=args.output_file,
        keep=args.keep,
    )


if __name__ == "__main__":
    main()
