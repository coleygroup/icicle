"""Create train/validation/test splits for mass spectrometry ML datasets.

Supports multiple splitting strategies:
- Random: Random stratified splits
- Scaffold: Bemis-Murcko scaffold-based splits
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Create train/val/test splits for molecular datasets."""

    def __init__(
        self,
        metadata_path: str,
        output_dir: str = "./splits",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ):
        """Initialize the data splitter.

        Args:
            metadata_path: Path to metadata.tsv file
            output_dir: Directory to save split files
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_state: Random seed for reproducibility
        """
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.df = self._load_and_filter_data()
        logger.info(f"Loaded {len(self.df)} molecules for splitting")

    def _load_and_filter_data(self) -> pd.DataFrame:
        """Load metadata and filter to molecules with spectra."""
        df = pd.read_csv(self.metadata_path, sep="\t")

        df_filtered = df[df["has_spectrum"] == True].copy()
        logger.info(
            f"Filtered from {len(df)} to {len(df_filtered)} molecules with spectra"
        )

        required_cols = ["mol_id", "standardized_smiles", "inchi_key"]
        missing_cols = [
            col for col in required_cols if col not in df_filtered.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df_filtered

    def get_bemis_murcko_scaffold(self, smiles: str) -> str:
        """Get Bemis-Murcko scaffold for a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "invalid_smiles"
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return "scaffold_error"

    def create_random_split(self) -> pd.DataFrame:
        """Create random splits using pandas sampling."""
        logger.info("Creating random splits...")

        df_copy = self.df[["mol_id", "inchi_key"]].copy()

        val_sample = df_copy.sample(
            frac=self.val_ratio, random_state=self.random_state
        )
        remaining = df_copy.drop(val_sample.index)

        test_frac = self.test_ratio / (1 - self.val_ratio)
        test_sample = remaining.sample(
            frac=test_frac, random_state=self.random_state + 1
        )

        df_copy["split"] = "train"
        df_copy.loc[val_sample.index, "split"] = "val"
        df_copy.loc[test_sample.index, "split"] = "test"

        return df_copy

    def create_scaffold_split(self) -> pd.DataFrame:
        """Create scaffold-based splits to prevent scaffold leakage."""
        logger.info("Creating scaffold splits...")

        logger.info("Computing Bemis-Murcko scaffolds...")
        scaffolds = {}
        scaffold_to_mols = defaultdict(list)

        for idx, row in self.df.iterrows():
            scaffold = self.get_bemis_murcko_scaffold(
                row["standardized_smiles"]
            )
            scaffolds[row["mol_id"]] = scaffold
            scaffold_to_mols[scaffold].append(row["mol_id"])

        logger.info(f"Found {len(scaffold_to_mols)} unique scaffolds")

        scaffold_sizes = [
            (scaffold, len(mols))
            for scaffold, mols in scaffold_to_mols.items()
        ]
        scaffold_sizes.sort(key=lambda x: x[1], reverse=True)

        train_mols, val_mols, test_mols = [], [], []
        train_size, val_size, test_size = 0, 0, 0
        total_mols = len(self.df)

        for scaffold, size in scaffold_sizes:
            mols = scaffold_to_mols[scaffold]

            train_frac = train_size / total_mols if total_mols > 0 else 0
            val_frac = val_size / total_mols if total_mols > 0 else 0

            if train_frac < self.train_ratio:
                train_mols.extend(mols)
                train_size += size
            elif val_frac < self.val_ratio:
                val_mols.extend(mols)
                val_size += size
            else:
                test_mols.extend(mols)
                test_size += size

        split_df = pd.DataFrame(
            {"mol_id": self.df["mol_id"], "inchi_key": self.df["inchi_key"]}
        )
        split_df["split"] = "train"
        split_df.loc[split_df["mol_id"].isin(val_mols), "split"] = "val"
        split_df.loc[split_df["mol_id"].isin(test_mols), "split"] = "test"

        return split_df

    def save_split(self, split_df: pd.DataFrame, split_name: str):
        """Save split to TSV file."""
        output_path = self.output_dir / f"{split_name}.tsv"
        split_df.to_csv(output_path, sep="\t", index=False)

        split_counts = split_df["split"].value_counts()
        logger.info(f"Saved {split_name} split to {output_path}")
        logger.info(
            f"  Train: {split_counts.get('train', 0)} ({split_counts.get('train', 0) / len(split_df) * 100:.1f}%)"
        )
        logger.info(
            f"  Val:   {split_counts.get('val', 0)} ({split_counts.get('val', 0) / len(split_df) * 100:.1f}%)"
        )
        logger.info(
            f"  Test:  {split_counts.get('test', 0)} ({split_counts.get('test', 0) / len(split_df) * 100:.1f}%)"
        )

    def create_all_splits(self):
        """Create random and scaffold splits."""
        splits_created = []

        random_split = self.create_random_split()
        self.save_split(random_split, "random")
        splits_created.append("random")

        scaffold_split = self.create_scaffold_split()
        self.save_split(scaffold_split, "scaffold")
        splits_created.append("scaffold")

        logger.info(
            f"Successfully created {len(splits_created)} splits: {splits_created}"
        )
        return splits_created


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for mass spec data"
    )
    parser.add_argument(
        "--metadata-path", required=True, help="Path to metadata.tsv file"
    )
    parser.add_argument(
        "--output-dir",
        default="./splits",
        help="Output directory for split files",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test set ratio"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--split-types",
        nargs="+",
        choices=["random", "scaffold", "all"],
        default=["all"],
        help="Types of splits to create",
    )

    args = parser.parse_args()

    splitter = DataSplitter(
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    if "all" in args.split_types:
        splitter.create_all_splits()
    else:
        for split_type in args.split_types:
            if split_type == "random":
                split_df = splitter.create_random_split()
                splitter.save_split(split_df, "random")
            elif split_type == "scaffold":
                split_df = splitter.create_scaffold_split()
                splitter.save_split(split_df, "scaffold")


if __name__ == "__main__":
    main()
