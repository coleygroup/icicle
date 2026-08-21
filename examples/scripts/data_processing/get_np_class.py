"""Script to get NPClassifier classifications."""

import os
import pandas as pd
from tqdm import tqdm
import argparse

from icicle.utils import get_compound_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-file", type=str, required=True)
    parser.add_argument("--splits-file", type=str, required=True)
    parser.add_argument("--test-only", type=bool, default=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_file, sep="\t")
    splits_df = pd.read_csv(args.splits_file, sep="\t")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classifications = []
    if args.test_only:
        analysis_df = labels_df[splits_df["split"] == "test"]
    else:
        analysis_df = labels_df

    for idx, row in tqdm(
        analysis_df.iterrows(),
        total=len(analysis_df),
        desc="Getting NPClassifier classifications",
    ):
        class_info = get_compound_class(row["standardized_smiles"])
        classifications.append(
            {
                "mol_id": row["mol_id"],
                "class": class_info["class_results"][0]
                if class_info["class_results"]
                else "Unknown",
                "superclass": class_info["superclass_results"][0]
                if class_info["superclass_results"]
                else "Unknown",
                "pathway": class_info["pathway_results"][0]
                if class_info["pathway_results"]
                else "Unknown",
                "isglycoside": "Glycoside"
                if class_info["isglycoside"]
                else "Non-glycoside",
            }
        )

    class_df = pd.DataFrame(classifications)
    class_df.to_csv(
        args.output_dir
        + "/"
        + args.splits_file.split("/")[-1].replace(
            ".tsv", "_np_classes_test_only.tsv"
        ),
        index=False,
        sep="\t",
    )
