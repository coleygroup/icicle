"""Filter existing eval CSVs down to the RASSP-native split's test molecules.

RASSP's own split TSV is a strict subset of the full split (same model
predictions either way), so for a head-to-head plot we downsample the already-
completed full eval outputs to RASSP's *test* rows instead of rerunning eval.
"""

import argparse

import pandas as pd


def rassp_inchikey14s(rassp_split_tsv: str) -> set[str]:
    split = pd.read_csv(rassp_split_tsv, sep="\t")
    test_split = split[split["split"] == "test"]
    return set(test_split["inchi_key"].str[:14])


def filter_csv(
    csv_path: str, ik14_col: str, keep_ik14: set[str], output_csv: str
) -> None:
    df = pd.read_csv(csv_path)
    ik14 = df[ik14_col].str[:14]
    filtered = df[ik14.isin(keep_ik14)]
    print(f"{output_csv}: {len(filtered)} / {len(df)} rows kept")
    filtered.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--rassp-split-tsv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--ik14-col",
        required=True,
        help="Column holding a (14+-char) InChIKey in --input-csv, e.g. "
        "'inchikey' (similarity) or 'query_inchikey14' (retrieval).",
    )
    parser.add_argument(
        "--mol-id-to-inchikey",
        help="For ICICLE similarity CSVs keyed by mol_id instead of inchikey: "
        "path to the full split TSV (mol_id, inchi_key columns) to join on.",
    )
    parser.add_argument(
        "--group-col",
        help="For ICICLE retrieval CSVs: group rows by this column (e.g. 'spec') "
        "and keep/drop the whole group based on the group's true inchikey "
        "(the row where --ik14-col is non-null).",
    )
    args = parser.parse_args()

    keep_ik14 = rassp_inchikey14s(args.rassp_split_tsv)
    df = pd.read_csv(args.input_csv)

    if args.group_col:
        true_ik14_by_group = (
            df.dropna(subset=[args.ik14_col])
            .set_index(args.group_col)[args.ik14_col]
            .str[:14]
        )
        keep_groups = set(
            true_ik14_by_group[true_ik14_by_group.isin(keep_ik14)].index
        )
        keep_mask = df[args.group_col].isin(keep_groups)
    elif args.mol_id_to_inchikey:
        mol_map = pd.read_csv(args.mol_id_to_inchikey, sep="\t")[
            ["mol_id", "inchi_key"]
        ]
        df = df.merge(mol_map, on="mol_id", how="left")
        keep_mask = df["inchi_key"].str[:14].isin(keep_ik14)
    else:
        keep_mask = df[args.ik14_col].str[:14].isin(keep_ik14)

    filtered = df[keep_mask]
    print(
        f"{args.output_csv}: {len(filtered)} / {len(df)} rows kept ({len(keep_ik14)} RASSP inchikey14s)"
    )
    filtered.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
