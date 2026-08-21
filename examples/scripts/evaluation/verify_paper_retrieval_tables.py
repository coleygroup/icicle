#!/usr/bin/env python
"""Regenerate every retrieval and naive-baseline-similarity number quoted in
main.tex / supplementary.tex from the underlying result files, and print them
for direct diffing against the tex source.

Run: uv run examples/scripts/evaluation/verify_paper_retrieval_tables.py
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"
EVAL = RESULTS / "eval"


def fmt(x: float) -> str:
    return f"{x:.3f}"


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def global_retrieval_table() -> None:
    print_header("Table: Global PubChem retrieval (no filter), autofail")
    dirs = {
        "ICICLE": RESULTS / "pubchem_retrieval_eval_icicle_rerun_260710",
        "NEIMS": RESULTS / "pubchem_retrieval_eval_neims",
        "MassFormer": RESULTS / "pubchem_retrieval_eval_massformer",
    }
    for model, d in dirs.items():
        with open(d / "retrieval_global_results.json") as f:
            data = json.load(f)
        s = data["all"]["autofail"]["cosine"]
        print(
            f"{model:>12}: top1={fmt(s['top_1_accuracy'])} top5={fmt(s['top_5_accuracy'])} "
            f"top10={fmt(s['top_10_accuracy'])} top20={fmt(s['top_20_accuracy'])} "
            f"top50={fmt(s['top_50_accuracy'])} mrr={fmt(s['mrr'])}"
        )


def ri_ablation_table() -> None:
    print_header(
        "Table: RI-filtered PubChem retrieval (ICICLE), all 3 RI types, autofail"
    )
    d = RESULTS / "pubchem_retrieval_eval_icicle_rerun_260710"
    for ri_type in ["StdNP", "SemiStdNP", "StdPolar"]:
        with open(d / f"retrieval_ablation_{ri_type}.json") as f:
            data = json.load(f)
        print(f"--- {ri_type} ---")
        for level in ["1000", "100000", "1000000", "10000000", "all"]:
            s = data[level]["autofail"]["cosine"]
            print(
                f"  {level:>10}: top1={fmt(s['top_1_accuracy'])} "
                f"top10={fmt(s['top_10_accuracy'])} mrr={fmt(s['mrr'])}"
            )


def heavy_atom_table() -> None:
    print_header(
        "Table: Heavy-atom-count-filtered retrieval (ICICLE), autofail"
    )
    d = RESULTS / "pubchem_retrieval_eval_icicle_rerun_260710"
    for w in [1, 2, 3, 6, 8]:
        with open(d / f"retrieval_heavy_atom{w}_global_results.json") as f:
            data = json.load(f)
        s = data["all"]["autofail"]["cosine"]
        print(
            f"  w=±{w}: top1={fmt(s['top_1_accuracy'])} top10={fmt(s['top_10_accuracy'])} mrr={fmt(s['mrr'])}"
        )


def mw_only_table() -> None:
    print_header("Table: MW-only-filtered retrieval, all models, autofail")
    dirs = {
        "ICICLE": RESULTS / "pubchem_retrieval_eval_icicle_rerun_260710",
        "NEIMS": RESULTS / "pubchem_retrieval_eval_neims",
        "MassFormer": RESULTS / "pubchem_retrieval_eval_massformer",
    }
    widths = [
        ("80", "±80Da"),
        ("10_80", "[-10,+80]"),
        ("10_10", "[-10,+10]"),
        ("5", "±5Da"),
    ]
    for model, d in dirs.items():
        print(f"--- {model} ---")
        for tag, label in widths:
            path = d / f"retrieval_mw{tag}_global_results.json"
            if not path.exists():
                print(f"  {label:>12}: MISSING")
                continue
            with open(path) as f:
                data = json.load(f)
            s = data["all"]["autofail"]["cosine"]
            print(
                f"  {label:>12}: top1={fmt(s['top_1_accuracy'])} top10={fmt(s['top_10_accuracy'])} mrr={fmt(s['mrr'])}"
            )


def union_table() -> None:
    print_header(
        "Table: RI∪MW-filtered retrieval (StdNP, N=1000), all models, autofail"
    )
    dirs = {
        "ICICLE": RESULTS / "pubchem_retrieval_eval_icicle_rerun_260710",
        "NEIMS": RESULTS / "pubchem_retrieval_eval_neims",
        "MassFormer": RESULTS / "pubchem_retrieval_eval_massformer",
    }
    widths = [
        ("80", "±80Da"),
        ("10_80", "[-10,+80]"),
        ("10_10", "[-10,+10]"),
        ("5", "±5Da"),
    ]
    for model, d in dirs.items():
        print(f"--- {model} ---")
        for tag, label in widths:
            path = d / f"retrieval_union_mw{tag}_results.json"
            if not path.exists():
                print(f"  {label:>12}: MISSING FILE")
                continue
            with open(path) as f:
                data = json.load(f)
            ri = data.get("StdNP")
            if ri is None:
                print(
                    f"  {label:>12}: StdNP missing (only {list(data.keys())} present -- likely overwritten)"
                )
                continue
            s = ri["1000"]["autofail"]["cosine"]
            print(
                f"  {label:>12}: top1={fmt(s['top_1_accuracy'])} top10={fmt(s['top_10_accuracy'])} mrr={fmt(s['mrr'])}"
            )


def global_retrieval_table_scaffold() -> None:
    print_header(
        "Table: Global PubChem retrieval (no filter), scaffold split, autofail "
        "(tab:si-pubchem-global-nofilter-scaffold)"
    )
    dirs = {
        "ICICLE": RESULTS / "pubchem_retrieval_eval_icicle_scaffold_s1",
        "NEIMS": RESULTS / "pubchem_retrieval_eval_neims_scaffold_s1",
        "MassFormer": RESULTS
        / "pubchem_retrieval_eval_massformer_scaffold_s1",
    }
    for model, d in dirs.items():
        path = d / "retrieval_global_results.json"
        if not path.exists():
            print(f"{model:>12}: MISSING (run not yet completed)")
            continue
        with open(path) as f:
            data = json.load(f)
        s = data["all"]["autofail"]["cosine"]
        print(
            f"{model:>12}: top1={fmt(s['top_1_accuracy'])} top5={fmt(s['top_5_accuracy'])} "
            f"top10={fmt(s['top_10_accuracy'])} top20={fmt(s['top_20_accuracy'])} "
            f"top50={fmt(s['top_50_accuracy'])} mrr={fmt(s['mrr'])} "
            f"median_rank={s['median_rank']:,.0f}"
        )


def mw_ablation_table_scaffold() -> None:
    print_header(
        "Table: MW-filtered global PubChem retrieval, scaffold split, autofail "
        "(tab:si-mw-scaffold-ablation -- ICICLE only in the current paper text; "
        "NEIMS/MassFormer printed here too for when/if they're added)"
    )
    dirs = {
        "ICICLE": RESULTS / "pubchem_retrieval_eval_icicle_scaffold_s1",
        "NEIMS": RESULTS / "pubchem_retrieval_eval_neims_scaffold_s1",
        "MassFormer": RESULTS
        / "pubchem_retrieval_eval_massformer_scaffold_s1",
    }
    windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 80]
    for model, d in dirs.items():
        print(f"--- {model} ---")
        for w in windows:
            path = d / f"retrieval_mw{w}_global_results.json"
            if not path.exists():
                print(f"  +-{w:>3}Da: MISSING")
                continue
            with open(path) as f:
                data = json.load(f)
            s = data["all"]["autofail"]["cosine"]
            print(
                f"  +-{w:>3}Da: top1={fmt(s['top_1_accuracy'])} "
                f"top10={fmt(s['top_10_accuracy'])} mrr={fmt(s['mrr'])} "
                f"median_rank={s['median_rank']:,.0f}"
            )


def heavy_atom_table_scaffold() -> None:
    print_header(
        "Table: Heavy-atom-count-filtered retrieval, scaffold split, autofail "
        "(tab:heavy-atom-retrieval-scaffold -- ICICLE + NEIMS in the current "
        "paper text; MassFormer printed here too for when/if it's added). "
        "Uses the scaffold-split heavy-atom predictor "
        "(checkpoints/heavy_atom_predictor_scaffold_split.joblib), NOT the "
        "random-split one, to avoid scaffold leakage."
    )
    dirs = {
        "ICICLE": RESULTS / "pubchem_retrieval_eval_icicle_scaffold_s1",
        "NEIMS": RESULTS / "pubchem_retrieval_eval_neims_scaffold_s1",
        "MassFormer": RESULTS
        / "pubchem_retrieval_eval_massformer_scaffold_s1",
    }
    for model, d in dirs.items():
        print(f"--- {model} ---")
        for w in [1, 2, 3, 6, 8]:
            path = d / f"retrieval_heavy_atom{w}_global_results.json"
            if not path.exists():
                print(f"  w=+-{w}: MISSING")
                continue
            with open(path) as f:
                data = json.load(f)
            s = data["all"]["autofail"]["cosine"]
            print(
                f"  w=+-{w}: top1={fmt(s['top_1_accuracy'])} "
                f"top10={fmt(s['top_10_accuracy'])} mrr={fmt(s['mrr'])}"
            )


def median_candidate_pool_sizes() -> None:
    print_header("Real per-query candidate pool sizes (median), ICICLE")
    d = RESULTS / "pubchem_retrieval_eval_icicle_rerun_260710"
    for level in ["1000", "100000", "1000000", "10000000"]:
        df = pd.read_csv(
            d / f"retrieval_per_query_StdNP_{level}.tsv", sep="\t"
        )
        print(
            f"  RI n={level:>10}: median n_candidates={df['n_candidates'].median():,.0f}"
        )
    for tag, label in [
        ("80", "±80Da"),
        ("10_80", "[-10,+80]"),
        ("10_10", "[-10,+10]"),
        ("5", "±5Da"),
    ]:
        path = d / f"retrieval_per_query_mw{tag}_all.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        print(
            f"  MW {label:>12}: median n_candidates={df['n_candidates'].median():,.0f}"
        )


def formula_match_table(subset_label: str, include_rassp: bool) -> None:
    def load_icicle_top1(path):
        raw = pd.read_csv(path)
        t = raw[~raw["is_decoy"]].rename(columns={"spec": "mol_id"})
        t["top1"] = t["rank_cosine_similarity"] == 1
        return t[["mol_id", "top1"]]

    def load_neims_style_top1(path):
        raw = pd.read_csv(path)
        t = raw[raw["is_correct"]].rename(
            columns={"query_inchikey14": "inchikey14"}
        )
        t["top1"] = t["rank_cosine_similarity"] == 1
        return t.groupby("inchikey14", as_index=False)["top1"].max()

    metadata = pd.read_csv(
        REPO_ROOT / "data" / "NIST2023_GCMS_main" / "metadata.tsv",
        sep="\t",
        usecols=["mol_id", "inchi_key"],
    )
    metadata["inchikey14"] = metadata["inchi_key"].astype(str).str[:14]

    def seed_rate(loader, paths, key_col):
        seeds = []
        for p in paths:
            df = loader(p)
            if key_col == "inchikey14" and "mol_id" in df.columns:
                df = df.merge(
                    metadata[["mol_id", "inchikey14"]], on="mol_id", how="left"
                )
            seeds.append(df.set_index(key_col)["top1"])
        wide = pd.concat(
            seeds, axis=1, keys=[f"s{i}" for i in range(1, len(seeds) + 1)]
        )
        return wide.dropna().mean(axis=1)

    print_header(f"Table: Formula-match retrieval top-1, {subset_label}")
    for split in ["random", "scaffold"]:
        icicle = seed_rate(
            load_icicle_top1,
            [
                EVAL
                / f"final_entropy_{split}_s{i}_retr"
                / "retrieval_with_formula_results.csv"
                for i in (1, 2, 3)
            ],
            "inchikey14",
        )
        neims = seed_rate(
            load_neims_style_top1,
            [
                EVAL
                / f"neims_{split}_s{i}"
                / "retrieval_with_formula_results.csv"
                for i in (1, 2, 3)
            ],
            "inchikey14",
        )
        mf_seeds = [1, 3] if split == "random" else [1, 2, 3]
        massformer = seed_rate(
            load_neims_style_top1,
            [
                EVAL
                / f"massformer_{split}_s{i}"
                / "retrieval_with_formula_results.csv"
                for i in mf_seeds
            ],
            "inchikey14",
        )
        models = {"ICICLE": icicle, "NEIMS": neims, "MassFormer": massformer}
        if include_rassp:
            rassp = seed_rate(
                load_neims_style_top1,
                [
                    EVAL
                    / f"rassp_{split}_s{i}"
                    / "retrieval_with_formula_results.csv"
                    for i in (1, 2, 3)
                ],
                "inchikey14",
            )
            models["RASSP"] = rassp
        common = set.intersection(*(set(s.index) for s in models.values()))
        print(f"--- {split} split (n={len(common)}) ---")
        for name, series in models.items():
            vals = series[series.index.isin(common)]
            print(f"  {name:>12}: top1={fmt(vals.mean())} (n={len(vals)})")


def naive_baseline_table() -> None:
    print_header(
        "Table: Naive baseline similarity (random / average / full-enumeration-barcode)"
    )
    cols = [
        "cosine_similarity",
        "entropy_similarity",
        "weighted_cosine_nist_gc",
        "composite_similarity_nist_gc",
    ]
    runs = [
        ("random_random_sim", "Random baseline, random split, full test"),
        ("average_random_sim", "Average baseline, random split, full test"),
        (
            "full_enumeration_barcode_random_sim",
            "Full-enumeration-barcode, random split, full test",
        ),
        ("random_scaffold_sim", "Random baseline, scaffold split, full test"),
        (
            "average_scaffold_sim",
            "Average baseline, scaffold split, full test",
        ),
        (
            "full_enumeration_barcode_scaffold_sim",
            "Full-enumeration-barcode, scaffold split, full test",
        ),
        (
            "random_random_rassp_subset_sim",
            "Random baseline, random split, RASSP subset",
        ),
        (
            "average_random_rassp_subset_sim",
            "Average baseline, random split, RASSP subset",
        ),
        (
            "full_enumeration_barcode_random_rassp_subset_sim",
            "Full-enumeration-barcode, random split, RASSP subset",
        ),
        (
            "random_scaffold_rassp_subset_sim",
            "Random baseline, scaffold split, RASSP subset",
        ),
        (
            "average_scaffold_rassp_subset_sim",
            "Average baseline, scaffold split, RASSP subset",
        ),
        (
            "full_enumeration_barcode_scaffold_rassp_subset_sim",
            "Full-enumeration-barcode, scaffold split, RASSP subset",
        ),
    ]
    for run_dir, label in runs:
        summary_path = EVAL / run_dir / "similarity_summary_metrics.txt"
        csv_path = EVAL / run_dir / "similarity_results.csv"
        if summary_path.exists():
            values = {}
            for line in summary_path.read_text().splitlines():
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                values[key.strip()] = val.strip()
            print(
                f"{label:>55}: n={values.get('n_molecules_total')} "
                f"cos={float(values.get('avg_cosine_similarity', 'nan')):.3f} "
                f"entr={float(values.get('avg_entropy_similarity', 'nan')):.3f} "
                f"wcs={float(values.get('avg_weighted_cosine_nist_gc', 'nan')):.3f} "
                f"ccs={float(values.get('avg_composite_similarity_nist_gc', 'nan')):.3f}"
            )
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            means = df[cols].mean()
            print(
                f"{label:>55}: n={len(df)} [recomputed from CSV, summary file missing] "
                f"cos={means['cosine_similarity']:.3f} entr={means['entropy_similarity']:.3f} "
                f"wcs={means['weighted_cosine_nist_gc']:.3f} ccs={means['composite_similarity_nist_gc']:.3f}"
            )
        else:
            print(f"{label:>55}: MISSING (run not yet completed)")


if __name__ == "__main__":
    global_retrieval_table()
    ri_ablation_table()
    heavy_atom_table()
    mw_only_table()
    union_table()
    median_candidate_pool_sizes()
    global_retrieval_table_scaffold()
    mw_ablation_table_scaffold()
    heavy_atom_table_scaffold()
    formula_match_table(
        "RASSP-restricted subset (main text Table)", include_rassp=True
    )
    formula_match_table(
        "ICICLE+NEIMS+MassFormer only, no RASSP (SI Table)",
        include_rassp=False,
    )
    naive_baseline_table()
