#!/usr/bin/env bash
# Regenerate results/eval/rassp_subset_comparison/*_rassp_subset.csv from the
# full scaffold/random eval CSVs, filtered down to RASSP's native test split.
# Run this after any new ICICLE/NEIMS/MassFormer scaffold/random eval seed lands.
set -euo pipefail

RESULTS=results/eval
OUT_DIR=$RESULTS/rassp_subset_comparison
mkdir -p "$OUT_DIR"

FILTER=examples/scripts/evaluation/filter_to_rassp_subset.py

run_filter() {
  local input_csv=$1
  shift
  if [ ! -f "$input_csv" ]; then
    echo "skip (missing input): $input_csv"
    return 0
  fi
  uv run "$FILTER" --input-csv "$input_csv" "$@"
}

for split_name in scaffold random; do
  if [ "$split_name" = "scaffold" ]; then
    SPLIT_TSV=data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated_rassp.tsv
    ICICLE_PREFIX=final_entropy_scaffold
  else
    SPLIT_TSV=data/NIST2023_GCMS_main/splits/random_no_xeno_aas_deduplicated_no_qcxms2_rassp.tsv
    ICICLE_PREFIX=final_entropy_random
  fi

  for seed in 1 2 3; do
    # Similarity CSVs
    run_filter "$RESULTS/${ICICLE_PREFIX}_s${seed}_sim/similarity_results.csv" \
      --rassp-split-tsv "$SPLIT_TSV" \
      --ik14-col inchi_key \
      --mol-id-to-inchikey "$SPLIT_TSV" \
      --output-csv "$OUT_DIR/icicle_${split_name}_s${seed}_sim_rassp_subset.csv"

    run_filter "$RESULTS/neims_${split_name}_s${seed}/similarity_results.csv" \
      --rassp-split-tsv "$SPLIT_TSV" \
      --ik14-col inchi_key \
      --mol-id-to-inchikey "$SPLIT_TSV" \
      --output-csv "$OUT_DIR/neims_${split_name}_s${seed}_sim_rassp_subset.csv"

    run_filter "$RESULTS/massformer_${split_name}_s${seed}/similarity_results.csv" \
      --rassp-split-tsv "$SPLIT_TSV" \
      --ik14-col inchi_key \
      --mol-id-to-inchikey "$SPLIT_TSV" \
      --output-csv "$OUT_DIR/massformer_${split_name}_s${seed}_sim_rassp_subset.csv"

    # Retrieval (formula-match) CSVs
    run_filter "$RESULTS/${ICICLE_PREFIX}_s${seed}_retr/retrieval_with_formula_results.csv" \
      --rassp-split-tsv "$SPLIT_TSV" \
      --ik14-col inchikey \
      --group-col spec \
      --output-csv "$OUT_DIR/icicle_${split_name}_s${seed}_retr_rassp_subset.csv"

    run_filter "$RESULTS/neims_${split_name}_s${seed}/retrieval_with_formula_results.csv" \
      --rassp-split-tsv "$SPLIT_TSV" \
      --ik14-col query_inchikey14 \
      --output-csv "$OUT_DIR/neims_${split_name}_s${seed}_retr_rassp_subset.csv"

    run_filter "$RESULTS/massformer_${split_name}_s${seed}/retrieval_with_formula_results.csv" \
      --rassp-split-tsv "$SPLIT_TSV" \
      --ik14-col query_inchikey14 \
      --output-csv "$OUT_DIR/massformer_${split_name}_s${seed}_retr_rassp_subset.csv"
  done
done
