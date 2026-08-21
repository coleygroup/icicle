#!/usr/bin/env bash
# One-time: build the formula-candidate pickle for the random_no_qcxms2 split
# (only a scaffold one exists on disk already). Required before
# run_icicle_formula_retrieval.sh's random-split runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

uv run examples/scripts/data_processing/retrieval_candidate_set/retrieval_create_retrieval_lists.py \
  --input-map data/NIST2023_GCMS_main/retrieval/pubchem_formula_map_subset.p \
  --labels-file data/NIST2023_GCMS_main/metadata.tsv \
  --split-file data/NIST2023_GCMS_main/splits/random_no_xeno_aas_deduplicated_no_qcxms2.tsv \
  --output-dir data/NIST2023_GCMS_main/retrieval/ \
  --max-k 50 --workers 16

echo "Done. Produced: data/NIST2023_GCMS_main/retrieval/cands_pickled_random_no_xeno_aas_deduplicated_no_qcxms2_50.pkl"
