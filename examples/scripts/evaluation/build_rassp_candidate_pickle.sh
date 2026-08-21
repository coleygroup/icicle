#!/usr/bin/env bash
# One-time: build the formula-candidate pickle for RASSP's native scaffold
# split (22,954 test mols). Required before run_rassp_headtohead.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

uv run examples/scripts/data_processing/retrieval_candidate_set/retrieval_create_retrieval_lists.py \
  --input-map data/NIST2023_GCMS_main/retrieval/pubchem_formula_map_subset.p \
  --labels-file data/NIST2023_GCMS_main/metadata.tsv \
  --split-file data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated_rassp.tsv \
  --output-dir data/NIST2023_GCMS_main/retrieval/ \
  --max-k 50 --workers 16

echo "Done. Produced: data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_no_xeno_aas_deduplicated_rassp_50.pkl"
