#!/usr/bin/env bash
# Produces: retrieval_per_query_union_mw{80,10,5}_StdNP_N1000.tsv -- per-query
# candidate-pool-size dumps for the RI-union-MW retrieval track, needed for
# the RI-union-MW row of the recall-vs-pool-size comparison.
# rerun_union_icicle_stdnp.sh computes the same union tracks but at the
# default N=1000 top_n_level only among others; this script exists
# specifically to guarantee the N=1000 per-query TSV dump is (re)written even
# if a prior run's per-query TSV write step was skipped.
#
# Usage: bash examples/scripts/evaluation/paper_reruns/rerun_mw_union_perquery.sh
set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

ICICLE_HDF5="results/inference/pubchem_predictions_rerun_260710.hdf5"
ICICLE_OUT="results/pubchem_retrieval_eval_icicle_rerun_260710"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
    local mw_da=$1
    log "=== MW union per-query dump, StdNP, N=1000, +-${mw_da}Da ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        ri_types=[StdNP] \
        top_n_levels=[1000] \
        mw_window_da="$mw_da" \
        mw_global=false \
        skip_ri_ladder=true \
        2>&1 | tee "$LOG_DIR/mw_union_perquery_${mw_da}.log"
}

run 80
run 10
run 5

log "=== DONE ==="
