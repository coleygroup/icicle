#!/usr/bin/env bash
# Produces: SI Table tab:si-recall-vs-pool, tab:heavy-atom-retrieval (StdNP-subset rows).
#
# MW-alone and heavy-atom-alone retrieval (ICICLE, N=1000 candidates), restricted
# to the StdNP RI subset (n=1,168 queries with a StdNP RI value) so these rows
# are directly comparable to the RI-ladder's N=1000 row on the same query set.
# MW widths: +-5/10/80 Da. Heavy-atom windows: +-1/2/3/6/8 atoms.
#
# Usage: bash examples/scripts/evaluation/paper_reruns/rerun_mw_ha_stdnp_subset.sh
set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

ICICLE_HDF5="results/inference/pubchem_predictions_rerun_260710.hdf5"
ICICLE_OUT="results/pubchem_retrieval_eval_icicle_rerun_260710"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_mw() {
    local mw_da=$1
    log "=== MW-alone, StdNP subset, +-${mw_da}Da ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        ri_types=[StdNP] \
        top_n_levels=[1000] \
        mw_window_da="$mw_da" mw_global=false \
        skip_ri_ladder=true \
        2>&1 | tee "$LOG_DIR/mw_stdnp_subset_${mw_da}.log"
}

run_ha() {
    local window=$1
    log "=== Heavy-atom-alone, StdNP subset, +-${window} atoms ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        ri_types=[StdNP] \
        top_n_levels=[1000] \
        mw_window_da=null \
        heavy_atom_window="$window" heavy_atom_global=false \
        skip_ri_ladder=true \
        2>&1 | tee "$LOG_DIR/ha_stdnp_subset_w${window}.log"
}

run_mw 80
run_mw 10
run_mw 5

run_ha 1
run_ha 2
run_ha 3
run_ha 6
run_ha 8

log "=== DONE ==="
