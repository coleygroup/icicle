#!/usr/bin/env bash
# Produces: SI Table tab:si-mw-union, tab:si-ri-vs-union-mw-stdnp (ICICLE, StdNP).
#
# RI-union-heavy-atom and RI-union-MW retrieval (candidate passes if it's in
# EITHER window, not the funnel/intersection variant) across the RI-window
# ladder (N=1000..all), StdNP RI type, ICICLE only. Heavy-atom windows
# +-1/2/3/6/8 atoms; MW widths +-5/10/80 Da.
#
# Usage: bash examples/scripts/evaluation/paper_reruns/rerun_union_icicle_stdnp.sh
set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

ICICLE_HDF5="results/inference/pubchem_predictions_rerun_260710.hdf5"
ICICLE_OUT="results/pubchem_retrieval_eval_icicle_rerun_260710"
RI_WINDOW_LEVELS="[1000,100000,1000000,10000000,all]"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_ha_union() {
    local window=$1
    log "=== HA union, StdNP, +-${window} atoms ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        ri_types=[StdNP] \
        top_n_levels="$RI_WINDOW_LEVELS" \
        mw_window_da=null \
        heavy_atom_window="$window" heavy_atom_ri_union=true \
        heavy_atom_global=false \
        skip_ri_ladder=true \
        2>&1 | tee "$LOG_DIR/union_ha_stdnp_w${window}.log"
}

run_mw_union() {
    local window=$1
    log "=== MW union, StdNP, +-${window}Da ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        ri_types=[StdNP] \
        top_n_levels="$RI_WINDOW_LEVELS" \
        mw_window_da="$window" mw_global=false \
        skip_ri_ladder=true \
        2>&1 | tee "$LOG_DIR/union_mw_stdnp_${window}.log"
}

run_ha_union 1
run_ha_union 2
run_ha_union 3
run_ha_union 6
run_ha_union 8

run_mw_union 5
run_mw_union 10
run_mw_union 80

log "=== DONE ==="
