#!/usr/bin/env bash
# Produces: SI Table tab:si-mw-only (NEIMS/MassFormer narrow-MW-width rows).
#
# RI-union-MW retrieval for NEIMS and MassFormer at the narrow MW widths
# already computed for ICICLE (+-5Da, [-10,+10]Da), so the union ladder is
# comparable across all three models. mw_global=true so each run also
# produces the MW-global track as a side effect.
#
# Usage: bash examples/scripts/evaluation/paper_reruns/rerun_union_neims_massformer_narrow.sh
set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

NEIMS_HDF5="baselines/neims/results/pubchem_predictions/neims_random_s1_pubchem_full.hdf5"
MASSFORMER_COLUMNAR="baselines/massformer/results/predictions/massformer_random_s2_pubchem_full_columnar.hdf5"
NEIMS_OUT="results/pubchem_retrieval_eval_neims"
MASSFORMER_OUT="results/pubchem_retrieval_eval_massformer"

RI_WINDOW_LEVELS="[1000,100000,1000000,10000000,all]"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
    local hdf5=$1 out_dir=$2 mw_args=$3 logfile=$4
    log "=== $logfile ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$hdf5]" \
        output_dir="$out_dir" \
        spectra_cache_dir="$out_dir" \
        skipped_log="$out_dir/skipped_queries.tsv" \
        ri_types=[StdNP] \
        top_n_levels="$RI_WINDOW_LEVELS" \
        $mw_args \
        mw_global=true \
        2>&1 | tee "$LOG_DIR/$logfile"
}

run "$NEIMS_HDF5" "$NEIMS_OUT" "mw_window_da=5" "union_narrow_neims_mw5.log"
run "$NEIMS_HDF5" "$NEIMS_OUT" "mw_window_da=10 mw_window_da_lo=10 mw_window_da_hi=10" "union_narrow_neims_mw10_10.log"
run "$MASSFORMER_COLUMNAR" "$MASSFORMER_OUT" "mw_window_da=5" "union_narrow_massformer_mw5.log"
run "$MASSFORMER_COLUMNAR" "$MASSFORMER_OUT" "mw_window_da=10 mw_window_da_lo=10 mw_window_da_hi=10" "union_narrow_massformer_mw10_10.log"

log "=== DONE ==="
