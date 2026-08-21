#!/usr/bin/env bash
# RI-windowed retrieval ladder (1000/100000/1000000/10000000 candidates per
# query) + the ±80 Da highest-observed-peak MW heuristic, for ICICLE, NEIMS,
# and MassFormer. SEPARATE from run_pubchem_global_retrieval_all_models.sh
# (which only runs the "all" / true-global-rank level — the priority result).
#
# Run this only after run_pubchem_global_retrieval_all_models.sh has
# completed the "all" level for all three models. This ladder is much more
# expensive: on a real run, the union of per-query RI windows blew up fast
# (top100000 already covered 45.6% of the ~86M-row database) and the
# windowed-scan code path is CPU-bound and slower per-row than a plain
# sequential scan — expect this to take from several hours to multiple
# days depending on RI type and level. Safe to interrupt (Ctrl-C / kill)
# and resume later: each stage/level is idempotent, and the true-predicted-
# spectra cache built during the "all" run is reused here.
#
# Usage:
#   screen -S pubchem_ri_ladder
#   cd /home/magled/icicle-dev
#   bash examples/scripts/evaluation/run_pubchem_ri_window_ladder.sh
#   # Ctrl-A D to detach, tail the log files below to monitor.

set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

ICICLE_HDF5="results/inference/pubchem_predictions_rerun_260710.hdf5"
NEIMS_HDF5="baselines/neims/results/pubchem_predictions/neims_random_s1_pubchem_full.hdf5"
MASSFORMER_COLUMNAR="baselines/massformer/results/predictions/massformer_random_s2_pubchem_full_columnar.hdf5"
MW_TSV="data/PubChem/PubChem_filtered.tsv"

RI_WINDOW_LEVELS="[1000,100000,1000000,10000000,all]"
MW_WINDOW_DA=80

ICICLE_OUT="results/pubchem_retrieval_eval_icicle_rerun_260710"
NEIMS_OUT="results/pubchem_retrieval_eval_neims"
MASSFORMER_OUT="results/pubchem_retrieval_eval_massformer"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

has_dataset() {
    # has_dataset <hdf5_path> <dataset_name>
    uv run python -c "
import h5py, sys
with h5py.File('$1', 'r') as f:
    sys.exit(0 if '$2' in f else 1)
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Stage 1: add MW sort index (needed for the ±80 Da heuristic), all 3 models
# ---------------------------------------------------------------------------
log "=== Stage 1/4: ICICLE — add MW sort index ==="
if has_dataset "$ICICLE_HDF5" sort_idx_mw; then
    log "Skipping — sort_idx_mw already present in $ICICLE_HDF5"
else
    uv run examples/scripts/evaluation/add_mw_to_hdf5.py \
        --hdf5 "$ICICLE_HDF5" \
        --tsv "$MW_TSV" \
        2>&1 | tee "$LOG_DIR/8_icicle_mw_index.log"
fi

log "=== Stage 1/4: NEIMS — add MW sort index ==="
if has_dataset "$NEIMS_HDF5" sort_idx_mw; then
    log "Skipping — sort_idx_mw already present in $NEIMS_HDF5"
else
    uv run examples/scripts/evaluation/add_mw_to_hdf5.py \
        --hdf5 "$NEIMS_HDF5" \
        --tsv "$MW_TSV" \
        2>&1 | tee "$LOG_DIR/8_neims_mw_index.log"
fi

log "=== Stage 1/4: MassFormer — add MW sort index ==="
if has_dataset "$MASSFORMER_COLUMNAR" sort_idx_mw; then
    log "Skipping — sort_idx_mw already present in $MASSFORMER_COLUMNAR"
else
    uv run examples/scripts/evaluation/add_mw_to_hdf5.py \
        --hdf5 "$MASSFORMER_COLUMNAR" \
        --tsv "$MW_TSV" \
        2>&1 | tee "$LOG_DIR/8_massformer_mw_index.log"
fi

# ---------------------------------------------------------------------------
# Stage 2-4: RI-window ladder + MW heuristic — ICICLE, NEIMS, MassFormer
# Reuses each model's true_pred_spectra_cache.npz built by the "all" run.
# ---------------------------------------------------------------------------
log "=== Stage 2/4: RI-window ladder + MW ±${MW_WINDOW_DA}Da — ICICLE ==="
if [ -f "$ICICLE_OUT/retrieval_ablation_all_ri_types.json" ]; then
    log "Skipping — $ICICLE_OUT already has RI-ladder results"
else
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        ri_types=[StdNP,SemiStdNP,StdPolar] \
        top_n_levels="$RI_WINDOW_LEVELS" \
        mw_window_da="$MW_WINDOW_DA" \
        2>&1 | tee "$LOG_DIR/9_icicle_ri_ladder.log"
fi

log "=== Stage 3/4: RI-window ladder + MW ±${MW_WINDOW_DA}Da — NEIMS ==="
if [ -f "$NEIMS_OUT/retrieval_ablation_all_ri_types.json" ]; then
    log "Skipping — $NEIMS_OUT already has RI-ladder results"
else
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$NEIMS_HDF5]" \
        output_dir="$NEIMS_OUT" \
        spectra_cache_dir="$NEIMS_OUT" \
        skipped_log="$NEIMS_OUT/skipped_queries.tsv" \
        ri_types=[StdNP,SemiStdNP,StdPolar] \
        top_n_levels="$RI_WINDOW_LEVELS" \
        mw_window_da="$MW_WINDOW_DA" \
        2>&1 | tee "$LOG_DIR/10_neims_ri_ladder.log"
fi

log "=== Stage 4/4: RI-window ladder + MW ±${MW_WINDOW_DA}Da — MassFormer ==="
if [ -f "$MASSFORMER_OUT/retrieval_ablation_all_ri_types.json" ]; then
    log "Skipping — $MASSFORMER_OUT already has RI-ladder results"
else
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$MASSFORMER_COLUMNAR]" \
        output_dir="$MASSFORMER_OUT" \
        spectra_cache_dir="$MASSFORMER_OUT" \
        skipped_log="$MASSFORMER_OUT/skipped_queries.tsv" \
        ri_types=[StdNP,SemiStdNP,StdPolar] \
        top_n_levels="$RI_WINDOW_LEVELS" \
        mw_window_da="$MW_WINDOW_DA" \
        2>&1 | tee "$LOG_DIR/11_massformer_ri_ladder.log"
fi

log "=== RI-window ladder + MW heuristic complete ==="
log "Results per model:"
log "  <output_dir>/retrieval_ablation_all_ri_types.json   — RI-window summary"
log "  <output_dir>/retrieval_mw${MW_WINDOW_DA}_results.json          — MW-only summary"
log "  <output_dir>/retrieval_union_mw${MW_WINDOW_DA}_results.json    — RI∪MW union summary"
log "  <output_dir>/retrieval_per_query_*.tsv               — per-query rank detail"
