#!/usr/bin/env bash
# Runs the full PubChem global-retrieval pipeline for ICICLE, NEIMS, and
# MassFormer, one stage after another. Meant to be launched inside a
# screen/tmux session and left to run unattended.
#
# Each stage is idempotent (checks for its expected output before doing
# work) so this script is safe to re-run after an interruption — it will
# skip already-completed stages and resume from where it left off.
#
# Usage:
#   screen -S pubchem_retrieval_all
#   cd /home/magled/icicle-dev
#   bash examples/scripts/evaluation/run_pubchem_global_retrieval_all_models.sh
#   # Ctrl-A D to detach, tail the log files below to monitor.

set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

ICICLE_HDF5="results/inference/pubchem_predictions_rerun_260710.hdf5"
NEIMS_HDF5="baselines/neims/results/pubchem_predictions/neims_random_s1_pubchem_full.hdf5"
MASSFORMER_COLUMNAR="baselines/massformer/results/predictions/massformer_random_s2_pubchem_full_columnar.hdf5"
RI_TSV="data/PubChem/AIRI_inference_output_full.tsv"

# "all" = true global rank, one straight sequential scan, no per-query
# windowing overhead — this is the priority result and runs first, on its
# own, for every model.
#
# RI-windowed levels (1000/100000/1000000/10000000) are a SEPARATE, later
# pass (see run_pubchem_ri_window_ladder.sh) — NOT included here. Observed
# behavior on a real run: with ~1,166 queries scattered across ~86M sorted
# RI rows, the union of per-query windows blows up fast (top100000 already
# unioned to 45.6% of the database) and the windowed scan path is CPU-bound
# and much slower per-row than the plain sequential "all" scan. Running
# them in the same pass as "all" risked "all" never being reached in
# reasonable time — so "all" is isolated here to guarantee it completes
# first, independent of how long the RI ladder takes.
ALL_ONLY_LEVELS="[all]"

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

inchikeys_fully_written() {
    # inchikeys_fully_written <hdf5_path> — True only if inchikey14 exists
    # AND every row has been written (add_inchikeys_inplace.py tracks this
    # via the rows_written attr so a mid-run interruption isn't mistaken
    # for completion).
    uv run python -c "
import h5py, sys
with h5py.File('$1', 'r') as f:
    if 'inchikey14' not in f:
        sys.exit(1)
    n = f['smiles'].shape[0]
    written = f['inchikey14'].attrs.get('rows_written', 0)
    sys.exit(0 if written >= n else 1)
" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Stage 1: ICICLE — attach inchikey14
# ---------------------------------------------------------------------------
log "=== Stage 1/7: ICICLE — add inchikey14 ==="
if inchikeys_fully_written "$ICICLE_HDF5"; then
    log "Skipping — inchikey14 already fully written in $ICICLE_HDF5"
else
    # add_inchikeys_inplace.py resumes from its own rows_written checkpoint
    # on re-invocation, so it's safe to call even after a partial run.
    uv run examples/scripts/evaluation/add_inchikeys_inplace.py \
        --hdf5 "$ICICLE_HDF5" \
        2>&1 | tee -a "$LOG_DIR/1_icicle_inchikeys.log"
fi

# ---------------------------------------------------------------------------
# Stage 2: ICICLE — attach RI columns
# ---------------------------------------------------------------------------
log "=== Stage 2/7: ICICLE — add RI columns ==="
if has_dataset "$ICICLE_HDF5" ri_StdNP; then
    log "Skipping — ri_StdNP already present in $ICICLE_HDF5"
else
    uv run examples/scripts/evaluation/add_ri_to_hdf5.py \
        --hdf5 "$ICICLE_HDF5" \
        --tsv "$RI_TSV" \
        2>&1 | tee "$LOG_DIR/2_icicle_ri.log"
fi

# ---------------------------------------------------------------------------
# Stage 3: ICICLE — build sort indices
# ---------------------------------------------------------------------------
log "=== Stage 3/7: ICICLE — add sort indices ==="
if has_dataset "$ICICLE_HDF5" sort_idx_StdNP; then
    log "Skipping — sort_idx_StdNP already present in $ICICLE_HDF5"
else
    uv run examples/scripts/evaluation/add_sort_index_to_hdf5.py \
        --hdf5 "$ICICLE_HDF5" \
        2>&1 | tee "$LOG_DIR/3_icicle_sort_index.log"
fi

# ---------------------------------------------------------------------------
# Stage 4: MassFormer — convert per-CID chunks to columnar format
# ---------------------------------------------------------------------------
log "=== Stage 4/7: MassFormer — convert chunks to columnar ==="
if [ -f "$MASSFORMER_COLUMNAR" ]; then
    log "Skipping — $MASSFORMER_COLUMNAR already exists"
else
    uv run examples/scripts/evaluation/convert_massformer_pubchem_chunks.py \
        2>&1 | tee "$LOG_DIR/4_massformer_convert.log"
fi

# ---------------------------------------------------------------------------
# Stage 5: MassFormer — build sort indices
# ---------------------------------------------------------------------------
log "=== Stage 5/7: MassFormer — add sort indices ==="
if has_dataset "$MASSFORMER_COLUMNAR" sort_idx_StdNP; then
    log "Skipping — sort_idx_StdNP already present in $MASSFORMER_COLUMNAR"
else
    uv run examples/scripts/evaluation/add_sort_index_to_hdf5.py \
        --hdf5 "$MASSFORMER_COLUMNAR" \
        2>&1 | tee "$LOG_DIR/5_massformer_sort_index.log"
fi

# ---------------------------------------------------------------------------
# Stage 6/7: "all"-only retrieval eval — ICICLE, NEIMS, MassFormer
# (priority result: true global rank against the full PubChem candidate set)
# ---------------------------------------------------------------------------
ICICLE_OUT="results/pubchem_retrieval_eval_icicle_rerun_260710"
NEIMS_OUT="results/pubchem_retrieval_eval_neims"
MASSFORMER_OUT="results/pubchem_retrieval_eval_massformer"

log "=== Stage 6/7: PubChem GLOBAL (all) retrieval eval — ICICLE ==="
if [ -f "$ICICLE_OUT/retrieval_global_results.json" ]; then
    log "Skipping — $ICICLE_OUT already has global results"
else
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$ICICLE_HDF5]" \
        output_dir="$ICICLE_OUT" \
        spectra_cache_dir="$ICICLE_OUT" \
        skipped_log="$ICICLE_OUT/skipped_queries.tsv" \
        top_n_levels="$ALL_ONLY_LEVELS" \
        2>&1 | tee "$LOG_DIR/6_icicle_global_eval.log"
fi

log "=== Stage 6/7: PubChem GLOBAL (all) retrieval eval — NEIMS ==="
if [ -f "$NEIMS_OUT/retrieval_global_results.json" ]; then
    log "Skipping — $NEIMS_OUT already has global results"
else
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$NEIMS_HDF5]" \
        output_dir="$NEIMS_OUT" \
        spectra_cache_dir="$NEIMS_OUT" \
        skipped_log="$NEIMS_OUT/skipped_queries.tsv" \
        top_n_levels="$ALL_ONLY_LEVELS" \
        2>&1 | tee "$LOG_DIR/6_neims_global_eval.log"
fi

log "=== Stage 7/7: PubChem GLOBAL (all) retrieval eval — MassFormer ==="
if [ -f "$MASSFORMER_OUT/retrieval_global_results.json" ]; then
    log "Skipping — $MASSFORMER_OUT already has global results"
else
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$MASSFORMER_COLUMNAR]" \
        output_dir="$MASSFORMER_OUT" \
        spectra_cache_dir="$MASSFORMER_OUT" \
        skipped_log="$MASSFORMER_OUT/skipped_queries.tsv" \
        top_n_levels="$ALL_ONLY_LEVELS" \
        2>&1 | tee "$LOG_DIR/7_massformer_global_eval.log"
fi

log "=== All 'all'-only global retrieval runs complete ==="
log "Results:"
log "  ICICLE:     $ICICLE_OUT/retrieval_global_results.json"
log "  NEIMS:      $NEIMS_OUT/retrieval_global_results.json"
log "  MassFormer: $MASSFORMER_OUT/retrieval_global_results.json"
log ""
log "RI-windowed ladder (1000/100k/1M/10M) is a separate pass — see"
log "run_pubchem_ri_window_ladder.sh, not run automatically here."
