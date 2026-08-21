#!/usr/bin/env bash
# Produces: SI Table tab:si-ri-ablation-all, tab:si-recall-vs-pool (RI rows).
#
# RI-window ladder (N=1000..all candidates by predicted-RI closeness) for
# ICICLE/NEIMS/MassFormer, all 3 RI column types (StdNP/SemiStdNP/StdPolar),
# random split only (RI infrastructure is random-split-trained).
#
# Each invocation below overwrites retrieval_ablation_<ri_type>.json and
# retrieval_per_query_<ri_type>_<level>.tsv in-place for one RI type, so the
# combined retrieval_ablation_all_ri_types.json is reassembled from the three
# per-type files at the end -- a single call with all three ri_types would
# also work in principle, but running one RI type at a time makes it trivial
# to rerun just one type if it needs to be redone without repeating the rest.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash examples/scripts/evaluation/paper_reruns/rerun_ri_ladder.sh
set -euo pipefail
cd /home/magled/icicle-dev

LOG_DIR="results/pubchem_retrieval_pipeline_logs"
mkdir -p "$LOG_DIR"

ICICLE_HDF5="results/inference/pubchem_predictions_rerun_260710.hdf5"
NEIMS_HDF5="baselines/neims/results/pubchem_predictions/neims_random_s1_pubchem_full.hdf5"
MASSFORMER_COLUMNAR="baselines/massformer/results/predictions/massformer_random_s2_pubchem_full_columnar.hdf5"

ICICLE_OUT="results/pubchem_retrieval_eval_icicle_rerun_260710"
NEIMS_OUT="results/pubchem_retrieval_eval_neims"
MASSFORMER_OUT="results/pubchem_retrieval_eval_massformer"

RI_WINDOW_LEVELS="[1000,100000,1000000,10000000,all]"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local hdf5=$1
    local out_dir=$2
    local ri_type=$3
    local logfile=$4
    log "=== RI ladder: $ri_type on $hdf5 ==="
    uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
        hdf5_files="[$hdf5]" \
        output_dir="$out_dir" \
        spectra_cache_dir="$out_dir" \
        skipped_log="$out_dir/skipped_queries.tsv" \
        ri_types="[$ri_type]" \
        top_n_levels="$RI_WINDOW_LEVELS" \
        2>&1 | tee "$LOG_DIR/$logfile"
    log "=== Done: $ri_type on $hdf5 ==="
}

run_one "$ICICLE_HDF5" "$ICICLE_OUT" "StdNP" "ri_ladder_icicle_stdnp.log"
run_one "$ICICLE_HDF5" "$ICICLE_OUT" "SemiStdNP" "ri_ladder_icicle_semistdnp.log"
run_one "$ICICLE_HDF5" "$ICICLE_OUT" "StdPolar" "ri_ladder_icicle_stdpolar.log"

run_one "$NEIMS_HDF5" "$NEIMS_OUT" "StdNP" "ri_ladder_neims_stdnp.log"
run_one "$NEIMS_HDF5" "$NEIMS_OUT" "SemiStdNP" "ri_ladder_neims_semistdnp.log"
run_one "$NEIMS_HDF5" "$NEIMS_OUT" "StdPolar" "ri_ladder_neims_stdpolar.log"

run_one "$MASSFORMER_COLUMNAR" "$MASSFORMER_OUT" "StdNP" "ri_ladder_massformer_stdnp.log"
run_one "$MASSFORMER_COLUMNAR" "$MASSFORMER_OUT" "SemiStdNP" "ri_ladder_massformer_semistdnp.log"
run_one "$MASSFORMER_COLUMNAR" "$MASSFORMER_OUT" "StdPolar" "ri_ladder_massformer_stdpolar.log"

log "=== Rebuilding combined retrieval_ablation_all_ri_types.json for each model ==="
for out_dir in "$ICICLE_OUT" "$NEIMS_OUT" "$MASSFORMER_OUT"; do
    uv run python3 -c "
import json
from pathlib import Path

out_dir = Path('$out_dir')
combined = {}
for ri_type in ['StdNP', 'SemiStdNP', 'StdPolar']:
    f = out_dir / f'retrieval_ablation_{ri_type}.json'
    if f.exists():
        combined[ri_type] = json.loads(f.read_text())
(out_dir / 'retrieval_ablation_all_ri_types.json').write_text(json.dumps(combined, indent=2))
print(f'Rebuilt {out_dir}/retrieval_ablation_all_ri_types.json with RI types: {list(combined.keys())}')
"
done

log "=== ALL RI LADDER RERUNS COMPLETE ==="
