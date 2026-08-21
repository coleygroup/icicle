#!/bin/bash
#SBATCH --job-name=neims_pubchem_infer
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=magled@mit.edu

# ── Config ────────────────────────────────────────────────────────────────────
CKPT="${CKPT:-outputs/neims_random_s1_usedofrpubcheminf/best_model.pt}"
INPUT="${INPUT:-../../data/PubChem/PubChem_filtered.tsv}"
OUTPUT="${OUTPUT:-results/pubchem_predictions/neims_random_s1_pubchem_full.hdf5}"

# Optional: set to slice the dataset for multi-node parallelism.
# Leave blank to process everything.
#   Node 1: START_IDX=0        END_IDX=60000000
#   Node 2: START_IDX=60000000 END_IDX=119314794
START_IDX=""
END_IDX=""
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../.env 2>/dev/null || true
module load miniforge cuda/12.4 gcc/12.2.0 2>/dev/null || true

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p logs results/pubchem_predictions

SLICE_ARGS=""
[ -n "$START_IDX" ] && SLICE_ARGS="$SLICE_ARGS --start-idx $START_IDX"
[ -n "$END_IDX"   ] && SLICE_ARGS="$SLICE_ARGS --end-idx $END_IDX"

uv run --no-sync python batch_infer_pubchem_neims.py \
    --checkpoint "$CKPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --batch-size 2048 \
    --num-workers 8 \
    $SLICE_ARGS

echo "Done: $OUTPUT"
