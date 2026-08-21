#!/bin/bash
#SBATCH --job-name=icicle_infer
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --partition=mit_preemptable,mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=magled@mit.edu

cd /home/magled/icicle-dev
source .env
module load miniforge cuda/12.4 gcc/12.2.0
export TORCH_CPP_LOG_LEVEL="ERROR"

export GCC_LIB_DIR=/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64
export LD_LIBRARY_PATH=$GCC_LIB_DIR:$LD_LIBRARY_PATH

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Config
CKPT="results/260304_random_best/best-model-val_loss=0.1473-epoch=98.ckpt"
INPUT="data/PubChem/PubChem_filtered_for_AIRI.csv"
OUTPUT="results/pubchem_predictions_20M_to_30M.hdf5"

# Optional: slice the dataset for this node (leave blank to process all)
# Example for two-node split:
#   Node 1: --start-idx 0        --end-idx 60000000
#   Node 2: --start-idx 60000000
START_IDX="20000000"
END_IDX="30000000"

# Build optional slice args
SLICE_ARGS=""
[ -n "$START_IDX" ] && SLICE_ARGS="$SLICE_ARGS --start-idx $START_IDX"
[ -n "$END_IDX"   ] && SLICE_ARGS="$SLICE_ARGS --end-idx $END_IDX"

# Signal handler: forward SIGUSR1 to the Python child so it can flush its
# in-memory buffer before the node is reclaimed, then wait for it to exit.
handle_preemption() {
    echo "Received preemption signal (SIGUSR1) at $(date) — forwarding to Python child (PID=$CHILD_PID)..."
    if [ -n "$CHILD_PID" ]; then
        kill -USR1 "$CHILD_PID" 2>/dev/null || true
        wait $CHILD_PID
    fi
    exit 0
}
trap handle_preemption SIGUSR1

mkdir -p logs results

uv run --no-sync src/icicle/batch_infer.py \
    --intensity-predictor "$CKPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --num-gpus 4 \
    --num-workers 6 \
    --batch-size 256 \
    --max-nodes 100 \
    --checkpoint-every 25000 \
    --no-header \
    --smiles-col-idx 1 \
    $SLICE_ARGS &

CHILD_PID=$!
wait $CHILD_PID
