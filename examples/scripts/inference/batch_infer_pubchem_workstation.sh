#!/bin/bash
# Run PubChem batch inference on a local multi-GPU workstation (no SLURM),
# with ONE FULLY INDEPENDENT OS PROCESS PER GPU.
#
# Why per-GPU processes instead of one multi-GPU batch_infer.py invocation:
# a single process managing all 8 GPUs via Python multiprocessing.Process
# can hang entirely if one worker's shutdown/signal-handling path stalls --
# taking every GPU down with it, since the dispatcher process blocks in
# p.join() waiting on the wedged child. Running each GPU as its own
# top-level `batch_infer.py --gpus N` process means a hang or crash on one
# GPU is fully contained: the other 7 processes share nothing with it and
# keep running untouched.
#
# Each per-GPU process is wrapped in its own restart loop, so a crash (e.g.
# CUDA OOM abort, an uncatchable C++ std::terminate) on GPU N only restarts
# GPU N -- it resumes from its last checkpoint within seconds, with zero
# manual intervention and no effect on the other GPUs.
#
# The restart loop only catches CRASHES (process exits). It does NOT catch
# HANGS -- observed in practice: a worker gets wedged after GPU warm-up,
# before the first DataLoader batch (a CUDA-context-vs-fork interaction),
# and the process never exits, consumes ~0% GPU, and sits there indefinitely
# with no error logged. A per-GPU watchdog below detects this by checking
# whether _shard_N.hdf5.progress.json has been updated recently; if not, it
# kills that GPU's process group so the restart loop relaunches it.
#
# Run this inside a screen/tmux session so it survives disconnects. Do not
# kill it with a broad `pkill -f batch_infer.py` while it's the *parent*
# script matching that pattern too -- that can kill this wrapper itself,
# orphaning all 8 GPU processes with no supervisor left to restart them.
# See the "Stopping" section in CLAUDE.md for the correct kill order.
#
# Usage: bash examples/scripts/inference/batch_infer_pubchem_workstation.sh

set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 1

CKPT="${CKPT:-checkpoints/entropy_random_s1/checkpoints/best-model-val_loss=0.1183-epoch=48.ckpt}"
INPUT="${INPUT:-/tmp/PubChem_filtered_for_AIRI.csv}"  # copy from data/PubChem/ to local disk first to avoid NFS contention
OUTPUT="${OUTPUT:-results/inference/pubchem_predictions_rerun_260710.hdf5}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"   # DataLoader workers per GPU
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NODES="${MAX_NODES:-100}"
SHARD_DIR="$(dirname "$OUTPUT")"   # shard files, progress json, and locks all live next to OUTPUT
LOG_DIR="${LOG_DIR:-$SHARD_DIR/gpu_logs}"
STALL_TIMEOUT_S=900   # kill+restart a GPU if its shard progress file hasn't
                       # been touched in this long (covers post-warmup hangs;
                       # generous enough to not false-positive on a slow but
                       # live GPU -- checkpoints flush every 100*BATCH_SIZE mols)

mkdir -p "$(dirname "$OUTPUT")" "$LOG_DIR"

if [ ! -f "$INPUT" ]; then
    echo "Input not found at $INPUT — copy the PubChem CSV to local disk first:"
    echo "  cp data/PubChem/PubChem_filtered_for_AIRI.csv $INPUT"
    exit 1
fi

TOTAL_ROWS=$(wc -l < "$INPUT")
CHUNK=$(( (TOTAL_ROWS + NUM_GPUS - 1) / NUM_GPUS ))

watchdog() {
    # Kills the given process group if the shard's progress file hasn't
    # advanced (in size) for STALL_TIMEOUT_S since THIS process launched --
    # catches hangs (process never exits on its own) that the exit-code-based
    # restart loop in run_gpu() cannot see.
    #
    # Tracks progress by file SIZE, not mtime: a fresh restart inherits a
    # shard file whose mtime is from a much earlier run, so comparing against
    # absolute mtime age would immediately look "stalled" and kill the new,
    # healthy process before it ever gets a chance to run (this happened in
    # practice -- see CLAUDE.md changelog). Size only changes on an actual
    # checkpoint append, so a growing size is unambiguous real progress.
    local gpu_id=$1
    local worker_pgid=$2
    local log_file="$LOG_DIR/gpu_${gpu_id}.out"
    local progress_file="$SHARD_DIR/_shard_${gpu_id}.hdf5.progress.json"
    local baseline_size=-1
    local baseline_time
    baseline_time=$(date +%s)
    if [ -f "$progress_file" ]; then
        baseline_size=$(stat -c %s "$progress_file" 2>/dev/null || echo -1)
    fi

    while kill -0 "-$worker_pgid" 2>/dev/null; do
        sleep 60
        local cur_size=-1
        if [ -f "$progress_file" ]; then
            cur_size=$(stat -c %s "$progress_file" 2>/dev/null || echo -1)
        fi
        if [ "$cur_size" != "$baseline_size" ]; then
            # Real progress happened -- reset the baseline and clock.
            baseline_size=$cur_size
            baseline_time=$(date +%s)
            continue
        fi
        local now
        now=$(date +%s)
        local stalled_for=$(( now - baseline_time ))
        if [ "$stalled_for" -ge "$STALL_TIMEOUT_S" ]; then
            echo "[$(date)] [GPU $gpu_id] WATCHDOG: no progress in ${stalled_for}s since this process launched -- killing wedged process group $worker_pgid." >> "$log_file"
            kill -9 "-$worker_pgid" 2>/dev/null
            return
        fi
    done
}

run_gpu() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local log_file="$LOG_DIR/gpu_${gpu_id}.out"
    local lock_file="$SHARD_DIR/_shard_${gpu_id}.hdf5.lock"

    # Exclusive lock on this GPU's shard for the lifetime of this run_gpu
    # invocation. Prevents two concurrent launches (e.g. the script started
    # twice by accident, or a stale process surviving a kill) from both
    # writing to the same shard file at once -- this happened in practice
    # and corrupted 4 shards with duplicate/interleaved rows (see CLAUDE.md
    # changelog). flock blocks here if another process already holds it,
    # rather than silently racing.
    exec {lock_fd}>"$lock_file"
    if ! flock -n "$lock_fd"; then
        echo "[$(date)] [GPU $gpu_id] Another process already holds the lock on $lock_file -- refusing to start to avoid corrupting the shard. Exiting." >> "$log_file"
        return 1
    fi

    while true; do
        echo "[$(date)] [GPU $gpu_id] Launching (rows [$start_idx:$end_idx))..." >> "$log_file"
        setsid uv run src/icicle/batch_infer.py \
            --intensity-predictor "$CKPT" \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --gpus "$gpu_id" \
            --num-gpus 1 \
            --shard-id "$gpu_id" \
            --start-idx "$start_idx" \
            --end-idx "$end_idx" \
            --num-workers "$NUM_WORKERS" \
            --batch-size "$BATCH_SIZE" \
            --max-nodes "$MAX_NODES" \
            --no-header --smiles-col-idx 0 \
            >> "$log_file" 2>&1 &
        local worker_pid=$!

        watchdog "$gpu_id" "$worker_pid" &
        local watchdog_pid=$!

        wait "$worker_pid"
        local exit_code=$?
        kill "$watchdog_pid" 2>/dev/null
        echo "[$(date)] [GPU $gpu_id] Exited with code $exit_code" >> "$log_file"

        if [ ! -f "$SHARD_DIR/_shard_${gpu_id}.hdf5.progress.json" ] \
           && [ -f "$SHARD_DIR/_shard_${gpu_id}.hdf5" ]; then
            echo "[$(date)] [GPU $gpu_id] Shard complete." >> "$log_file"
            break
        fi

        echo "[$(date)] [GPU $gpu_id] Restarting in 10s..." >> "$log_file"
        sleep 10
    done
}

echo "Launching $NUM_GPUS independent per-GPU processes. Tail logs with:"
echo "  tail -f $LOG_DIR/gpu_*.out"
echo

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    start_idx=$(( gpu_id * CHUNK ))
    end_idx=$(( (gpu_id + 1) * CHUNK ))
    if [ "$end_idx" -gt "$TOTAL_ROWS" ]; then
        end_idx=$TOTAL_ROWS
    fi
    run_gpu "$gpu_id" "$start_idx" "$end_idx" &
done

wait
echo "All GPU processes finished."
