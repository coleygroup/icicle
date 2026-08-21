#!/usr/bin/env bash
# Appendix loss-function ablation: weighted_cosine + composite_weighted_cosine,
# scaffold split, 3 seeds each, similarity only. Verified each checkpoint dir
# has exactly one best-model*.ckpt file before relying on the glob.
#
# Fire-and-forget: does NOT use `set -e`. One run hanging or failing must not
# block the rest -- torchrun's DDP teardown reliably hangs after the actual
# work is already done and saved (observed repeatedly: eval.log shows
# "Evaluation complete" and all outputs written, but the process never exits
# because one rank blocks in process-group cleanup). A stall watchdog kills
# the run if its output directory goes quiet for STALL_TIMEOUT (default 10
# min) -- much tighter than a flat multi-hour timeout, since a real run
# writes to eval.log/checkpoints continuously while it's actually working.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

STALL_TIMEOUT="${LOSS_ABLATION_STALL_TIMEOUT:-600}"  # seconds of no file activity before killing
HARD_TIMEOUT="${LOSS_ABLATION_HARD_TIMEOUT:-4h}"     # absolute ceiling in case the watchdog itself misbehaves

for LOSS in weighted_cosine composite_weighted_cosine; do
  for SEED in 1 2 3; do
    OUT_DIR="${LOSS}_scaffold_s${SEED}_sim_final"
    FULL_OUT_DIR="results/eval/${OUT_DIR}"

    if [ -f "${FULL_OUT_DIR}/similarity_results.csv" ]; then
      echo "[$(date)] ${OUT_DIR}: similarity_results.csv already exists -- skipping."
      continue
    fi

    CKPT=$(realpath "$(ls checkpoints/${LOSS}_scaffold_s${SEED}/checkpoints/best-model*.ckpt)")  # hydra.job.chdir=true moves cwd to hydra.run.dir, so relative paths break

    echo "[$(date)] Starting ${OUT_DIR} (stall timeout ${STALL_TIMEOUT}s, hard ceiling ${HARD_TIMEOUT})..."
    mkdir -p "${FULL_OUT_DIR}"

    # Explicit master-port -- torchrun defaults to 29500, which collides with
    # any other concurrently-running torchrun job (independent of which GPUs
    # each targets; the port is a separate resource). Pick a free one fresh
    # per run so this script can run alongside other eval.py torchrun jobs.
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

    timeout --signal=TERM --kill-after=60s "${HARD_TIMEOUT}" \
      uv run torchrun --nproc_per_node=2 --master-port="${MASTER_PORT}" src/icicle/eval.py \
        data=NIST data.split_name=scaffold_no_xeno_aas_deduplicated \
        eval=icicle_fe eval.similarity.enable=True \
        "eval.model.architecture.intensity_predictor_checkpoint='${CKPT}'" \
        system.devices=[1,2] \
        hydra.run.dir="${FULL_OUT_DIR}" &
    RUN_PID=$!

    (
      while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep 30
        newest=$(find "${FULL_OUT_DIR}" -type f -newermt "@$(( $(date +%s) - STALL_TIMEOUT ))" 2>/dev/null | head -1)
        if [ -z "$newest" ] && [ -n "$(find "${FULL_OUT_DIR}" -type f 2>/dev/null)" ]; then
          echo "[$(date)] ${OUT_DIR}: no file activity in ${STALL_TIMEOUT}s -- killing (pid $RUN_PID)."
          kill -TERM "$RUN_PID" 2>/dev/null
          sleep 5
          kill -KILL "$RUN_PID" 2>/dev/null
          break
        fi
      done
    ) &
    WATCHDOG_PID=$!

    wait "$RUN_PID" 2>/dev/null
    EXIT_CODE=$?
    kill "$WATCHDOG_PID" 2>/dev/null

    if [ -f "${FULL_OUT_DIR}/similarity_results.csv" ]; then
      echo "[$(date)] ${OUT_DIR}: results present, treating as success (exit code was $EXIT_CODE)."
    else
      echo "[$(date)] ${OUT_DIR}: exited with code $EXIT_CODE and no results written -- moving on."
    fi
  done
done
