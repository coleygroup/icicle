#!/usr/bin/env bash
# ICICLE similarity eval, scaffold + random_no_qcxms2, 3 seeds each.
# Checkpoints hardcoded (from PUB_READY_PLAN.md §1) to avoid Hydra override
# parsing issues with `=` characters in checkpoint filenames (val_loss=...).
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

STALL_TIMEOUT="${SIM_EVAL_STALL_TIMEOUT:-600}"  # seconds of no file activity before killing; override with SIM_EVAL_STALL_TIMEOUT=300
HARD_TIMEOUT="${SIM_EVAL_HARD_TIMEOUT:-4h}"     # absolute ceiling in case the watchdog itself misbehaves

run_sim() {
    local split_name="$1"
    local ckpt="$2"
    local out_dir="$3"
    ckpt="$(realpath "${ckpt}")"  # hydra.job.chdir=true moves cwd to hydra.run.dir, so relative paths break
    local full_out_dir="results/eval/${out_dir}"

    if [ -f "${full_out_dir}/similarity_results.csv" ]; then
        echo "[$(date)] ${out_dir}: similarity_results.csv already exists -- skipping."
        return 0
    fi

    echo "[$(date)] Starting ${out_dir} (stall timeout ${STALL_TIMEOUT}s, hard ceiling ${HARD_TIMEOUT})..."
    mkdir -p "${full_out_dir}"

    # Explicit master-port -- torchrun defaults to 29500, which collides with
    # any other concurrently-running torchrun job (independent of which GPUs
    # each targets; the port is a separate resource). Pick a free one fresh
    # per run so this script can run alongside other eval.py torchrun jobs.
    local master_port
    master_port=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

    timeout --signal=TERM --kill-after=60s "${HARD_TIMEOUT}" \
        uv run torchrun --nproc_per_node=2 --master-port="${master_port}" src/icicle/eval.py \
            data=NIST data.split_name="${split_name}" \
            eval=icicle_fe eval.similarity.enable=True \
            "eval.model.architecture.intensity_predictor_checkpoint='${ckpt}'" \
            system.devices=[0,5] \
            hydra.run.dir="${full_out_dir}" &
    local run_pid=$!

    # Stall watchdog: if nothing in full_out_dir has changed for STALL_TIMEOUT
    # seconds, the run is either wedged mid-computation or (far more likely
    # in practice) done and stuck in DDP teardown -- kill it either way and
    # let the exit-code/results check below decide success vs failure.
    (
        while kill -0 "$run_pid" 2>/dev/null; do
            sleep 30
            newest=$(find "${full_out_dir}" -type f -newermt "@$(( $(date +%s) - STALL_TIMEOUT ))" 2>/dev/null | head -1)
            if [ -z "$newest" ] && [ -n "$(find "${full_out_dir}" -type f 2>/dev/null)" ]; then
                echo "[$(date)] ${out_dir}: no file activity in ${STALL_TIMEOUT}s -- killing (pid $run_pid)."
                kill -TERM "$run_pid" 2>/dev/null
                sleep 5
                kill -KILL "$run_pid" 2>/dev/null
                break
            fi
        done
    ) &
    local watchdog_pid=$!

    wait "$run_pid" 2>/dev/null
    local exit_code=$?
    kill "$watchdog_pid" 2>/dev/null

    if [ -f "${full_out_dir}/similarity_results.csv" ]; then
        echo "[$(date)] ${out_dir}: results present, treating as success (exit code was $exit_code)."
    else
        echo "[$(date)] ${out_dir}: exited with code $exit_code and no results written -- moving on."
    fi
}

run_sim scaffold_no_xeno_aas_deduplicated \
    "checkpoints/entropy_scaffold_s1/checkpoints/best-model-val_loss=0.1793-epoch=39.ckpt" \
    final_entropy_scaffold_s1_sim

run_sim scaffold_no_xeno_aas_deduplicated \
    "checkpoints/entropy_scaffold_s2/checkpoints/best-model-val_loss=0.1792-epoch=43.ckpt" \
    final_entropy_scaffold_s2_sim

run_sim scaffold_no_xeno_aas_deduplicated \
    "checkpoints/entropy_scaffold_s3/checkpoints/best-model-val_loss=0.1815-epoch=38.ckpt" \
    final_entropy_scaffold_s3_sim

run_sim random_no_xeno_aas_deduplicated_no_qcxms2 \
    "checkpoints/entropy_random_s1/checkpoints/best-model-val_loss=0.1183-epoch=48.ckpt" \
    final_entropy_random_s1_sim

run_sim random_no_xeno_aas_deduplicated_no_qcxms2 \
    "checkpoints/entropy_random_s2/checkpoints/best-model-val_loss=0.1188-epoch=47.ckpt" \
    final_entropy_random_s2_sim

run_sim random_no_xeno_aas_deduplicated_no_qcxms2 \
    "checkpoints/entropy_random_s3/checkpoints/best-model-val_loss=0.1179-epoch=49.ckpt" \
    final_entropy_random_s3_sim
