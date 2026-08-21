#!/usr/bin/env bash
# Produces: similarity + formula-retrieval numbers for ICICLE/NEIMS/MassFormer/
# RASSP, all four evaluated on RASSP's own native scaffold split -- the
# four-way head-to-head comparison against RASSP.
#
# RASSP head-to-head: ICICLE, NEIMS, MassFormer, RASSP all evaluated on
# RASSP's native scaffold split (22,954 test mols) for a fair comparison.
# Requires: build_rassp_candidate_pickle.sh run first.
#
# Idempotent: each sub-run is skipped if its expected output CSV already
# exists, so this script can be safely rerun to fill in only what's missing
# (e.g. after a partial prior run, or to regenerate one baseline's dir).
#
# Fire-and-forget: does NOT use `set -e`. One run hanging or failing must not
# block the rest -- torchrun's DDP teardown reliably hangs after the actual
# work is already done and saved (observed repeatedly: eval.log shows
# "Evaluation complete" and all outputs written, but the process never exits
# because one rank blocks in process-group cleanup). A stall watchdog kills
# the run if its output directory goes quiet for STALL_TIMEOUT (default 10
# min) -- much tighter than a flat multi-hour timeout, since a real run
# writes to eval.log/checkpoints continuously while it's actually working.
#
# Usage: bash examples/scripts/evaluation/paper_reruns/run_rassp_headtohead.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

STALL_TIMEOUT="${RASSP_H2H_STALL_TIMEOUT:-600}"  # seconds of no file activity before killing
HARD_TIMEOUT="${RASSP_H2H_HARD_TIMEOUT:-4h}"     # absolute ceiling in case the watchdog itself misbehaves

RASSP_SPLIT="scaffold_no_xeno_aas_deduplicated_rassp"
RASSP_SPLIT_TSV="data/NIST2023_GCMS_main/splits/${RASSP_SPLIT}.tsv"
CANDS="data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_no_xeno_aas_deduplicated_rassp_50.pkl"
FORMULA_MAP="data/NIST2023_GCMS_main/retrieval/pubchem_formula_map_subset.p"
METADATA="data/NIST2023_GCMS_main/metadata.tsv"
SPECTRA="data/NIST2023_GCMS_main/spectra.hdf5"

if [[ ! -f "${CANDS}" ]]; then
    echo "ERROR: ${CANDS} not found. Run build_rassp_candidate_pickle.sh first." >&2
    exit 1
fi

# Absolute path: run_icicle's torchrun calls set hydra.job.chdir=true, which
# moves cwd to hydra.run.dir before eval.py reads eval.retrieval_with_formula.candidates_pickle,
# so a relative CANDS path breaks there (run_from_predictions's eval_from_predictions.py
# doesn't chdir, so it works with either — kept absolute here for one shared variable).
CANDS="$(realpath "${CANDS}")"

# ── ICICLE (similarity + formula retrieval, 3 seeds) ────────────────────────
run_timeout_torchrun() {
    # $1 = expected output csv basename (e.g. similarity_results.csv)
    # $2 = out_dir under results/eval/
    # remaining args = the eval.py args to pass through
    local expected_csv="$1"
    local out_dir="$2"
    shift 2
    local full_out_dir="results/eval/${out_dir}"

    if [ -f "${full_out_dir}/${expected_csv}" ]; then
        echo "[$(date)] ${out_dir}: ${expected_csv} already exists -- skipping."
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
        uv run torchrun --nproc_per_node=2 --master-port="${master_port}" src/icicle/eval.py "$@" \
            hydra.run.dir="${full_out_dir}" &
    local run_pid=$!

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

    if [ -f "${full_out_dir}/${expected_csv}" ]; then
        echo "[$(date)] ${out_dir}: results present, treating as success (exit code was $exit_code)."
    else
        echo "[$(date)] ${out_dir}: exited with code $exit_code and no results written -- moving on."
    fi
}

run_icicle() {
    local ckpt="$1"
    local out_dir="$2"
    ckpt="$(realpath "${ckpt}")"  # hydra.job.chdir=true moves cwd to hydra.run.dir, so relative paths break

    run_timeout_torchrun similarity_results.csv "${out_dir}_sim" \
        data=NIST data.split_name="${RASSP_SPLIT}" \
        eval=icicle_fe eval.similarity.enable=True \
        "eval.model.architecture.intensity_predictor_checkpoint='${ckpt}'" \
        system.devices=[0,1]

    run_timeout_torchrun retrieval_with_formula_results.csv "${out_dir}_retr" \
        data=NIST data.split_name="${RASSP_SPLIT}" \
        eval=icicle_fe eval.retrieval_with_formula.enable=True \
        eval.retrieval_with_formula.candidates_pickle="${CANDS}" \
        "eval.model.architecture.intensity_predictor_checkpoint='${ckpt}'" \
        system.devices=[0,1]
}

run_icicle "checkpoints/entropy_scaffold_s1/checkpoints/best-model-val_loss=0.1793-epoch=39.ckpt" entropy_scaffold_s1_rassp_split
run_icicle "checkpoints/entropy_scaffold_s2/checkpoints/best-model-val_loss=0.1792-epoch=43.ckpt" entropy_scaffold_s2_rassp_split
run_icicle "checkpoints/entropy_scaffold_s3/checkpoints/best-model-val_loss=0.1815-epoch=38.ckpt" entropy_scaffold_s3_rassp_split

# ── NEIMS / MassFormer / RASSP (via eval_from_predictions.py, 3 seeds) ──────
run_from_predictions() {
    # No DDP/torchrun here so no hang risk, but still guard so one missing
    # prediction file or bad baseline run doesn't abort the whole loop.
    local predictions="$1"
    local out_dir="$2"
    local cands="$3"

    if [ -f "results/eval/${out_dir}/similarity_results.csv" ]; then
        echo "[$(date)] ${out_dir}: similarity_results.csv already exists -- skipping."
        return 0
    fi
    if [ ! -f "${predictions}" ]; then
        echo "[$(date)] ${out_dir}: predictions file ${predictions} not found -- skipping."
        return 0
    fi

    uv run src/icicle/eval_from_predictions.py \
        --predictions "${predictions}" \
        --ground-truth "${SPECTRA}" \
        --labels "${METADATA}" \
        --splits "${RASSP_SPLIT_TSV}" \
        --output "results/eval/${out_dir}" \
        --mode all \
        --candidates-pickle "${cands}" \
        --formula-map "${FORMULA_MAP}"
    echo "[$(date)] ${out_dir}: exited with code $? "
}

for SEED in 1 2 3; do
    run_from_predictions "baselines/neims/results/predictions/neims_scaffold_s${SEED}_test.hdf5" \
        "neims_scaffold_s${SEED}_rassp_split" "${CANDS}"

    run_from_predictions "baselines/massformer/results/predictions/massformer_scaffold_s${SEED}_test.hdf5" \
        "massformer_scaffold_s${SEED}_rassp_split" "${CANDS}"

    run_from_predictions "baselines/rassp/results/predictions/rassp_scaffold_s${SEED}_test.hdf5" \
        "rassp_scaffold_s${SEED}_native_split" "${CANDS}"
done
