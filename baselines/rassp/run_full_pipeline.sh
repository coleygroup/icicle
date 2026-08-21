#!/usr/bin/env bash
# Full RASSP training + prediction + evaluation pipeline.
# Runs for scaffold and random splits, seeds 1/2/3.
# Must be run from the repo root (icicle-dev/).
#
# Usage:
#   bash baselines/rassp/run_full_pipeline.sh
#
# Prerequisites:
#   - conda env "rassp" installed (see baselines/rassp/rassp/environment.yml)
#
# Environment split:
#   Data conversion, training, inference: /mnt/home/magled/miniconda3/envs/rassp/bin/python (direct)
#   Evaluation:                           uv run (ICICLE env, from repo root)
#
# Hyperopt is run once on the scaffold split; its hparams are reused for random.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RASSP_DIR="${REPO_ROOT}/baselines/rassp"
DATA_DIR="${REPO_ROOT}/data/NIST2023_GCMS_main"

METADATA="${DATA_DIR}/metadata.tsv"
SPECTRA="${DATA_DIR}/spectra.hdf5"
FORMULA_MAP="${DATA_DIR}/retrieval/pubchem_formula_map_subset.p"

# Extra test datasets (similarity evaluation only)
MONA_DIR="${REPO_ROOT}/data/MoNA-export-GC-MS_Spectra"
MONA_CLEAN_SPLIT="${MONA_DIR}/splits/all_test_no_nist_scaffold_train.tsv"
XENO_SPLIT="${DATA_DIR}/splits/xeno_amino_acids.tsv"
VGWD_DIR="${REPO_ROOT}/data/VGWD2023_GCMS/raw"
VGWD_CLEAN_SPLIT="${VGWD_DIR}/splits/all_test_no_nist_scaffold_train.tsv"

# Parquet base names (suffix _train/_val/_test.parquet appended by convert script)
declare -A PARQUET_BASE=(
    [scaffold]="nist-scaffold-noxenoaas"
    [random]="nist-random"
)
# Full splits (used for eval_from_predictions.py which needs all molecules)
declare -A SPLITS_TSV=(
    [scaffold]="${DATA_DIR}/splits/scaffold_no_xeno_aas_deduplicated.tsv"
    [random]="${DATA_DIR}/splits/random_no_xeno_aas_deduplicated_no_qcxms2.tsv"
)
# RASSP-compatible filtered splits (used for parquet conversion and inference)
declare -A RASSP_SPLITS_TSV=(
    [scaffold]="${DATA_DIR}/splits/scaffold_no_xeno_aas_deduplicated_rassp.tsv"
    [random]="${DATA_DIR}/splits/random_no_xeno_aas_deduplicated_no_qcxms2_rassp.tsv"
)
declare -A CANDS_PICKLE=(
    [scaffold]="${DATA_DIR}/retrieval/cands_pickled_scaffold_50_compat.pkl"
    [random]="${DATA_DIR}/retrieval/cands_pickled_random_50_compat.pkl"
)

SEEDS=(1 2 3)
HYPEROPT_CONFIG="${RASSP_DIR}/rassp/expconfig/hyperopt.yaml"
# HYPEROPT_RESULTS="${RASSP_DIR}/results/hyperopt/rassp_hyperopt_scaffold_results.yaml"
HYPEROPT_RESULTS="${RASSP_DIR}/results/hyperopt_old/rassp_hyperopt_nist_scaffold_noxenoaas_results.yaml"
HYPEROPT_EPOCHS=100

# Path to the mamba-managed rassp env; use direct binary to avoid conda run
# picking up the active venv's Python instead of the env's Python.
RASSP_ENV="/mnt/home/magled/miniconda3/envs/rassp"
RASSP_PYTHON="${RASSP_ENV}/bin/python"

# Helper: find the latest checkpoint (.model file) in a checkpoint subdir
find_latest_checkpoint() {
    local ckpt_dir="$1"
    local prefix="${2:-}"
    find "${ckpt_dir}" -name "${prefix}*.model" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | awk '{print $2}'
}

# Filter splits to RASSP-compatible molecules (uv env, RDKit only)
for split_name in random; do
    rassp_split="${RASSP_SPLITS_TSV[$split_name]}"
    if [[ -f "${rassp_split}" ]]; then
        echo "[filter] ${split_name}: RASSP split already exists, skipping."
    else
        echo "[filter] ${split_name}: filtering split for RASSP compatibility ..."
        uv run examples/scripts/data_processing/filter_splits_for_rassp.py \
            --splits-path "${SPLITS_TSV[$split_name]}" \
            --metadata-path "${METADATA}" \
            --output-path "${rassp_split}" \
            --max-n-atoms 48
    fi
done

# Convert HDF5 → Parquet (skipped if all three split files exist)
# Uses RASSP-filtered split. Runs in rassp env (imports rassp.msutil).
for split_name in random; do
    base="${PARQUET_BASE[$split_name]}"
    train_pq="${RASSP_DIR}/${base}_train.parquet"
    val_pq="${RASSP_DIR}/${base}_val.parquet"
    test_pq="${RASSP_DIR}/${base}_test.parquet"

    if [[ -f "${train_pq}" && -f "${val_pq}" && -f "${test_pq}" ]]; then
        echo "[parquet] ${split_name}: parquet files already exist, skipping conversion."
    else
        echo "[parquet] ${split_name}: converting HDF5 → parquet ..."
        (
            cd "${RASSP_DIR}"
            ${RASSP_PYTHON} convert_hdf5_to_parquet.py \
                    --hdf5-path "${SPECTRA}" \
                    --split-file "${RASSP_SPLITS_TSV[$split_name]}" \
                    --name "${RASSP_DIR}/${base}"
        )
    fi
done

# Hyperopt on scaffold split only
if [[ -f "${HYPEROPT_RESULTS}" ]]; then
    echo "[hyperopt] scaffold: results already exist, skipping."
else
    echo "[hyperopt] scaffold: running hyperopt ..."
    mkdir -p "${RASSP_DIR}/results/hyperopt"
    (
        cd "${RASSP_DIR}/rassp"
        ${RASSP_PYTHON} hyperopt.py "${HYPEROPT_CONFIG}" \
                --output-dir "${RASSP_DIR}/results/hyperopt" \
                --study-name "rassp_hyperopt_scaffold"
    )
fi

# Per-(split, seed): generate config, train, predict, eval
# Scaffold hyperopt hparams are reused for both splits.
for split_name in random; do
    base="${PARQUET_BASE[$split_name]}"
    splits_tsv="${SPLITS_TSV[$split_name]}"
    cands_pickle="${CANDS_PICKLE[$split_name]}"

    for seed in "${SEEDS[@]}"; do
        run_tag="rassp_${split_name}_s${seed}"
        cfg_path="${RASSP_DIR}/rassp/expconfig/${run_tag}.yaml"
        ckpt_subdir="${RASSP_DIR}/rassp/checkpoints"
        pred_dir="${RASSP_DIR}/results/predictions"
        eval_dir="${REPO_ROOT}/results/eval/${run_tag}"

        mkdir -p "${ckpt_subdir}" "${pred_dir}" "${eval_dir}"

        # Generate per-seed training config from scaffold hyperopt results
        if [[ ! -f "${cfg_path}" ]]; then
            echo "[config] ${run_tag}: generating training config ..."
            (
                cd "${RASSP_DIR}/rassp"
                ${RASSP_PYTHON} generate_config_from_hyperopt.py \
                        "${HYPEROPT_RESULTS}" \
                        "${HYPEROPT_CONFIG}" \
                        -o "${cfg_path}" \
                        --max-epochs "${HYPEROPT_EPOCHS}"
            )
            # Patch seed, checkpoint dir, and split-specific parquet filenames
            python3 - <<PYEOF
import yaml

with open("${cfg_path}") as f:
    cfg = yaml.safe_load(f)

cfg["seed"] = ${seed}
cfg["cluster_config"]["checkpoint_dir"] = "${ckpt_subdir}"
cfg["cluster_config"]["data_dir"] = "${RASSP_DIR}"
cfg["exp_data"]["data"][0]["db_filename"] = "${base}_train.parquet"
cfg["exp_data"]["data"][1]["db_filename"] = "${base}_val.parquet"

with open("${cfg_path}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
PYEOF
        else
            echo "[config] ${run_tag}: config already exists, skipping generation."
        fi

        # Train
        latest_ckpt=$(find_latest_checkpoint "${ckpt_subdir}" "${run_tag}")
        if [[ -n "${latest_ckpt}" ]]; then
            echo "[train] ${run_tag}: checkpoint exists (${latest_ckpt}), skipping training."
        else
            echo "[train] ${run_tag}: training ..."
            (
                cd "${RASSP_DIR}/rassp"
                USE_CUDA=1 ${RASSP_PYTHON} forward_train.py "${cfg_path}" "${run_tag}" --skip-timestamp
            )
            latest_ckpt=$(find_latest_checkpoint "${ckpt_subdir}" "${run_tag}")
        fi

        ckpt_meta="$(dirname "${latest_ckpt}")/${run_tag}.${run_tag}.meta"

        # ── Predict: all test sets ──────────────────────────────────────

        test_pred="${pred_dir}/${run_tag}_test.hdf5"
        if [[ -f "${test_pred}" ]]; then
            echo "[predict/test] ${run_tag}: already exists, skipping."
        else
            echo "[predict/test] ${run_tag}: generating NIST test predictions ..."
            (
                cd "${RASSP_DIR}"
                ${RASSP_PYTHON} scripts/run_inference_for_eval.py \
                    --checkpoint "${latest_ckpt}" \
                    --meta "${ckpt_meta}" \
                    --metadata "${METADATA}" \
                    --splits "${RASSP_SPLITS_TSV[$split_name]}" \
                    --eval-split test \
                    --output "${test_pred}" \
                    --gpu --data-parallel --batch-size 32 --num-workers 4
            )
        fi

        mona_pred="${pred_dir}/${run_tag}_mona_test.hdf5"
        if [[ ! -f "${mona_pred}" ]]; then
            echo "[predict/mona] ${run_tag}: generating MoNA predictions ..."
            (
                cd "${RASSP_DIR}"
                ${RASSP_PYTHON} scripts/run_inference_for_eval.py \
                    --checkpoint "${latest_ckpt}" \
                    --meta "${ckpt_meta}" \
                    --metadata "${MONA_DIR}/metadata.tsv" \
                    --splits "${MONA_CLEAN_SPLIT}" \
                    --eval-split test \
                    --output "${mona_pred}" \
                    --gpu --data-parallel --batch-size 32 --num-workers 4
            )
        fi

        xeno_pred="${pred_dir}/${run_tag}_xeno_aas_test.hdf5"
        if [[ ! -f "${xeno_pred}" ]]; then
            echo "[predict/xeno_aas] ${run_tag}: generating xeno-AA predictions ..."
            (
                cd "${RASSP_DIR}"
                ${RASSP_PYTHON} scripts/run_inference_for_eval.py \
                    --checkpoint "${latest_ckpt}" \
                    --meta "${ckpt_meta}" \
                    --metadata "${METADATA}" \
                    --splits "${XENO_SPLIT}" \
                    --eval-split test \
                    --output "${xeno_pred}" \
                    --gpu --data-parallel --batch-size 32 --num-workers 4
            )
        fi

        vgwd_pred="${pred_dir}/${run_tag}_vgwd_test.hdf5"
        if [[ ! -f "${vgwd_pred}" ]]; then
            echo "[predict/vgwd] ${run_tag}: generating VGWD predictions ..."
            (
                cd "${RASSP_DIR}"
                ${RASSP_PYTHON} scripts/run_inference_for_eval.py \
                    --checkpoint "${latest_ckpt}" \
                    --meta "${ckpt_meta}" \
                    --metadata "${VGWD_DIR}/labels.tsv" \
                    --splits "${VGWD_CLEAN_SPLIT}" \
                    --eval-split test \
                    --output "${vgwd_pred}" \
                    --gpu --data-parallel --batch-size 32 --num-workers 4
            )
        fi

        # ── Similarity evals (all test sets) ───────────────────────────

        if [[ -f "${eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/nist-sim] ${run_tag}: exists, skipping."
        else
            echo "[eval/nist-sim] ${run_tag}: running NIST similarity evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${test_pred}" \
                    --ground-truth "${DATA_DIR}/spectra.hdf5" \
                    --labels "${METADATA}" \
                    --splits "${splits_tsv}" \
                    --output "${eval_dir}" \
                    --mode similarity
            )
        fi

        mona_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_mona"
        if [[ -f "${mona_eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/mona] ${run_tag}: exists, skipping."
        else
            echo "[eval/mona] ${run_tag}: running similarity evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${mona_pred}" \
                    --ground-truth "${MONA_DIR}/spectra.hdf5" \
                    --labels "${MONA_DIR}/metadata.tsv" \
                    --splits "${MONA_CLEAN_SPLIT}" \
                    --output "${mona_eval_dir}" \
                    --mode similarity
            )
        fi

        xeno_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_xeno_aas"
        if [[ -f "${xeno_eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/xeno_aas] ${run_tag}: exists, skipping."
        else
            echo "[eval/xeno_aas] ${run_tag}: running similarity evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${xeno_pred}" \
                    --ground-truth "${SPECTRA}" \
                    --labels "${METADATA}" \
                    --splits "${XENO_SPLIT}" \
                    --output "${xeno_eval_dir}" \
                    --mode similarity
            )
        fi

        vgwd_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_vgwd"
        if [[ -f "${vgwd_eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/vgwd] ${run_tag}: exists, skipping."
        else
            echo "[eval/vgwd] ${run_tag}: running similarity evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${vgwd_pred}" \
                    --ground-truth "${VGWD_DIR}/mgf_files/VGWD2023.mgf" \
                    --labels "${VGWD_DIR}/labels.tsv" \
                    --splits "${VGWD_CLEAN_SPLIT}" \
                    --output "${vgwd_eval_dir}" \
                    --mode similarity
            )
        fi

        # ── PubChem retrieval ───────────────────────────────────────────
        # TEST: run inference + eval on x% of test query molecules
        QUERY_FRACTION=1.0

        cands_pred="${pred_dir}/${run_tag}_pubchem_cands_50.hdf5"
        if [[ -f "${cands_pred}" ]]; then
            echo "[predict/cands] ${run_tag}: already exists, skipping."
        else
            echo "[predict/cands] ${run_tag}: generating PubChem candidate predictions ..."
            (
                cd "${RASSP_DIR}"
                ${RASSP_PYTHON} scripts/run_inference_for_eval.py \
                    --checkpoint "${latest_ckpt}" \
                    --meta "${ckpt_meta}" \
                    --candidates-pickle "${cands_pickle}" \
                    --query-fraction "${QUERY_FRACTION}" \
                    --output "${cands_pred}" \
                    --gpu --data-parallel --batch-size 32 --num-workers 4
            )
        fi

        if [[ -f "${eval_dir}/retrieval_with_formula_results.csv" ]]; then
            echo "[eval/retrieval] ${run_tag}: exists, skipping."
        else
            echo "[eval/retrieval] ${run_tag}: running retrieval evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${cands_pred}" \
                    --test-predictions "${test_pred}" \
                    --ground-truth "${DATA_DIR}/spectra.hdf5" \
                    --labels "${METADATA}" \
                    --splits "${splits_tsv}" \
                    --output "${eval_dir}" \
                    --mode retrieval \
                    --candidates-pickle "${cands_pickle}" \
                    --formula-map "${FORMULA_MAP}"
            )
        fi

        echo "[done] ${run_tag}"
    done
done

echo ""
echo "All runs complete. Results in ${REPO_ROOT}/results/eval/."
