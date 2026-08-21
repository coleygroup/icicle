#!/usr/bin/env bash
# Full NEIMS training + prediction + evaluation pipeline.
# Runs for scaffold and random splits, both model types, seeds 1/2/3.
# Must be run from the repo root (icicle-dev/).
#
# Usage:
#   bash baselines/neims/run_full_pipeline.sh
#
# Hyperopt is only run once per (split, model_type) pair using the scaffold
# split for the first pass; results are reused across seeds.
# Candidate pickle used: cands_pickled_scaffold_50_compat.pkl (50 formula cands).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEIMS_DIR="${REPO_ROOT}/baselines/neims"
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

declare -A SPLITS=(
    [scaffold]="${DATA_DIR}/splits/scaffold_no_xeno_aas_deduplicated.tsv"
    [random]="${DATA_DIR}/splits/random_no_xeno_aas_deduplicated.tsv"
)
declare -A CANDS_PICKLE=(
    [scaffold]="${DATA_DIR}/retrieval/cands_pickled_scaffold_50_compat.pkl"
    [random]="${DATA_DIR}/retrieval/cands_pickled_random_50_compat.pkl"
)

SEEDS=(1 2 3)
RANDOM_SEEDS=(1)   # random split: single seed for PubChem comparison
MODEL_TYPES=(neims)

neims_hparam_args() {
    local json_path="$1"
    python3 - <<EOF
import json, sys
with open("${json_path}") as f:
    d = json.load(f)
p = d["best_params"]
args = []
num_layers      = p.pop("num_layers", None)
hidden_size     = p.pop("hidden_size", None)
use_ffnn        = p.pop("use_ffnn", None)
ffnn_num_layers = p.pop("ffnn_num_layers", None)
ffnn_hidden_size = p.pop("ffnn_hidden_size", None)
for k, v in p.items():
    flag = "--" + k.replace("_", "-")
    if isinstance(v, bool):
        args += [flag, str(v).lower()]
    else:
        args += [flag, str(v)]
if num_layers is not None and hidden_size is not None:
    args += ["--hidden-sizes"] + [str(hidden_size)] * int(num_layers)
if use_ffnn is not None and not use_ffnn:
    args += ["--ffnn-hidden-sizes"]
elif ffnn_num_layers is not None and ffnn_hidden_size is not None:
    args += ["--ffnn-hidden-sizes"] + [str(ffnn_hidden_size)] * int(ffnn_num_layers)
print(" ".join(args))
EOF
}

# Hyperopt (once per model type, using scaffold split)
run_hyperopt() {
    local model_type="$1"
    local hopt_dir="${NEIMS_DIR}/hyperopt_results_${model_type}"

    if [[ -f "${hopt_dir}/best_params.json" ]]; then
        echo "[hyperopt] ${model_type}: best_params.json already exists, skipping."
        return
    fi

    echo "[hyperopt] Running hyperopt for ${model_type} ..."
    (
        cd "${NEIMS_DIR}"
        uv run hyperopt.py \
            --model-type "${model_type}" \
            --metadata-path "${METADATA}" \
            --spectra-path "${SPECTRA}" \
            --splits-path "${SPLITS[scaffold]}" \
            --n-trials 20 \
            --output-dir "hyperopt_results_${model_type}"
    )
}

for model_type in "${MODEL_TYPES[@]}"; do
    run_hyperopt "${model_type}"
done

# Train + Predict + Eval for each (split, model_type, seed)
for split_name in scaffold random; do
    splits_path="${SPLITS[$split_name]}"
    cands_pickle="${CANDS_PICKLE[$split_name]}"

    if [[ "$split_name" == "scaffold" ]]; then
        eval_splits_flag="${DATA_DIR}/splits/scaffold_no_xeno_aas_deduplicated.tsv"
    else
        eval_splits_flag="${DATA_DIR}/splits/random_no_xeno_aas_deduplicated.tsv"
    fi

    for model_type in "${MODEL_TYPES[@]}"; do
        hopt_json="${NEIMS_DIR}/hyperopt_results_${model_type}/best_params.json"
        hparam_args=$(neims_hparam_args "${hopt_json}")

        seeds_for_split=("${SEEDS[@]}")
        [[ "$split_name" == "random" ]] && seeds_for_split=("${RANDOM_SEEDS[@]}")

        for seed in "${seeds_for_split[@]}"; do
            run_tag="${model_type}_${split_name}_s${seed}"
            ckpt_dir="${NEIMS_DIR}/outputs/${run_tag}"
            pred_dir="${NEIMS_DIR}/results/predictions"
            eval_dir="${REPO_ROOT}/results/eval/${run_tag}"

            mkdir -p "${pred_dir}" "${eval_dir}"

            # Train
            if [[ -f "${ckpt_dir}/best_model.pt" ]]; then
                echo "[train] ${run_tag}: checkpoint exists, skipping training."
            else
                echo "[train] ${run_tag}: training ..."
                (
                    cd "${NEIMS_DIR}"
                    eval uv run train.py \
                        --model-type "${model_type}" \
                        --metadata-path "${METADATA}" \
                        --spectra-path "${SPECTRA}" \
                        --splits-path "${splits_path}" \
                        --seed "${seed}" \
                        --output-dir "outputs/${run_tag}" \
                        --wandb-run-name "${run_tag}" \
                        "${hparam_args}"
                )
            fi

            # Predict: NIST test set
            test_pred="${pred_dir}/${run_tag}_test.hdf5"
            if [[ -f "${test_pred}" ]]; then
                echo "[predict/test] ${run_tag}: already exists, skipping."
            else
                echo "[predict/test] ${run_tag}: generating NIST test predictions ..."
                (
                    cd "${NEIMS_DIR}"
                    uv run predict.py \
                        --checkpoint "outputs/${run_tag}/best_model.pt" \
                        --metadata "${METADATA}" \
                        --spectra "${SPECTRA}" \
                        --splits "${splits_path}" \
                        --eval-split test \
                        --output "${test_pred}"
                )
            fi

            # Predict: PubChem formula candidates
            cands_pred="${pred_dir}/${run_tag}_pubchem_cands_50.hdf5"
            if [[ -f "${cands_pred}" ]]; then
                echo "[predict/cands] ${run_tag}: already exists, skipping."
            else
                echo "[predict/cands] ${run_tag}: generating PubChem candidate predictions ..."
                (
                    cd "${NEIMS_DIR}"
                    uv run predict.py \
                        --checkpoint "outputs/${run_tag}/best_model.pt" \
                        --candidates-pickle "${cands_pickle}" \
                        --output "${cands_pred}"
                )
            fi

            # Eval (similarity + retrieval)
            if [[ -f "${eval_dir}/similarity_results.csv" ]]; then
                echo "[eval] ${run_tag}: similarity_results.csv exists, skipping."
            else
                echo "[eval] ${run_tag}: running evaluation ..."
                (
                    cd "${REPO_ROOT}"
                    uv run src/icicle/eval_from_predictions.py \
                        --predictions "${cands_pred}" \
                        --test-predictions "${test_pred}" \
                        --ground-truth "${DATA_DIR}/spectra.hdf5" \
                        --labels "${DATA_DIR}/metadata.tsv" \
                        --splits "${eval_splits_flag}" \
                        --output "${eval_dir}" \
                        --mode all \
                        --candidates-pickle "${cands_pickle}" \
                        --formula-map "${FORMULA_MAP}"
                )
            fi

            # ── Extra test sets (similarity only) ──────────────────────────

            # MoNA (clean subset: no overlap with NIST scaffold train)
            mona_pred="${pred_dir}/${run_tag}_mona_test.hdf5"
            mona_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_mona"
            if [[ -f "${mona_eval_dir}/similarity_results.csv" ]]; then
                echo "[eval/mona] ${run_tag}: exists, skipping."
            else
                if [[ ! -f "${mona_pred}" ]]; then
                    echo "[predict/mona] ${run_tag}: generating MoNA predictions ..."
                    (
                        cd "${NEIMS_DIR}"
                        uv run predict.py \
                            --checkpoint "outputs/${run_tag}/best_model.pt" \
                            --metadata "${MONA_DIR}/metadata.tsv" \
                            --metadata-format mona \
                            --splits "${MONA_CLEAN_SPLIT}" \
                            --eval-split test \
                            --output "${mona_pred}"
                    )
                fi
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

            # Xeno amino acids (subset of NIST, uses xeno_amino_acids.tsv split)
            xeno_pred="${pred_dir}/${run_tag}_xeno_aas_test.hdf5"
            xeno_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_xeno_aas"
            if [[ -f "${xeno_eval_dir}/similarity_results.csv" ]]; then
                echo "[eval/xeno_aas] ${run_tag}: exists, skipping."
            else
                if [[ ! -f "${xeno_pred}" ]]; then
                    echo "[predict/xeno_aas] ${run_tag}: generating xeno-AA predictions ..."
                    (
                        cd "${NEIMS_DIR}"
                        uv run predict.py \
                            --checkpoint "outputs/${run_tag}/best_model.pt" \
                            --metadata "${METADATA}" \
                            --spectra "${SPECTRA}" \
                            --splits "${XENO_SPLIT}" \
                            --eval-split test \
                            --output "${xeno_pred}"
                    )
                fi
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

            # VGWD (raw MGF spectra, clean subset: no overlap with NIST scaffold train)
            vgwd_pred="${pred_dir}/${run_tag}_vgwd_test.hdf5"
            vgwd_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_vgwd"
            if [[ -f "${vgwd_eval_dir}/similarity_results.csv" ]]; then
                echo "[eval/vgwd] ${run_tag}: exists, skipping."
            else
                if [[ ! -f "${vgwd_pred}" ]]; then
                    echo "[predict/vgwd] ${run_tag}: generating VGWD predictions ..."
                    (
                        cd "${NEIMS_DIR}"
                        uv run predict.py \
                            --checkpoint "outputs/${run_tag}/best_model.pt" \
                            --metadata "${VGWD_DIR}/labels.tsv" \
                            --metadata-format vgwd \
                            --splits "${VGWD_CLEAN_SPLIT}" \
                            --eval-split test \
                            --output "${vgwd_pred}"
                    )
                fi
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

            echo "[done] ${run_tag}"
        done
    done
done

echo ""
echo "All runs complete. Results in ${REPO_ROOT}/results/eval/."
