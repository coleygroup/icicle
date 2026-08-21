#!/usr/bin/env bash
# Full MassFormer training + prediction + evaluation pipeline.
# Runs for scaffold and random splits, seeds 1/2/3.
# Must be run from the repo root (icicle-dev/).
#
# Usage:
#   bash baselines/massformer/run_full_pipeline.sh
#
# Prerequisites:
#   - conda env "MF-GPU" installed (see baselines/massformer/env/requirements-gpu.txt)
#
# Environment split:
#   Hyperopt, training, inference: /home/magled/miniconda3/envs/MF-GPU/bin/python (direct)
#   Evaluation:                    uv run (ICICLE env, from repo root)
#
# Hyperopt is run once on the scaffold split; its hparams are reused for random.
#
# Proc data is split-specific (split assignments are baked in):
#   data/proc/nist23_scaffold/   ← scaffold_no_xeno_aas_deduplicated
#   data/proc/nist23_random/     ← random_no_xeno_aas_deduplicated
#
# Checkpoint detection for parallel safety:
#   Before training, snapshot the set of existing wandb run dirs. After training,
#   find the new dir and locate chkpt.pkl within it. This is safe across parallel
#   seed runs because each training process creates its own wandb run dir.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MF_DIR="${REPO_ROOT}/baselines/massformer"
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
    [random]="${DATA_DIR}/splits/random_no_xeno_aas_deduplicated_no_qcxms2.tsv"
)
declare -A CANDS_PICKLE=(
    [scaffold]="${DATA_DIR}/retrieval/cands_pickled_scaffold_50_compat.pkl"
    [random]="${DATA_DIR}/retrieval/cands_pickled_random_50_compat.pkl"
)
declare -A PROC_DIR=(
    [scaffold]="${MF_DIR}/data/proc/nist23_scaffold"
    [random]="${MF_DIR}/data/proc/nist23_random"
)

SEEDS=(1 3)
HYPEROPT_CONFIG="${MF_DIR}/config/hyperopt_nist23_gcms.yml"
HYPEROPT_RESULTS_DIR="${MF_DIR}/results/hyperopt"
HYPEROPT_RESULTS_YAML="${HYPEROPT_RESULTS_DIR}/massformer_hyperopt_scaffold_results.yaml"

MF_PYTHON="/home/magled/miniconda3/envs/MF-GPU/bin/python"
TEMPLATE_CFG="${MF_DIR}/config/template.yml"
TRAIN_CFG="${MF_DIR}/config/train_nist23_gcms.yml"

# Extract best hyperparameters from YAML and write a patched custom config.
# Args: $1 = hyperopt results YAML, $2 = split_name, $3 = seed,
#       $4 = output config path, $5 = proc_dir for this split
generate_seed_config() {
    local hopt_yaml="$1"
    local split_name="$2"
    local seed="$3"
    local out_cfg="$4"
    local proc_dp="$5"

    "${MF_PYTHON}" - <<PYEOF
import yaml

with open("${hopt_yaml}") as f:
    hopt = yaml.safe_load(f)

with open("${TRAIN_CFG}") as f:
    cfg = yaml.safe_load(f)

best = hopt.get("best_params")
if not best:
    raise RuntimeError(f"No best_params found in ${hopt_yaml} — did hyperopt complete successfully?")

model_keys = {"ff_h_dim", "ff_num_layers", "ff_skip", "dropout"}
run_keys = {
    "learning_rate", "batch_size", "weight_decay", "scheduler",
    "scheduler_peak_lr", "scheduler_warmup_frac",
    "flag", "flag_m", "flag_step_size", "flag_mag",
}

for k, v in best.items():
    if k in model_keys:
        cfg["model"][k] = v
    elif k in run_keys:
        cfg["run"][k] = v

cfg["data"]["proc_dp"] = "${proc_dp}"
cfg["run"]["train_seed"] = ${seed}
cfg["run"]["split_key"] = "${split_name}"
cfg["run"]["split_seed"] = 420

with open("${out_cfg}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"Config written to ${out_cfg}")
PYEOF
}

# Record the set of existing wandb run dirs into a temp file before training.
# Returns the path of the snapshot file.
snapshot_wandb_runs() {
    local snapshot_file
    snapshot_file=$(mktemp)
    find "${MF_DIR}/wandb" -maxdepth 1 -name "run-*" -type d 2>/dev/null \
        | sort > "${snapshot_file}"
    echo "${snapshot_file}"
}

# After training, find the new wandb run dir (not in the pre-training snapshot)
# and locate chkpt.pkl within it. Symlinks it to checkpoints/<run_tag>/chkpt.pkl.
# Args: $1 = run_tag, $2 = snapshot file from snapshot_wandb_runs()
link_new_checkpoint() {
    local run_tag="$1"
    local snapshot_file="$2"
    local ckpt_dir="${MF_DIR}/checkpoints/${run_tag}"
    mkdir -p "${ckpt_dir}"

    # Find run dirs that didn't exist before training started
    local new_run_dir
    new_run_dir=$(find "${MF_DIR}/wandb" -maxdepth 1 -name "run-*" -type d 2>/dev/null \
        | sort \
        | comm -23 - "${snapshot_file}" \
        | head -1)

    if [[ -z "${new_run_dir}" ]]; then
        echo "[ckpt] ERROR: no new wandb run dir found after training ${run_tag}" >&2
        rm -f "${snapshot_file}"
        return 1
    fi

    local chkpt_pkl="${new_run_dir}/files/chkpt.pkl"
    if [[ ! -f "${chkpt_pkl}" ]]; then
        echo "[ckpt] ERROR: chkpt.pkl not found in ${new_run_dir}/files/" >&2
        rm -f "${snapshot_file}"
        return 1
    fi

    ln -sf "${chkpt_pkl}" "${ckpt_dir}/chkpt.pkl"
    echo "[ckpt] ${run_tag}: linked ${chkpt_pkl} → ${ckpt_dir}/chkpt.pkl"
    rm -f "${snapshot_file}"
}

# ── Data preprocessing (per split) ───────────────────────────────────────────
for split_name in random; do
    proc_dir="${PROC_DIR[$split_name]}"
    if [[ -d "${proc_dir}" && -n "$(ls -A "${proc_dir}" 2>/dev/null)" ]]; then
        echo "[preproc] ${split_name}: proc dir already exists, skipping conversion."
    else
        echo "[preproc] ${split_name}: converting NIST HDF5 → MassFormer proc format ..."
        mkdir -p "${proc_dir}"
        (
            cd "${MF_DIR}"
            "${MF_PYTHON}" preproc_scripts/convert_nist23_gcms.py \
                --hdf5-path "${SPECTRA}" \
                --splits-path "${SPLITS[$split_name]}" \
                --output-dir "${proc_dir}"
        )
    fi
done

# ── Hyperopt (scaffold split, once) ──────────────────────────────────────────
if [[ -f "${HYPEROPT_RESULTS_YAML}" ]]; then
    echo "[hyperopt] scaffold: results already exist, skipping."
else
    echo "[hyperopt] scaffold: running hyperopt ..."
    mkdir -p "${HYPEROPT_RESULTS_DIR}"
    (
        cd "${MF_DIR}"
        "${MF_PYTHON}" scripts/hyperopt.py "${HYPEROPT_CONFIG}" \
            --output-dir "${HYPEROPT_RESULTS_DIR}" \
            --study-name "massformer_hyperopt_scaffold" \
            --no-wandb
    )
fi

# ── Per-(split, seed): generate config, train, predict, eval ─────────────────
for split_name in random; do
    splits_path="${SPLITS[$split_name]}"
    cands_pickle="${CANDS_PICKLE[$split_name]}"
    proc_dir="${PROC_DIR[$split_name]}"

    for seed in "${SEEDS[@]}"; do
        run_tag="massformer_${split_name}_s${seed}"
        seed_cfg="${MF_DIR}/config/train_nist23_gcms_${run_tag}.yml"
        ckpt_dir="${MF_DIR}/checkpoints/${run_tag}"
        pred_dir="${MF_DIR}/results/predictions"
        eval_dir="${REPO_ROOT}/results/eval/${run_tag}"

        mkdir -p "${pred_dir}" "${eval_dir}"

        # Generate per-seed training config
        if [[ ! -f "${seed_cfg}" ]]; then
            echo "[config] ${run_tag}: generating training config ..."
            generate_seed_config \
                "${HYPEROPT_RESULTS_YAML}" "${split_name}" "${seed}" \
                "${seed_cfg}" "${proc_dir}"
        else
            echo "[config] ${run_tag}: config already exists, skipping."
        fi

        # Train
        if [[ -L "${ckpt_dir}/chkpt.pkl" || -f "${ckpt_dir}/chkpt.pkl" ]]; then
            echo "[train] ${run_tag}: checkpoint exists, skipping training."
        else
            echo "[train] ${run_tag}: training ..."
            snapshot=$(snapshot_wandb_runs)
            (
                cd "${MF_DIR}"
                "${MF_PYTHON}" scripts/run_train_eval.py \
                    --template_fp "${TEMPLATE_CFG}" \
                    --custom_fp "${seed_cfg}" \
                    --wandb_mode online \
                    --wandb_meta_dp "${MF_DIR}" \
                    --device_id 0
            )
            link_new_checkpoint "${run_tag}" "${snapshot}"
        fi

        ckpt_pkl="${ckpt_dir}/chkpt.pkl"

        # Build per-run inference config (points to this seed's checkpoint + splits)
        inference_cfg="${MF_DIR}/config/inference_${run_tag}.yml"
        if [[ ! -f "${inference_cfg}" ]]; then
            cat > "${inference_cfg}" <<INFCFG
massformer:
  template_config: "${TEMPLATE_CFG}"
  custom_config: "${seed_cfg}"
  checkpoint_path: "${ckpt_pkl}"
  device_id: 0

data:
  labels_path: "${METADATA}"
  splits_path: "${splits_path}"
  min_mz: 0
  max_mz: 750
  bin_width: 1.0
INFCFG
        fi

        # Predict: NIST test set
        test_pred="${pred_dir}/${run_tag}_test.hdf5"
        if [[ -f "${test_pred}" ]]; then
            echo "[predict/test] ${run_tag}: already exists, skipping."
        else
            echo "[predict/test] ${run_tag}: generating NIST test predictions ..."
            (
                cd "${MF_DIR}"
                "${MF_PYTHON}" scripts/run_inference_for_eval.py \
                    --config "${inference_cfg}" \
                    --output "${test_pred}" \
                    --split test
            )
        fi

        # Predict: PubChem formula candidates
        cands_pred="${pred_dir}/${run_tag}_pubchem_cands_50.hdf5"
        if [[ -f "${cands_pred}" ]]; then
            echo "[predict/cands] ${run_tag}: already exists, skipping."
        else
            echo "[predict/cands] ${run_tag}: generating PubChem candidate predictions ..."
            (
                cd "${MF_DIR}"
                "${MF_PYTHON}" scripts/run_inference_for_eval.py \
                    --config "${inference_cfg}" \
                    --output "${cands_pred}" \
                    --candidates-pickle "${cands_pickle}"
            )
        fi

        # Eval: NIST similarity (ICICLE env)
        if [[ -f "${eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/nist-sim] ${run_tag}: similarity_results.csv exists, skipping."
        else
            echo "[eval/nist-sim] ${run_tag}: running NIST similarity evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${test_pred}" \
                    --ground-truth "${DATA_DIR}/spectra.hdf5" \
                    --labels "${METADATA}" \
                    --splits "${splits_path}" \
                    --output "${eval_dir}" \
                    --mode similarity
            )
        fi

        # Eval: PubChem formula retrieval (ICICLE env)
        if [[ -f "${eval_dir}/retrieval_with_formula_results.csv" ]]; then
            echo "[eval/retrieval] ${run_tag}: retrieval_with_formula_results.csv exists, skipping."
        else
            echo "[eval/retrieval] ${run_tag}: running retrieval evaluation ..."
            (
                cd "${REPO_ROOT}"
                uv run src/icicle/eval_from_predictions.py \
                    --predictions "${cands_pred}" \
                    --test-predictions "${test_pred}" \
                    --ground-truth "${DATA_DIR}/spectra.hdf5" \
                    --labels "${METADATA}" \
                    --splits "${splits_path}" \
                    --output "${eval_dir}" \
                    --mode retrieval \
                    --candidates-pickle "${cands_pickle}" \
                    --formula-map "${FORMULA_MAP}"
            )
        fi

        # ── Extra test sets (similarity only) ──────────────────────────

        # MoNA (clean subset: no overlap with NIST scaffold train)
        mona_pred="${pred_dir}/${run_tag}_mona_test.hdf5"
        mona_eval_dir="${REPO_ROOT}/results/eval/${run_tag}_mona"
        mona_inf_cfg="${MF_DIR}/config/inference_${run_tag}_mona.yml"
        if [[ -f "${mona_eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/mona] ${run_tag}: exists, skipping."
        else
            if [[ ! -f "${mona_inf_cfg}" ]]; then
                cat > "${mona_inf_cfg}" <<INFCFG
massformer:
  template_config: "${TEMPLATE_CFG}"
  custom_config: "${seed_cfg}"
  checkpoint_path: "${ckpt_pkl}"
  device_id: 0

data:
  labels_path: "${MONA_DIR}/metadata.tsv"
  splits_path: "${MONA_CLEAN_SPLIT}"
  min_mz: 0
  max_mz: 750
  bin_width: 1.0
INFCFG
            fi
            if [[ ! -f "${mona_pred}" ]]; then
                echo "[predict/mona] ${run_tag}: generating MoNA predictions ..."
                (
                    cd "${MF_DIR}"
                    "${MF_PYTHON}" scripts/run_inference_for_eval.py \
                        --config "${mona_inf_cfg}" \
                        --output "${mona_pred}" \
                        --split test
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
        xeno_inf_cfg="${MF_DIR}/config/inference_${run_tag}_xeno_aas.yml"
        if [[ -f "${xeno_eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/xeno_aas] ${run_tag}: exists, skipping."
        else
            if [[ ! -f "${xeno_inf_cfg}" ]]; then
                cat > "${xeno_inf_cfg}" <<INFCFG
massformer:
  template_config: "${TEMPLATE_CFG}"
  custom_config: "${seed_cfg}"
  checkpoint_path: "${ckpt_pkl}"
  device_id: 0

data:
  labels_path: "${METADATA}"
  splits_path: "${XENO_SPLIT}"
  min_mz: 0
  max_mz: 750
  bin_width: 1.0
INFCFG
            fi
            if [[ ! -f "${xeno_pred}" ]]; then
                echo "[predict/xeno_aas] ${run_tag}: generating xeno-AA predictions ..."
                (
                    cd "${MF_DIR}"
                    "${MF_PYTHON}" scripts/run_inference_for_eval.py \
                        --config "${xeno_inf_cfg}" \
                        --output "${xeno_pred}" \
                        --split test
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
        vgwd_inf_cfg="${MF_DIR}/config/inference_${run_tag}_vgwd.yml"
        if [[ -f "${vgwd_eval_dir}/similarity_results.csv" ]]; then
            echo "[eval/vgwd] ${run_tag}: exists, skipping."
        else
            if [[ ! -f "${vgwd_inf_cfg}" ]]; then
                cat > "${vgwd_inf_cfg}" <<INFCFG
massformer:
  template_config: "${TEMPLATE_CFG}"
  custom_config: "${seed_cfg}"
  checkpoint_path: "${ckpt_pkl}"
  device_id: 0

data:
  labels_path: "${VGWD_DIR}/labels.tsv"
  splits_path: "${VGWD_CLEAN_SPLIT}"
  min_mz: 0
  max_mz: 750
  bin_width: 1.0
INFCFG
            fi
            if [[ ! -f "${vgwd_pred}" ]]; then
                echo "[predict/vgwd] ${run_tag}: generating VGWD predictions ..."
                (
                    cd "${MF_DIR}"
                    "${MF_PYTHON}" scripts/run_inference_for_eval.py \
                        --config "${vgwd_inf_cfg}" \
                        --output "${vgwd_pred}" \
                        --split test
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

echo ""
echo "All runs complete. Results in ${REPO_ROOT}/results/eval/."
