#!/bin/bash
# evaluate_all_baselines.sh
#
# Runs all baseline evaluations across all datasets and eval modes.
#
# NATIVE BASELINES (random, average, full_enumeration_barcode)
# ─────────────────────────────────────────────────────────────
# Run directly via eval.py.  Results land in:
#
#   results/eval/<SLURM_JOB_ID|local>_<description>/
#     ├── similarity_results.csv          # per-molecule similarity metrics
#     ├── similarity_summary_metrics.txt  # aggregate stats + timing
#     ├── all_evaluation_spectra.hdf5     # predicted + ground-truth spectra
#     ├── retrieval_with_formula_results.csv          (if retrieval_formula)
#     ├── retrieval_with_ri_<type>_results.csv        (if retrieval_ri)
#     └── evaluation.log
#
# EXTERNAL BASELINES (NEIMS, RASSP, MassFormer)
# ──────────────────────────────────────────────
# Two-step process (see section at bottom of this script):
#   Step 1  Run inference in the baseline's own conda env -> predictions HDF5
#   Step 2  Run eval_from_predictions.py in the ICICLE env -> metrics CSVs
#
# Predictions are saved to:
#   results/predictions/<baseline>_<dataset>_<split>.hdf5
#
# Eval outputs land in:
#   results/eval/<SLURM_JOB_ID|local>_<description>/
#     ├── similarity_results.csv
#     ├── retrieval_results.csv
#     └── ...
#
# USAGE
# ──────
#   bash examples/scripts/evaluation/evaluate_all_baselines.sh
#   # or, to submit via SLURM: set USE_SLURM=true below

# Fire-and-forget: does NOT use `set -e` for the local (non-SLURM) path --
# one entry in EVAL_MATRIX failing must not abort the rest of the matrix.
set -uo pipefail

# ─── USER CONFIG ──────────────────────────────────────────────────────────────

FRACTION=1.0          # 0.1 for a quick sanity-check, 1.0 for full evaluation
USE_SLURM=true        # true -> sbatch; false -> run locally (blocks until done)
TIME_SIM="4:00:00"    # wall time for similarity jobs
TIME_RETR="24:00:00"  # wall time for retrieval jobs

NIST_DATA_DIR="/home/magled/orcd/pool/data/NIST2023_GCMS_main/"

# ─── EVAL MATRIX ──────────────────────────────────────────────────────────────
# Each entry: "model_type|dataset|data_dir|split|eval_mode|description"
#
# model_type  : random, average, full_enumeration_barcode
# dataset     : NIST
# split       : scaffold_no_xeno_aas_deduplicated, random_no_xeno_aas_deduplicated,
#               all_test_no_exact_matches, all_test_no_scaffold_matches
# eval_mode   : similarity, retrieval_formula, retrieval_ri
# description : short label used in the output dir name (keep it filesystem-safe)

EVAL_MATRIX=(
    # ── random ──────────────────────────────────────────────────────────────
    "random|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|similarity|random_nist_scaffold_sim"
    "random|NIST|${NIST_DATA_DIR}|random_no_xeno_aas_deduplicated|similarity|random_nist_random_sim"
    "random|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|retrieval_formula|random_nist_scaffold_fretr"
    "random|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|retrieval_ri|random_nist_scaffold_riretr"

    # ── average ─────────────────────────────────────────────────────────────
    # Average predicts the same spectrum for all queries; only similarity is meaningful.
    "average|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|similarity|average_nist_scaffold_sim"
    "average|NIST|${NIST_DATA_DIR}|random_no_xeno_aas_deduplicated|similarity|average_nist_random_sim"

    # ── nearest_neighbor ────────────────────────────────────────────────────
    "nearest_neighbor|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|similarity|nn_nist_scaffold_sim"
    "nearest_neighbor|NIST|${NIST_DATA_DIR}|random_no_xeno_aas_deduplicated|similarity|nn_nist_random_sim"
    "nearest_neighbor|NIST|${NIST_DATA_DIR}|random_no_xeno_aas_deduplicated_no_qcxms2_rassp|similarity|nn_random_rassp_subset_sim"
    "nearest_neighbor|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated_rassp|similarity|nn_scaffold_rassp_subset_sim"

    # ── full_enumeration_barcode ─────────────────────────────────────────────
    "full_enumeration_barcode|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|similarity|febc_nist_scaffold_sim"
    "full_enumeration_barcode|NIST|${NIST_DATA_DIR}|random_no_xeno_aas_deduplicated|similarity|febc_nist_random_sim"
    "full_enumeration_barcode|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|retrieval_formula|febc_nist_scaffold_fretr"
    "full_enumeration_barcode|NIST|${NIST_DATA_DIR}|scaffold_no_xeno_aas_deduplicated|retrieval_ri|febc_nist_scaffold_riretr"
)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

get_eval_flags() {
    local eval_mode="$1"
    case "$eval_mode" in
        similarity)        echo "eval.similarity.enable=True" ;;
        retrieval_formula) echo "eval.retrieval_with_formula.enable=True" ;;
        retrieval_ri)      echo "eval.retrieval_with_ri.enable=True" ;;
        *) echo "Unknown eval mode: $eval_mode" >&2; exit 1 ;;
    esac
}

get_expected_csv() {
    case "$1" in
        similarity)        echo "similarity_results.csv" ;;
        retrieval_formula) echo "retrieval_with_formula_results.csv" ;;
        retrieval_ri)      echo "retrieval_with_ri_StdNP_results.csv" ;;
    esac
}

run_eval() {
    local model_type="$1"
    local dataset="$2"
    local data_dir="$3"
    local split="$4"
    local eval_mode="$5"
    local description="$6"

    local eval_flags
    eval_flags=$(get_eval_flags "$eval_mode")

    local out_dir="results/eval/${SLURM_JOB_ID:-local}_${description}"
    local expected_csv
    expected_csv=$(get_expected_csv "$eval_mode")
    if [[ "$USE_SLURM" != "true" && -f "${out_dir}/${expected_csv}" ]]; then
        echo ">>> ${description}: ${expected_csv} already exists -- skipping."
        return 0
    fi

    local cmd="uv run --no-sync src/icicle/eval.py"
    cmd+=" data=${dataset}"
    cmd+=" data.data_dir=${data_dir}"
    cmd+=" data.split_name=${split}"
    cmd+=" eval=${model_type}"
    cmd+=" ${eval_flags}"

    if [[ "$FRACTION" != "1.0" && "$FRACTION" != "1" ]]; then
        cmd+=" eval.fraction_of_spectra_to_compute=${FRACTION}"
    fi

    # Output dir: results/eval/<SLURM_JOB_ID>_<description>
    cmd+=" hydra.run.dir=${out_dir}"

    echo ">>> ${description}"
    echo "    ${cmd}"

    if [[ "$USE_SLURM" == "true" ]]; then
        local time_limit="$TIME_SIM"
        [[ "$eval_mode" != "similarity" ]] && time_limit="$TIME_RETR"
        sbatch --job-name="eval_${description}" --time="${time_limit}" \
            ./examples/scripts/training/submit_slurm_job.sh "$cmd"
    else
        # Isolate failures: one bad entry must not stop the rest of the matrix.
        eval "$cmd"
        local exit_code=$?
        if [ -f "${out_dir}/${expected_csv}" ]; then
            echo ">>> ${description}: done."
        else
            echo ">>> ${description}: exited with code ${exit_code}, no ${expected_csv} written -- moving on."
        fi
    fi
}

# ─── NATIVE BASELINES ─────────────────────────────────────────────────────────

echo " Baseline Evaluation Script"
echo " Fraction : ${FRACTION}"
echo " SLURM    : ${USE_SLURM}"
echo ""
echo "── Native baselines (random / average / febc) ──"

for entry in "${EVAL_MATRIX[@]}"; do
    IFS='|' read -r model_type dataset data_dir split eval_mode description <<< "$entry"
    run_eval "$model_type" "$dataset" "$data_dir" "$split" "$eval_mode" "$description"
done

# ─── EXTERNAL BASELINES (two-step) ────────────────────────────────────────────
#
# For NEIMS, RASSP, and MassFormer you must first produce an HDF5 predictions
# file in the baseline's own environment, then evaluate it in the ICICLE env.
#
# The commands below are PRINTED as instructions, not executed automatically,
# because they require a different conda/pip environment.
#
# Set the paths below and uncomment the eval step to automate step 2.

NEIMS_CKPT="baselines/neims/outputs/best_model.pt"
RASSP_CKPT="baselines/rassp/checkpoints/best_config.<timestamp>.00000000.model"
RASSP_META="baselines/rassp/checkpoints/best_config.<timestamp>.meta"
MASSFORMER_CFG="baselines/massformer/config/inference_nist23.yml"

NIST_SPECTRA="data/NIST2023_GCMS_main/spectra.hdf5"
NIST_LABELS="data/NIST2023_GCMS_main/metadata.tsv"
NIST_SPLIT_SCAFFOLD="data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv"

echo ""
echo "── External baselines: manual steps required ──"
cat <<'INSTRUCTIONS'

╔══════════════════════════════════════════════════════════════════════════════╗
║  NEIMS  (run in the NEIMS Python env, then eval back in ICICLE env)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

# Step 0 — train NEIMS (skip if already trained)
cd baselines/neims
python train.py --config configs/neims_nist.yaml

# Step 1 — predict -> HDF5
cd baselines/neims
python predict.py \
    --checkpoint outputs/best_model.pt \
    --metadata   ../../data/NIST2023_GCMS_main/metadata.tsv \
    --spectra    ../../data/NIST2023_GCMS_main/spectra.hdf5 \
    --splits     ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output     outputs/predictions_scaffold_test.hdf5 \
    --eval-split test
# -> saves: baselines/neims/outputs/predictions_scaffold_test.hdf5

# Step 2 — evaluate (from repo root, ICICLE env)
uv run src/icicle/eval_from_predictions.py \
    --predictions baselines/neims/outputs/predictions_scaffold_test.hdf5 \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/neims_nist_scaffold \
    --mode all
# -> saves: results/eval/neims_nist_scaffold/similarity_results.csv
#          results/eval/neims_nist_scaffold/retrieval_results.csv

╔══════════════════════════════════════════════════════════════════════════════╗
║  RASSP  (run in rassp conda env, then eval back in ICICLE env)              ║
╚══════════════════════════════════════════════════════════════════════════════╝

# Step 1 — predict -> HDF5  (rassp conda env)
conda activate rassp
cd baselines/rassp
python scripts/run_inference_for_eval.py \
    --checkpoint checkpoints/best_config.<timestamp>.00000000.model \
    --meta       checkpoints/best_config.<timestamp>.meta \
    --metadata   ../../data/NIST2023_GCMS_main/metadata.tsv \
    --splits     ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output     results/rassp_nist23_scaffold.hdf5 \
    --eval-split test \
    --gpu
# -> saves: baselines/rassp/results/rassp_nist23_scaffold.hdf5

# Step 2 — evaluate (from repo root, ICICLE env)
uv run src/icicle/eval_from_predictions.py \
    --predictions baselines/rassp/results/rassp_nist23_scaffold.hdf5 \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/rassp_nist_scaffold \
    --mode all
# -> saves: results/eval/rassp_nist_scaffold/similarity_results.csv
#          results/eval/rassp_nist_scaffold/retrieval_results.csv

╔══════════════════════════════════════════════════════════════════════════════╗
║  MassFormer  (massformer conda env, then eval back in ICICLE env)           ║
╚══════════════════════════════════════════════════════════════════════════════╝

# Step 1 — predict -> HDF5  (massformer conda env)
cd baselines/massformer
python scripts/run_inference_for_eval.py \
    --config config/inference_nist23.yml \
    --output results/predictions/massformer_nist_scaffold.hdf5
# -> saves: baselines/massformer/results/predictions/massformer_nist_scaffold.hdf5

# Step 2 — evaluate (from repo root, ICICLE env)
uv run src/icicle/eval_from_predictions.py \
    --predictions baselines/massformer/results/predictions/massformer_nist_scaffold.hdf5 \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/massformer_nist_scaffold \
    --mode all
# -> saves: results/eval/massformer_nist_scaffold/similarity_results.csv
#          results/eval/massformer_nist_scaffold/retrieval_results.csv

INSTRUCTIONS

echo ""
echo "All native baseline jobs submitted."
echo "See INSTRUCTIONS above for external baseline (NEIMS/RASSP/MassFormer) steps."
