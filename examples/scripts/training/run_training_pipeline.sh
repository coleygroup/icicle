#!/usr/bin/env bash
# Training pipeline for icicle: submit one SLURM job per (run_tag, seed).
# Skips runs where best_model.pt already exists.
# Logs submitted job IDs to results/train/job_manifest.tsv.
#
# Usage (from repo root):
#   bash examples/scripts/training/run_training_pipeline.sh
#
# To add a variant: add an entry to RUN_CONFIGS below.
# Hyperopt is assumed done (see hyperopt_models.sh).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SUBMIT="${REPO_ROOT}/examples/scripts/training/submit_slurm_job.sh"
DATA_DIR="/home/magled/orcd/pool/data/NIST2023_GCMS_main"
SEEDS=(1 2 3)

MANIFEST="${REPO_ROOT}/results/train/job_manifest.tsv"
mkdir -p "$(dirname "${MANIFEST}")"
[[ -f "${MANIFEST}" ]] || echo -e "run_tag\tseed\tslurm_job_id\toutput_dir" > "${MANIFEST}"

# ---------------------------------------------------------------------------
# Run configurations: each entry is "<run_tag>|<split>|<extra hydra overrides>"
#
# split values: scaffold_no_xeno_aas_deduplicated  /  random_no_xeno_aas_deduplicated
# extra overrides: space-separated hydra key=value pairs (can be empty)
# ---------------------------------------------------------------------------
RUN_CONFIGS=(
    # tag                              | split     | extra overrides
    # "entropy_scaffold                  | scaffold  | model.architecture.loss_fn=entropy data.training_data_fraction=1.0"
    # "weighted_cosine_scaffold          | scaffold  | model.architecture.loss_fn=weighted_cosine_nist_gc data.training_data_fraction=1.0"
    # "composite_weighted_cosine_scaffold| scaffold  | model.architecture.loss_fn=composite_weighted_cosine_nist_gc data.training_data_fraction=1.0"
    # "entropy_scaffold_10pct            | scaffold  | model.architecture.loss_fn=entropy data.training_data_fraction=0.1"
    # "entropy_scaffold_1pct             | scaffold  | model.architecture.loss_fn=entropy data.training_data_fraction=0.01"
    # "entropy_scaffold_noiso            | scaffold  | model.architecture.loss_fn=entropy data.training_data_fraction=1.0 model.architecture.add_isotopes=false"
    "entropy_random                    | random    | model.architecture.loss_fn=entropy data.training_data_fraction=1.0"
)

for entry in "${RUN_CONFIGS[@]}"; do
    IFS='|' read -r run_tag split extra_overrides <<< "${entry}"
    run_tag="${run_tag// /}"   # strip surrounding whitespace
    split="${split// /}"
    extra_overrides="${extra_overrides# }"

    split_name="${split}_no_xeno_aas_deduplicated_no_qcxms2"

    for seed in "${SEEDS[@]}"; do
        full_tag="${run_tag}_s${seed}"
        out_dir="${REPO_ROOT}/results/train/${full_tag}"

        if [[ -f "${out_dir}/best_model.pt" ]]; then
            echo "[skip]   ${full_tag}: checkpoint exists."
            continue
        fi

        job_id=$(sbatch \
            --job-name="${full_tag}" \
            --parsable \
            "${SUBMIT}" \
            uv run --no-sync src/icicle/train.py \
                data.data_dir="${DATA_DIR}" \
                data=NIST \
                data.split_name="${split_name}" \
                model=intensity_predictor \
                system.seed="${seed}" \
                "hydra.run.dir=${out_dir}" \
                ${extra_overrides})

        echo "[submit] ${full_tag} → job ${job_id}  (${out_dir})"
        echo -e "${full_tag}\t${seed}\t${job_id}\t${out_dir}" >> "${MANIFEST}"
    done
done

echo ""
echo "All jobs submitted. Manifest: ${MANIFEST}"
echo "Track with: squeue -u \$USER"
