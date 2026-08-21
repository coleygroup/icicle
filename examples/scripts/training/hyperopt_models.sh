#!/bin/bash

DATA_DIR="/home/magled/orcd/scratch/icicle-data/NIST2023_GCMS_main/"
SPLIT="scaffold_no_xeno_aas_deduplicated"
STUDY_NAME="icicle_sweep"

# uv sync --extra cu124

# sbatch CLI args override #SBATCH directives in submit_slurm_job.sh
sbatch \
    --job-name=hyperopt_icicle \
    --time=48:00:00 \
    --partition=pi_ccoley \
    ./examples/scripts/training/submit_slurm_job.sh uv run --no-sync src/icicle/train.py \
    data.data_dir=$DATA_DIR \
    data=NIST \
    data.split_name=$SPLIT \
    data.training_data_fraction=0.1 \
    model=intensity_predictor \
    model.architecture.h_shift_range=6 \
    model.architecture.add_isotopes=true \
    system.debug=false \
    hyperparameter_sweep=default \
    "hydra.run.dir=results/multi_run/\${SLURM_JOB_ID}" \
    --multirun


# # local
# uv run src/icicle/train.py \
#     data.data_dir=/home/magled/icicle-dev/data/NIST2023_GCMS_main/ \
#     data=NIST \
#     data.split_name=scaffold_no_xeno_aas_deduplicated \
#     data.training_data_fraction=0.1 \
#     model=intensity_predictor \
#     model.architecture.h_shift_range=6 \
#     model.architecture.add_isotopes=true \
#     system.debug=false \
#     hyperparameter_sweep=default \
#     --multirun