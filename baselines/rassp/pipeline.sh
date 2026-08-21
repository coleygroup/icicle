#!/bin/bash

# # Dataset conversion

# python convert_hdf5_to_parquet.py \
#     --split-file /home/magled/icicle-dev/data/NIST2023_GCMS_main/splits/random.tsv \
#     --hdf5-path  /home/magled/icicle-dev/data/NIST2023_GCMS_main/spectra.hdf5 \
#     --name /home/magled/icicle-dev/baselines/rassp/nist-random

# python convert_hdf5_to_parquet.py \
#     --split-file /home/magled/icicle-dev/data/NIST2023_GCMS_main/splits/scaffold.tsv \
#     --hdf5-path  /home/magled/icicle-dev/data/NIST2023_GCMS_main/spectra.hdf5 \
#     --name /home/magled/icicle-dev/baselines/rassp/nist-scaffold

# python convert_hdf5_to_parquet.py \
#     --split-file /home/magled/icicle-dev/data/MoNA-export-GC-MS_Spectra/splits/random.tsv \
#     --hdf5-path  /home/magled/icicle-dev/data/MoNA-export-GC-MS_Spectra/spectra.hdf5 \
#     --name /home/magled/icicle-dev/baselines/rassp/mona-random

# # Hyperparameter tuning
USE_CUDA=1 CUDA_VISIBLE_DEVICES='1' python rassp/hyperopt.py rassp/expconfig/hyperopt.yaml
# use optuna delete-study --study-name rassp_hyperopt_nist_scaffold --storage sqlite:///hyperopt.db to delete previous study if needed

python rassp/generate_config_from_hyperopt.py \
    results/hyperopt/rassp_hyperopt_nist_scaffold_results.yaml \
    rassp/expconfig/hyperopt.yaml \
    -o rassp/expconfig/best_config.yaml \
    --max-epochs 100

# # Final training
USE_CUDA=1 CUDA_VISIBLE_DEVICES='1' python rassp/forward_train.py rassp/expconfig/best_config.yaml

# # Final eval / inference? TODO