# Train AIRI models for different RI column types
# Requires masskit_ai conda environment

conda activate masskit_ai

python examples/scripts/retention_index/prepare_airi_training_data.py \
    --data-dir data/NIST2023_GCMS_main \
    --output-dir data/NIST2023_GCMS_main/airi_data_stdnp_random \
    --ri-type StdNP

python examples/scripts/retention_index/prepare_airi_training_data.py \
    --data-dir data/NIST2023_GCMS_main \
    --output-dir data/NIST2023_GCMS_main/airi_data_semistdnp_random \
    --ri-type SemiStdNP

python examples/scripts/retention_index/prepare_airi_training_data.py \
    --data-dir data/NIST2023_GCMS_main \
    --output-dir data/NIST2023_GCMS_main/airi_data_stdpolar_random \
    --ri-type StdPolar

python examples/scripts/retention_index/train_airi_models.py \
    --data-dir   data/NIST2023_GCMS_main/airi_data_semistdnp_random \
    --output-dir results/airi_models_semistdnp \
    --epochs 150 --batch-size 64

python examples/scripts/retention_index/train_airi_models.py \
    --data-dir   data/NIST2023_GCMS_main/airi_data_stdnp_random \
    --output-dir results/airi_models_stdnp \
    --epochs 150 --batch-size 64

python examples/scripts/retention_index/train_airi_models.py \
    --data-dir   data/NIST2023_GCMS_main/airi_data_stdpolar_random \
    --output-dir results/airi_models_stdpolar \
    --epochs 150 --batch-size 64