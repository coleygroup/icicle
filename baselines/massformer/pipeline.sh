
# Data processing
python preproc_scripts/convert_nist23_gcms.py \
  --hdf5-path ../../data/NIST2023_GCMS_main/spectra.hdf5 \
  --splits-path ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
  --output-dir baselines/massformer/data/proc/nist23

# Hyperparameter optimization


# Training
uv run python scripts/run_train_eval.py \
  --template_fp config/template.yml \
  --custom_fp config/train_nist23_gcms.yml \
  --wandb_mode online \
  --device_id 0 \
  --num_seeds 3

# Inference
uv run python scripts/run_inference.py \
  --checkpoint_path <path_to_checkpoint.pt> \
  --test_data_path data/proc/nist23/ \
  --output_path predictions/