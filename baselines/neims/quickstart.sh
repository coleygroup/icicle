
# Train with defaults

cd baselines/neims/

uv run hyperopt.py \
  --model-type neims_gnn \
  --metadata-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/metadata.tsv \
  --spectra-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/spectra.hdf5 \
  --splits-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
  --n-trials 20 \
  --output-dir hyperopt_results_neims_gnn

uv run hyperopt.py \
  --model-type neims \
  --metadata-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/metadata.tsv \
  --spectra-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/spectra.hdf5 \
  --splits-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
  --n-trials 20 \
  --output-dir hyperopt_results_neims

# for now, manually assign best hparameters from hyperopt_results/best_params.json to configs/default.yaml

# Train with best hyperparameters
uv run train.py \
  --metadata-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/metadata.tsv \
  --spectra-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/spectra.hdf5 \
  --splits-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
  --model-type neims

uv run train.py \
  --metadata-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/metadata.tsv \
  --spectra-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/spectra.hdf5 \
  --splits-path /home/magled/icicle-dev/data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
  --model-type neims_gnn