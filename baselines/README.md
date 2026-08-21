# Baseline Models

This directory contains three baseline models for comparison with ICICLE: **NEIMS**, **RASSP**, and **MassFormer**.

All baselines follow the same two-phase evaluation workflow:
1. **Predict** (in the baseline's own conda env) -> save predictions to an HDF5 file
2. **Evaluate** (in the ICICLE env) -> run `eval_from_predictions.py` on that HDF5 file

This ensures fair, identical metric computation across all models regardless of their Python/CUDA environment.

NIST_META="data/NIST2023_GCMS_main/metadata.tsv"
NIST_SPECTRA="data/NIST2023_GCMS_main/spectra.hdf5"
NIST_SPLIT="data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv"

---

## Unified Evaluation Command

Once you have a predictions HDF5 file from any baseline, run from the repo root:

```bash
uv run src/icicle/eval_from_predictions.py \
    --predictions <path/to/predictions.hdf5> \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/<model_name> \
    --mode all
```

Outputs: `similarity_results.csv`, `similarity_summary.txt`, `retrieval_results.csv`, `retrieval_summary.txt`.

---

## HDF5 Predictions Format

All inference scripts produce HDF5 files with this structure:
Outputs: `similarity_results.csv`, `similarity_summary.txt`, `retrieval_results.csv`, `retrieval_summary.txt`.

---

## HDF5 Predictions Format

All inference scripts produce HDF5 files with this structure:

```
/<mol_id>/
    attrs:
        smiles:    "CCO"
        inchi_key: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    datasets:
        predicted_intensities: float32[num_bins]
        mz_bins:               float32[num_bins]
file-level attrs:
    min_mz, max_mz, bin_width
```

---

## NEIMS

NEIMS (Neural Electron-Ionization Mass Spectrometry) is an ECFP-based MLP predictor.

### Installation

```bash
cd baselines/neims
pip install -e .
```

### 1. Hyperparameter optimization (optional)

```bash
cd baselines/neims

python hyperopt.py \
    --metadata-path ../../data/NIST2023_GCMS_main/metadata.tsv \
    --spectra-path  ../../data/NIST2023_GCMS_main/spectra.hdf5 \
    --splits-path   ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --n-trials 50 \
    --output-dir hyperopt_results
```

### 2. Train

```bash
cd baselines/neims

python train.py \
    --metadata-path ../../data/NIST2023_GCMS_main/metadata.tsv \
    --spectra-path  ../../data/NIST2023_GCMS_main/spectra.hdf5 \
    --splits-path   ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output-dir outputs/neims_scaffold \
    --wandb-mode online
```

Saves `outputs/neims_scaffold/best_model.pt`.

### 3. Predict -> HDF5

```bash
cd baselines/neims

python predict.py \
    --checkpoint outputs/neims_scaffold/best_model.pt \
    --metadata   ../../data/NIST2023_GCMS_main/metadata.tsv \
    --spectra    ../../data/NIST2023_GCMS_main/spectra.hdf5 \
    --splits     ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output     outputs/neims_scaffold/predictions_test.hdf5 \
    --eval-split test
```

### 4. Evaluate (ICICLE env)

```bash
cd ../../  # repo root

uv run src/icicle/eval_from_predictions.py \
    --predictions baselines/neims/outputs/neims_scaffold/predictions_test.hdf5 \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/neims_scaffold \
    --mode all
```

---

## RASSP

RASSP (Rapid Approximate Spectrum Simulation and Prediction) is a graph-based model.

**Note:** RASSP requires Python 3.7–3.8 and has conflicting dependencies with ICICLE. Always use its own conda environment.

### Installation

```bash
cd baselines/rassp

conda env create -n rassp -f rassp/environment.yml
conda activate rassp

pip install -e .
pip install git+https://github.com/thejonaslab/tinygraph.git
pip install networkx click tqdm SQLAlchemy diskcache h5py pyarrow PyYAML natsort tensorboard optuna numba
```

**NumPy version fix** (if you see Numba or tinygraph errors):
```bash
pip uninstall numpy -y
pip install numpy==1.21
```

(You may need to run `conda deactivate && conda activate /path/to/miniconda3/envs/rassp`.)

### 1. Convert ICICLE data to RASSP parquet format

```bash
cd baselines/rassp

python convert_hdf5_to_parquet.py \
    --split-file ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --hdf5-path  ../../data/NIST2023_GCMS_main/spectra.hdf5 \
    --name nist-scaffold
```

This creates `nist-scaffold_train.parquet`, `nist-scaffold_val.parquet`, `nist-scaffold_test.parquet`.

### 2. Hyperparameter optimization (optional)

```bash
cd baselines/rassp
conda activate rassp

CUDA_VISIBLE_DEVICES=0 python rassp/hyperopt.py rassp/expconfig/hyperopt.yaml
```

### 3. Generate best config and train
cd baselines/rassp
conda activate rassp

CUDA_VISIBLE_DEVICES=0 python rassp/hyperopt.py rassp/expconfig/hyperopt.yaml
```

### 3. Generate best config and train

```bash
cd baselines/rassp
conda activate rassp

# Generate best config from hyperopt results
python rassp/generate_config_from_hyperopt.py \
    results/hyperopt/rassp_hyperopt_nist_scaffold_results.yaml \
    rassp/expconfig/hyperopt.yaml \
    -o rassp/expconfig/best_config.yaml \
    --max-epochs 100

# Train
CUDA_VISIBLE_DEVICES=0 python rassp/forward_train.py rassp/expconfig/best_config.yaml
```

Saves checkpoints to `checkpoints/` with naming: `<config>.<timestamp>.<epoch>.model` + `.meta`.

### 4. Predict -> HDF5

```bash
cd baselines/rassp
conda activate rassp

# List checkpoints to find your trained model
ls checkpoints/

python scripts/run_inference_for_eval.py \
    --checkpoint checkpoints/best_config.<timestamp>.00000100.model \
    --meta       checkpoints/best_config.<timestamp>.meta \
    --metadata   ../../data/NIST2023_GCMS_main/metadata.tsv \
    --splits     ../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output     results/rassp_nist23_scaffold.hdf5 \
    --eval-split test \
    --gpu
```

### 5. Evaluate (ICICLE env)

```bash
conda deactivate  # switch out of rassp env
cd ../../  # repo root

uv run src/icicle/eval_from_predictions.py \
    --predictions baselines/rassp/results/rassp_nist23_scaffold.hdf5 \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/rassp_scaffold \
    --mode all
```

---

## MassFormer

MassFormer is a transformer-based model.

**Note:** MassFormer requires its own conda environment.
MassFormer is a transformer-based model.

**Note:** MassFormer requires its own conda environment.

### Installation

```bash
cd baselines/massformer

conda create -n MF-GPU python=3.8 -y
conda activate MF-GPU

pip install Mako decorator pre-commit jsonschema
pip install -r env/requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    -f https://data.dgl.ai/wheels/repo.html

pip install -I -e .

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PyG (match your PyTorch CUDA version)
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv -y
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Check your CUDA version with `pip list | grep torch` and adjust `cu124` accordingly throughout.

### 1. Hyperparameter optimization (optional)

```bash
cd baselines/massformer
conda activate MF-GPU

python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml --n-trials 50
```

### 2. Train

```bash
cd baselines/massformer
conda activate MF-GPU

python scripts/run_train_eval.py \
    -t config/template.yml \
    -c config/train_nist23_gcms.yml \
    -w online
```

### 3. Configure inference

Edit `config/inference_nist23.yml` and set `massformer.checkpoint_path` to your trained checkpoint.

```yaml
massformer:
  template_config: "config/template.yml"
  custom_config: "config/train_nist23_gcms.yml"
  checkpoint_path: "/path/to/your/checkpoint.pkl"   # <- update this
  device_id: 0

data:
  labels_path: "../../data/NIST2023_GCMS_main/metadata.tsv"
  splits_path:  "../../data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv"
  min_mz: 0
  max_mz: 750
  bin_width: 1.0

include_retrieval_candidates: true
```

### 4. Predict -> HDF5

```bash
cd baselines/massformer
conda activate MF-GPU

python scripts/run_inference_for_eval.py \
    --config config/inference_nist23.yml \
    --output results/predictions/massformer_nist23_scaffold_test.hdf5 \
    --split  test
```

### 5. Evaluate (ICICLE env)

```bash
conda deactivate  # switch out of MF-GPU env
cd ../../  # repo root

uv run src/icicle/eval_from_predictions.py \
    --predictions baselines/massformer/results/predictions/massformer_nist23_scaffold.hdf5 \
    --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
    --labels  data/NIST2023_GCMS_main/metadata.tsv \
    --splits  data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output  results/eval/massformer_scaffold \
    --mode all
```

---

## Naïve Baselines (random, average, full-enumeration barcode)

These run entirely within the ICICLE environment using the model-loading `eval.py`:

```bash
# Random spectrum baseline
uv run src/icicle/eval.py eval=random data=NIST eval.similarity.enable=true

# Average spectrum baseline
uv run src/icicle/eval.py eval=average data=NIST eval.similarity.enable=true

# Full enumeration + barcode matching
uv run src/icicle/eval.py eval=full_enumeration_barcode data=NIST eval.similarity.enable=true
```

---

## Troubleshooting

**RASSP: `ModuleNotFoundError: No module named 'rassp'`**
-> Run `pip install -e .` from `baselines/rassp/`

**RASSP: NumPy / Numba incompatibility**
-> `pip uninstall numpy -y && pip install numpy==1.21`

**RASSP: many molecules skipped**
-> RASSP only supports H, C, N, O, F, P, S, Cl atoms and ≤48 atoms. Molecules outside these constraints produce zero-intensity predictions and are excluded from similarity metrics automatically.

**MassFormer: torch-scatter/torch-sparse version mismatch**
-> Check `pip list | grep torch` for your exact PyTorch + CUDA version and reinstall the PyG wheels to match.

**NEIMS: `KeyError: 'model_config'`**
-> Re-train using the current `train.py`; old checkpoints saved before `model_config` was added are incompatible.
