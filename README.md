# ICICLE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22854374.svg)](https://doi.org/10.5281/zenodo.22854374)
[![Preprint](https://img.shields.io/badge/preprint-ChemRxiv-blue)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15008135/v1)

Implementation of the work presented in "Structure elucidation of unknown molecules with physics-constrained neural simulation of electron ionization mass spectrometry". This codebase contains scripts to preprocess data, models, baselines, evaluation pipelines, visualization functions, and example data to replicate results from the paper.

![](assets/asset.jpeg)

## Table of Contents

1. [Installation](#installation)  
2. [Usage – Simulating Mass Spectra](#usage)
3. [Replicating Paper Results](#replicate-results)  
   3.1 [Data Processing](#data-processing)  
   3.2 [Training Models](#training)   
   3.3 [Evaluation](#evaluation)  
   3.4 [Running Baselines](#baselines)
4. [Citation](#citation)  

## Installation <a name="installation"></a>

### Prerequisites
- Python 3.12 or higher
- CUDA 12.4 (for GPU support)

### Installation Steps

1. **Install `uv`** (recommended package manager): Follow instructions at [github.com/astral-sh/uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)

2. **Install dependencies**:
   ```bash
   # For GPU with CUDA 12.4
   uv sync --extra cu124
   
   # For CPU only
   uv sync --extra cpu
   
   # Install the package in editable mode
   uv pip install -e .
   
   # Install pre-commit hooks (optional, for development)
   uv run pre-commit install
   ```

3. **Configure environment variables** (required for training/evaluation):
   
   If you plan to train models or run evaluations, set up Weights & Biases (WandB) for experiment tracking:
   ```bash
   export WANDB_PROJECT="your-project-name"
   export WANDB_ENTITY="your-wandb-username"
   ```
   
   You will be prompted to log in to WandB when running training for the first time.

**Note on CUDA versions**: If your CUDA version differs from 12.4, modify `pyproject.toml` and replace `cu124` with your CUDA version (e.g., `cu121` for CUDA 12.1).
If you get the error message `Warning: lazyInitCUDA is deprecated. Please use lazyInitDevice(at::kCUDA) instead. (function lazyInitCUDA)`, please run `export TORCH_CPP_LOG_LEVEL=ERROR` in your shell.


## Usage – Simulating Mass Spectra <a name="usage"></a>

### Command Line Inference

Run predictions with pretrained checkpoints using the command line interface:

```bash
uv run src/icicle/main.py -h  # Show all available options

uv run src/icicle/main.py \
  --intensity-predictor /path/to/ckpt.ckpt \
  --smiles "CCO" "CCC" "C1=CC=C(C=C1)C(=O)O" \
  --plot \
  --output-format csv \
  --output-folder predictions
```

**Options:**
- `--intensity-predictor`: Path to intensity predictor checkpoint (required)
- `--fragment-generator`: Path to fragment generator checkpoint (optional)
- `--smiles`: One or more SMILES strings
- `--smiles-file`: Path to CSV file with SMILES (one per line)
- `--plot`: Generate spectrum plots
- `--output-format`: Output format (`csv`, `hdf5`, or `none`)
- `--output-folder`: Directory to save predictions

**Note**: This runs full enumeration with MAGMa and predicts intensities using the Intensity Predictor (recommended). Optionally, you can use a Fragment Generator checkpoint to predict fragments instead of full enumeration.

### Large-Scale GPU Inference (`batch_infer.py`)

For large compound libraries (millions of molecules), use the multi-GPU batch inference script. It distributes the input file across GPUs, runs GPU-accelerated fragment enumeration, checkpoints progress to HDF5 so interrupted jobs can resume, and merges shard files at the end.

```bash
# Minimal: full enumeration, 8 GPUs
uv run src/icicle/batch_infer.py \
    --intensity-predictor /path/to/ckpt.ckpt \
    --input compounds.csv \
    --smiles-col smiles \
    --output predictions.hdf5 \
    --num-gpus 8

# With fragment generator, headerless CSV, row range (for splitting across machines)
uv run src/icicle/batch_infer.py \
    --intensity-predictor /path/to/ip.ckpt \
    --fragment-generator /path/to/fg.ckpt \
    --input compounds.csv \
    --no-header --smiles-col-idx 0 \
    --output predictions.hdf5 \
    --num-gpus 8 --num-workers 6 --batch-size 64 \
    --checkpoint-every 500 \
    --start-idx 0 --end-idx 10000000
```

**Key options:**
- `--num-gpus N` / `--gpus 0,1,2,3`: number of GPUs or explicit GPU indices
- `--batch-size`: molecules per GPU forward pass (default 64)
- `--num-workers`: DataLoader workers per GPU for root-graph preprocessing (default 4)
- `--max-nodes`: skip molecules with more heavy atoms than this (default 50)
- `--checkpoint-every N`: flush results to disk every N batches — enables resume on crash
- `--start-idx` / `--end-idx`: row range (inclusive/exclusive) for splitting a large file across machines
- `--no-header` / `--smiles-col-idx`: for headerless CSV/TSV files

**Output HDF5 layout:**
```
predictions.hdf5
├── smiles          (N,)         variable-length UTF-8 string
├── intensities     (N, n_bins)  float32  — zeros for invalid SMILES
├── num_fragments   (N,)         int32    — 0 for invalid
└── valid           (N,)         bool
```

### Python API

For programmatic access, load models from checkpoints:

```python
from icicle.models.eims_predictor import (
    EIMSPredictorWithFragmentGenerator,
    EIMSPredictorFromFullEnumeration,
)
from icicle.utils.visualization.mass_spectra import plot_mass_spectrum

# Load model with full enumeration (recommended)
model = EIMSPredictorFromFullEnumeration.load_from_checkpoint(
    intensity_predictor_checkpoint=ckpt_path
)

# Or load with fragment generator
model = EIMSPredictorWithFragmentGenerator.load_from_checkpoint(
    fragment_generator_checkpoint=fg_ckpt_path,
    intensity_predictor_checkpoint=ip_ckpt_path
)

# Predict spectrum
predicted_spectrum = model.predict_from_smiles("CCO")

# Visualize
plot_mass_spectrum(
    spectrum=predicted_spectrum,
    output_path="ethanol.png"
)
```

For interactive examples, see the Jupyter notebooks in `examples/notebooks/`.

**Note**: Please contact the authors for pre-trained model checkpoints.


## Replicating Paper Results <a name="replicate-results"></a>

This section describes how to reproduce the experiments from the paper, including data processing, model training, and evaluation.

### Data Processing <a name="data-processing"></a>

Before training models, process raw mass spectrometry data. The pipeline extracts spectra from SDF files, creates train/validation/test splits, and generates fragment trees using MAGMa.

**Note on `data/` layout**: `data/NIST2023_GCMS_main/` may be a symlink to a
shared location on multi-user setups rather than a plain directory — this is
expected and transparent to every script/config in this repo, which
reference it by its `data/` path regardless. On a fresh machine, either
point this path at an existing processed copy or generate it from raw data
using the steps below.

#### Processing NIST Data

The main dataset used in the paper is NIST 2023 GC-MS. To process it:

1. **Set up paths** in `examples/scripts/data_processing/prepare_NIST_main.sh`:
   ```bash
   SDF_PATH="/path/to/gcms_nist23.SDF"
   DATA_DIR="/path/to/output/directory"
   ```

2. **Run the processing script**:
   ```bash
   bash examples/scripts/data_processing/prepare_NIST_main.sh
   ```

Or run the individual steps manually:

```bash
# Step 1: Extract spectra and metadata from SDF
uv run examples/scripts/data_processing/extract_spectra_from_sdf.py \
   --sdf-path $SDF_PATH \
   --output-dir $DATA_DIR

# Step 2: Create data splits (random and scaffold used in paper)
uv run examples/scripts/data_processing/create_splits.py \
   --metadata-path $DATA_DIR/metadata.tsv \
   --output-dir $DATA_DIR/splits \
   --split-types random scaffold

# Step 3: (Optional) Deduplicate stereoisomers
uv run examples/scripts/data_processing/deduplicate_stereoisomers.py \
   --split-file $DATA_DIR/splits/scaffold.tsv \
   --metadata-file $DATA_DIR/metadata.tsv

# Step 4: Run MAGMa for full fragment enumeration (slow, run once)
uv run examples/scripts/data_processing/label_ground_truth_dags.py \
   --data-dir $DATA_DIR \
   --num-h-shifts 6 \
   --detect-isotope-patterns
```

This creates a MAGMa tree file at:
`$DATA_DIR/processed_6_h_shifts_50_peaks_3_tree_depth_6_broken_bonds_True_isotope_patterns/magma_tree.hdf5`

#### Preparing PubChem for Retrieval Tasks

```bash
bash examples/scripts/data_processing/prepare_pubchem.sh
```

After batch inference over PubChem (see [Large-Scale GPU Inference](#large-scale-gpu-inference-batch_inferpy) above), attach InChIKey-14 identifiers to the HDF5 and build the RI sort index:

```bash
# Attach InChIKey-14 identifiers (required for retrieval eval)
uv run examples/scripts/evaluation/add_inchikeys_inplace.py \
    --hdf5 results/pubchem_predictions.hdf5

# Build RI sort index (required for top-N RI retrieval)
uv run examples/scripts/evaluation/add_sort_index_to_hdf5.py \
    --hdf5 results/inference/pubchem_predictions_rerun_260710.hdf5
```


### Training Models <a name="training"></a>

ICICLE includes two main models:
- **Intensity Predictor**: Predicts peak intensities given a fragment tree
- **Fragment Generator**: Predicts which fragments will be present

#### Basic Training

```bash
uv run src/icicle/train.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    model=intensity_predictor
```

**Key Configuration Options:**

1. **Dataset** (`data=...`): `NIST`
2. **Split** (`data.split_name=...`):
   - `random_no_xeno_aas_deduplicated` — primary random split
   - `scaffold_no_xeno_aas_deduplicated` — primary scaffold split (more challenging)
3. **Model** (`model=...`): `intensity_predictor` (main), `fragment_generator`
4. **Data fraction** (`data.training_data_fraction=0.01`): subset for fast experiments

#### Training the Fragment Generator

The fragment generator is an alternative to full MAGMa enumeration at inference
time — it predicts which fragments are present directly, rather than
enumerating and scoring all of them. Train it the same way as the intensity
predictor, swapping `model=`:

```bash
uv run src/icicle/train.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    model=fragment_generator
```

See [Python API](#usage) above for loading a trained fragment-generator
checkpoint alongside an intensity-predictor checkpoint at inference time.

#### Advanced Training Configuration

```bash
uv run src/icicle/train.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    model=intensity_predictor \
    model.architecture.h_shift_range=6 \
    model.architecture.add_isotopes=true \
    model.architecture.loss_fn="entropy" \
    model.architecture.hidden_size=256 \
    model.architecture.gnn_message_passing_steps=2 \
    model.architecture.inter_fragment_attention_layers=2 \
    trainer.max_steps=50000 \
    trainer.gradient_clip_val=1.0 \
    system.devices=[0,1] \
    system.seed=42
```

**Available loss functions**: `cosine_similarity` (default), `entropy` (recommended), `mse`, `weighted_cosine_nist_gc`, `composite_weighted_cosine_nist_gc`

#### Hyperparameter Optimization

```bash
uv run src/icicle/train.py \
    hyperparameter_sweep=default \
    data=NIST \
    model=intensity_predictor \
    --multirun
```

Search space: `examples/configs/hyperparameter_sweep/default.yaml`.

#### Running on SLURM Clusters

```bash
sbatch examples/scripts/training/submit_slurm_job.sh uv run src/icicle/train.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    model=intensity_predictor
```

Batch training scripts:
```bash
bash examples/scripts/training/run_training_pipeline.sh    # submit multiple jobs
bash examples/scripts/training/hyperopt_models.sh          # hyperparameter search
```

#### Monitoring Training

- **Checkpoints**: saved automatically to `results/single_run/<date>/<time>/checkpoints/` — move best `.ckpt` to `checkpoints/` manually
- **WandB**: online dashboard at wandb.ai


### Evaluation <a name="evaluation"></a>

Three evaluation tasks:
1. **Spectrum similarity** — compare predicted vs. experimental spectra
2. **Formula-match retrieval** — rank PubChem candidates with same molecular formula
3. **Top-N RI retrieval** — rank the N PubChem candidates with closest predicted retention index

#### Spectrum Similarity

```bash
# Single split
uv run src/icicle/eval.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    eval=icicle_fe \
    eval.similarity.enable=true

# Multi-GPU (recommended for speed)
uv run torchrun --nproc_per_node=2 src/icicle/eval.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    eval=icicle_fe \
    eval.similarity.enable=true \
    system.devices=[0,1]
```

Computed metrics: cosine similarity, Jensen-Shannon similarity, entropy similarity/distance, spectral contrast angle, MSE, weighted cosine (NIST GC weighting), composite similarity.

Output: `results/eval/<run>/similarity_results.csv`

#### Formula-Match Retrieval

Candidates are all PubChem molecules sharing the query's molecular formula; the 50 most similar by Tanimoto (Morgan fingerprint, radius 2, 2048 bits) plus the true molecule → 51 candidates.

```bash
uv run torchrun --nproc_per_node=2 src/icicle/eval.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    eval=icicle_fe \
    eval.retrieval_with_formula.enable=true \
    system.devices=[0,1]
```

Output: `results/eval/<run>/retrieval_with_formula_results.csv`

#### Top-N RI Retrieval

Predicts RI for all PubChem compounds and, for each query, takes the N closest by predicted RI as the candidate set. N ∈ {1 000, 100 000, 1 000 000, 10 000 000, all} — the `all` level is the full ~95M-compound global rank.

```bash
# Step 1: Train an AIRI retention-index model (one per column type). Requires
# the masskit_ai conda env (see "RI Prediction Workflow" below) and prepared
# AIRI parquet files under a per-column-type data dir, e.g.
# data/NIST2023_GCMS_main/airi_data_stdnp_random/. There is no --column-type
# flag — the column type is implicit in which --data-dir you point at.
uv run examples/scripts/retention_index/train_airi_models.py \
    --data-dir data/NIST2023_GCMS_main/airi_data_stdnp_random/ \
    --output-dir airi_models/stdnp
# Repeat for airi_data_stdpolar_random/ and airi_data_semistdnp_random/ to
# cover all three RI column types.

# Step 2: Predict RI for all PubChem compounds using the trained model(s),
# then merge per-shard predictions into one file (see "RI Prediction
# Workflow" below for the full predict/merge commands).

# Step 3: Build RI sort index on master HDF5
uv run examples/scripts/evaluation/add_sort_index_to_hdf5.py \
    --hdf5 results/inference/pubchem_predictions_rerun_260710.hdf5

# Step 4: Run global retrieval eval (config: examples/configs/pubchem_retrieval/default.yaml)
uv run examples/scripts/evaluation/pubchem_global_retrieval.py

# Override config values on the CLI
uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
    output_dir=results/my_run \
    ri_types=[StdNP] \
    top_n_levels=[all,1000,100000]
```

Key config fields (`examples/configs/pubchem_retrieval/default.yaml`):
- `hdf5_files`: PubChem prediction HDF5 (output of `batch_infer` + `add_inchikeys`)
- `pubchem_ri_parquet`: PubChem + AIRI-predicted RI (parquet, sorted by InChIKey)
- `top_n_levels`: candidate set sizes — `"all"` = full global rank, integers = N closest by RI
- `ri_types`: column types to evaluate (`StdNP`, `SemiStdNP`, `StdPolar`)
- `ranking_metrics`: `cosine`, `entropy`, `weighted_cosine`

Output: `results/pubchem_retrieval_eval_<run>/retrieval_global_results.json`
(global rank) and `retrieval_ablation_<ri_type>.json` (per RI-window level).

**Regenerating every paper retrieval table at once**: once the global
retrieval eval above has been run for ICICLE, NEIMS, and MassFormer,
`examples/scripts/evaluation/dump_complete_retrieval_tables.py` reads every
`retrieval_*_results.json`/`retrieval_per_query_*.tsv` file on disk and
regenerates the complete set of CSV + LaTeX tables (global, RI-window
ladder, MW-window, RI∪MW union, heavy-atom ladder) in one pass. Always uses
autofail (never inject) — a query whose true molecule falls outside its own
candidate window should fail the retrieval task, not be silently excluded.

```bash
uv run examples/scripts/evaluation/dump_complete_retrieval_tables.py
```

Output: `figures/retrieval_pubchem_tables/*.csv` and `*.tex`.

**RI-window coverage ceiling** (how often the true molecule falls inside a
given RI-window size at all, independent of any model's predictions — an
upper bound on what RI-window retrieval could ever achieve):

```bash
uv run examples/scripts/evaluation/compute_ri_coverage.py
# Override: top_n_levels=[1000,10000,100000] ri_types=[StdNP]
```

**Running the full pipeline for all three models unattended**: rather than
calling `pubchem_global_retrieval.py` once per model by hand,
`run_pubchem_global_retrieval_all_models.sh` runs the global ("all"-level)
retrieval for ICICLE, NEIMS, and MassFormer end to end, and
`run_pubchem_ri_window_ladder.sh` runs the more expensive RI-window ladder
(1000/100000/1000000/10000000 candidates) afterward. Both are idempotent —
safe to interrupt and rerun, they resume from the last completed stage.
`run_pubchem_global_retrieval_all_models.sh` automatically calls
`examples/scripts/evaluation/convert_massformer_pubchem_chunks.py` internally
to convert MassFormer's per-chunk HDF5 predictions into the same columnar
format ICICLE/NEIMS use before scanning — no separate step needed.

```bash
# Global rank, all 3 models (run first)
bash examples/scripts/evaluation/run_pubchem_global_retrieval_all_models.sh

# RI-window ladder, all 3 models (run after the above completes; much slower)
bash examples/scripts/evaluation/run_pubchem_ri_window_ladder.sh
```

#### RI Retrieval (Pre-Built Candidate Lists)

A second, simpler RI-retrieval path lives in `src/icicle/eval.py` itself
(`eval.retrieval_with_ri.enable=true`), reachable via
`evaluate_all_baselines.sh`. Unlike the [Top-N RI Retrieval](#top-n-ri-retrieval)
path above, it does not scan the full PubChem prediction HDF5 at eval time —
it reads pre-built, per-RI-type candidate TSVs from
`${data.data_dir}/retrieval_scaffold/` (or `retrieval/` for the random
split), predicts spectra for just those candidates, and ranks them. Use the
full-PubChem path above for the paper's global-rank numbers; use this path
for a quick, cheap RI-retrieval check that doesn't require a full PubChem
inference run.

The candidate TSVs must be built once per split before this path can run:

```bash
uv run examples/scripts/retention_index/create_retrieval_candidates.py \
    --ri-predictions-file data/PubChem/PubChem_filtered_with_ri_new_random_inchikey_cache.parquet \
    --ri-dataset-file data/NIST2023_GCMS_main/retention_index_airi/ri_dataset_random_split_no_xeno_aas.tsv \
    --output-dir data/NIST2023_GCMS_main/retrieval_scaffold/ \
    --mode top-n --top-n 1000
```

```bash
uv run src/icicle/eval.py \
    data=NIST \
    data.split_name=scaffold_no_xeno_aas_deduplicated \
    eval=icicle_fe \
    eval.retrieval_with_ri.enable=true
```

Output: `results/eval/<run>/retrieval_with_ri_<ri_type>_results.csv`

#### Running All Evaluations

```bash
# All native baselines + ICICLE similarity + retrieval
bash examples/scripts/evaluation/evaluate_all_baselines.sh

# ICICLE similarity + formula retrieval, all seeds/splits (checkpoints hardcoded at top)
bash examples/scripts/evaluation/run_icicle_similarity.sh
bash examples/scripts/evaluation/run_icicle_formula_retrieval.sh
```

#### Visualizing Results

Jupyter notebooks in `examples/notebooks/`:
- `fig_similarity_results.ipynb` — similarity metric distributions
- `fig_retrieval_results_formula_match.ipynb` — formula-match retrieval
- `fig_retrieval_results_ri_match.ipynb` — RI-based retrieval
- `fig_retrieval_results_pubchem_global.ipynb` / `fig_retrieval_results_pubchem_global_scaffold.ipynb` — PubChem global retrieval ablations (random / scaffold split)
- `fig_retrieval_win_loss_examples.ipynb` — ICICLE-vs-NEIMS win/loss spectrum examples by functional group (candidate examples themselves are generated by `examples/scripts/evaluation/build_win_loss_examples_v2.py`, which selects examples by rank-outcome category across formula-match and global retrieval)
- `fig_retrieval_icicle_vs_neims_structural.ipynb` — structural/descriptor-based analysis of where ICICLE vs. NEIMS wins or loses at retrieval; its molecular-complexity panels require running `examples/scripts/evaluation/compute_molecular_complexity.py` first to append SAScore/NPScore/SPScore/Boettcher complexity columns to the merged structural-analysis CSV the notebook reads


### Running Baselines <a name="baselines"></a>

| Baseline | Type | Env |
|---|---|---|
| random, average, full_enumeration_barcode | native | ICICLE uv env |
| NEIMS, NEIMS-GNN | external | `baselines/neims/` — own pip env |
| RASSP | external | conda: `rassp` |
| MassFormer | external | conda: `MF-GPU` |

For detailed installation and training instructions: **[baselines/README.md](baselines/README.md)**.

**Native baselines** run directly via `eval.py`:
```bash
uv run src/icicle/eval.py eval=random data=NIST eval.similarity.enable=true
uv run src/icicle/eval.py eval=average data=NIST eval.similarity.enable=true
uv run src/icicle/eval.py eval=full_enumeration_barcode data=NIST eval.similarity.enable=true
```

**External baselines** use a two-step predict → eval workflow:
```bash
# Step 1: run inference in the baseline's own conda env
# (see baselines/README.md for per-baseline commands)

# Step 2: evaluate in ICICLE env
uv run src/icicle/eval_from_predictions.py \
  --predictions results/predictions/<model>.hdf5 \
  --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \
  --labels data/NIST2023_GCMS_main/metadata.tsv \
  --splits data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
  --output results/eval/<model>_nist_scaffold \
  --mode all
```

All eval results land in `results/eval/<SLURM_JOB_ID|local>_<description>/`.

### Verifying Reported Numbers <a name="verify-paper-tables"></a>

Every retrieval and similarity number reported for this project (global
PubChem retrieval, RI-window ablation, heavy-atom and MW filtering,
RI∪MW union, formula-match retrieval) is regenerable directly from the
result files on disk — nothing was hand-copied without a traceable source.
To re-verify every number against the underlying data:

```bash
uv run examples/scripts/evaluation/verify_paper_retrieval_tables.py
```

This prints every metric freshly computed from the raw
`retrieval_*_results.json` / `retrieval_per_query_*.tsv` /
`retrieval_with_formula_results.csv` files — rerun after any change to the
underlying eval pipeline to confirm nothing has drifted.

**Creating the RASSP-restricted split**: RASSP can only score molecules
meeting its own structural constraints (allowed elements, atom-count limits,
single connected fragment — see the script's docstring for the exact rules).
The `_rassp`-suffixed split files used below are generated once via:

```bash
uv run examples/scripts/data_processing/filter_splits_for_rassp.py \
    --splits-path data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --metadata-path data/NIST2023_GCMS_main/metadata.tsv \
    --output-path data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated_rassp.tsv \
    --max-n-atoms 48
```

**Naive baselines** (random / average / full-enumeration-barcode
similarity) are deterministic given the training set (no learnable
parameters, so no seed variation) and are run once per split, on both the
full test set and RASSP's own filtered subset (so every model can be
compared against the baselines on its native query scope):

```bash
# Full test set
uv run src/icicle/eval.py data=NIST data.split_name=random_no_xeno_aas_deduplicated \
    eval=random eval.similarity.enable=True hydra.run.dir=results/eval/random_random_sim
uv run src/icicle/eval.py data=NIST data.split_name=scaffold_no_xeno_aas_deduplicated \
    eval=average eval.similarity.enable=True hydra.run.dir=results/eval/average_scaffold_sim
uv run src/icicle/eval.py data=NIST data.split_name=random_no_xeno_aas_deduplicated \
    eval=full_enumeration_barcode eval.similarity.enable=True \
    hydra.run.dir=results/eval/full_enumeration_barcode_random_sim

# RASSP-restricted subset (swap split_name for the _rassp variant)
uv run src/icicle/eval.py data=NIST \
    data.split_name=random_no_xeno_aas_deduplicated_no_qcxms2_rassp \
    eval=random eval.similarity.enable=True \
    hydra.run.dir=results/eval/random_random_rassp_subset_sim
```

Numbers are read from each run's `similarity_summary_metrics.txt`
(`avg_cosine_similarity`, `avg_entropy_similarity`,
`avg_weighted_cosine_nist_gc`, `avg_composite_similarity_nist_gc`). If that
file is missing due to a transient write failure but
`similarity_results.csv` exists, the summary can be recomputed directly:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('results/eval/<run>/similarity_results.csv')
print(df[['cosine_similarity','entropy_similarity','weighted_cosine_nist_gc','composite_similarity_nist_gc']].mean())
"
```

**Known outstanding gap**: heavy-atom-count-filtered retrieval on the
**random split** is ICICLE-only — NEIMS and MassFormer runs repeatedly hit
GPU out-of-memory errors under concurrent scheduling and were never
completed; this comparison should be treated as preliminary until re-run on
dedicated (non-concurrent) GPU allocation. (The scaffold-split version of
this comparison is complete for all three models — see below.)

### Reproducing Specific Retrieval Ablations (`paper_reruns/`) <a name="paper-reruns"></a>

`examples/scripts/evaluation/paper_reruns/` holds one script per group of
retrieval-filtering ablations — each has a header comment describing exactly
what it computes and for which model(s)/split. These are the exact scripts
used to produce the reported filtering-ablation numbers, not a paraphrase —
running one from a clean `results/` state reproduces its numbers exactly
(modulo model non-determinism already captured by reporting a ± across
seeds where applicable).

| Script | Computes |
|---|---|
| `rerun_ri_ladder.sh` | RI-window retrieval ladder (N=1,000..all candidates by predicted-RI closeness), all 3 RI column types, ICICLE/NEIMS/MassFormer, random split |
| `rerun_mw_ha_stdnp_subset.sh` | MW-alone and heavy-atom-alone retrieval (ICICLE), restricted to the StdNP RI subset for direct comparison against the RI ladder's N=1,000 row |
| `rerun_mw_ha_fullset.sh` | MW-alone and heavy-atom-alone retrieval (ICICLE) against the full test set, no RI restriction |
| `rerun_union_icicle_stdnp.sh` | RI∪MW and RI∪heavy-atom union retrieval (ICICLE, StdNP), across the RI-window ladder |
| `rerun_mw_union_perquery.sh` | Per-query candidate-pool-size dumps for the RI∪MW union track (N=1,000, StdNP) |
| `rerun_union_neims_massformer_narrow.sh` | RI∪MW union retrieval for NEIMS/MassFormer at the narrow MW widths already computed for ICICLE |
| `run_rassp_headtohead.sh` | ICICLE/NEIMS/MassFormer/RASSP similarity + formula retrieval, all evaluated on RASSP's own native scaffold split, for a fair four-way comparison |
| `count_dataset_sizes.sh` | Every dataset/split/AIRI-predictor size, with the source file for each count (provenance only, stdout) |

All of these except `count_dataset_sizes.sh` require the base full-PubChem
scan (`retrieval_global_results.json`) to already exist for the model(s)
they touch — run the `all`-track command from
[Top-N RI Retrieval](#top-n-ri-retrieval) first if starting from scratch.

### Scaffold-Split Full-PubChem Retrieval

The [Top-N RI Retrieval](#top-n-ri-retrieval) workflow above is documented
for the **random split**. The scaffold split (out-of-distribution setting)
uses the same `pubchem_global_retrieval.py` script and config, with three
differences:

1. **Checkpoints**: scaffold-split-trained checkpoints
   (`checkpoints/entropy_scaffold_s*/`, `baselines/neims/outputs/neims_scaffold_s*/`,
   MassFormer's scaffold-trained config) instead of random-split ones.
2. **RI ladder excluded**: the AIRI retention-index model is trained on the
   random split, so applying it to scaffold-split queries would leak
   information the split is meant to withhold. Pass `skip_ri_ladder=true`
   and use `mw_global=true` / `heavy_atom_global=true` instead of the
   RI-window levels.
3. **Heavy-atom filter uses a scaffold-split-specific predictor**: the
   default `heavy_atom_model_path` (`checkpoints/heavy_atom_predictor_random_split.joblib`)
   is trained on the random split; 79.3% of the scaffold-split test set's
   molecules also appear in the random split's train partition, so using it
   for scaffold-split filtering leaks information. Train a scaffold-split
   counterpart first and pass it explicitly:
   ```bash
   uv run examples/scripts/evaluation/train_heavy_atom_predictor.py \
       --split-path data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
       --output-model-path checkpoints/heavy_atom_predictor_scaffold_split.joblib
   ```

Example invocation (ICICLE, scaffold, MW window ±5Da, full test set):
```bash
uv run examples/scripts/evaluation/pubchem_global_retrieval.py \
    hdf5_files=[results/inference/scaffold/pubchem_predictions_scaffold_s1.hdf5] \
    nist_split_path=data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    output_dir=results/pubchem_retrieval_eval_icicle_scaffold_s1 \
    spectra_cache_dir=results/pubchem_retrieval_eval_icicle_scaffold_s1 \
    skipped_log=results/pubchem_retrieval_eval_icicle_scaffold_s1/skipped_queries.tsv \
    top_n_levels=[1000,all] \
    skip_ri_ladder=true \
    mw_global=true mw_window_da=5
```

Same pattern for `heavy_atom_global=true heavy_atom_window=<N> heavy_atom_model_path=checkpoints/heavy_atom_predictor_scaffold_split.joblib`.
Output dirs follow the `<model>_scaffold_s<seed>` naming convention
(`results/pubchem_retrieval_eval_{icicle,neims,massformer}_scaffold_s1`).


## Citation <a name="citation"></a>

```
@article{lederbauer2026fragment,
  title   = {Fragment-Grounded Neural Simulation of Electron Ionization Mass Spectra at Library Scale},
  author  = {Magdalena Lederbauer and Runzhong Wang and Connor Coley},
  year    = {2026},
  journal = {ChemRxiv preprint},
  doi     = {10.26434/chemrxiv.15008135},
  url     = {https://chemrxiv.org/doi/full/10.26434/chemrxiv.15008135/v1}
}
```
