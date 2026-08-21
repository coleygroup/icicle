# NEIMS: ECFP-based MLP for Mass Spectrum Prediction

This is a standalone implementation of the NEIMS baseline model (ECFP + MLP) for mass spectrum prediction. The model is based on the architecture introduced in ACS Cent. Sci. 2019, 5, 700-708.

## Overview

NEIMS uses Extended Connectivity Fingerprints (ECFP, Morgan fingerprints) as input to a multi-layer perceptron (MLP) with residual connections to predict mass spectra. The model supports bidirectional prediction with optional gating.

## Installation

### Quick Start

The easiest way to get started is to use the quickstart script:

```bash
cd baselines/neims
./quickstart.sh
```

This will install dependencies, run tests, and show you example commands to get started.

### Manual Installation

The NEIMS baseline can be installed as a standalone package:

```bash
cd baselines/neims
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

The NEIMS baseline requires the following dependencies:

- torch (>= 2.0.0)
- numpy (>= 1.20.0)
- pandas (>= 1.3.0)
- h5py (>= 3.0.0)
- rdkit (>= 2022.9.1)
- tqdm (>= 4.60.0)
- wandb (>= 0.13.0)
- optuna (>= 3.0.0, for hyperparameter optimization)
- pyyaml (>= 6.0, optional, for config file support)

## Usage

### Training

To train the NEIMS model with default parameters:

```bash
python train.py \
  --metadata-path /path/to/metadata.tsv \
  --spectra-path /path/to/spectra.h5 \
  --splits-path /path/to/splits.tsv \
  --output-dir outputs/neims_run
```

Key arguments:
- `--metadata-path`: Path to the metadata TSV file containing molecule information
- `--spectra-path`: Path to the HDF5 file containing mass spectra
- `--splits-path`: Path to the TSV file defining train/val/test splits
- `--output-dir`: Directory to save model checkpoints and logs

Wandb logging arguments (optional):
- `--wandb-project`: Wandb project name (default: "neims")
- `--wandb-entity`: Wandb entity/team name (default: None)
- `--wandb-mode`: Wandb mode - "online", "offline", or "disabled" (default: "online")
- `--wandb-run-name`: Custom run name for Wandb (default: None)

Additional arguments for customization:
- `--fp-radius`: Morgan fingerprint radius (default: 2)
- `--fp-length`: Morgan fingerprint length (default: 4096)
- `--hidden-sizes`: Hidden layer sizes (default: [2000] * 8)
- `--dropout`: Dropout rate (default: 0.25)
- `--batch-size`: Batch size (default: 64)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)

### Hyperparameter Optimization

To run hyperparameter optimization using Optuna:

```bash
python hyperopt.py \
  --metadata-path /path/to/metadata.tsv \
  --spectra-path /path/to/spectra.h5 \
  --splits-path /path/to/splits.tsv \
  --n-trials 100 \
  --n-epochs 10 \
  --output-dir hyperopt_results
```

Key arguments:
- `--n-trials`: Number of hyperparameter configurations to try (default: 100)
- `--n-epochs`: Number of epochs per trial (default: 10)
- `--study-name`: Name for the Optuna study (default: "neims_hyperopt")
- `--storage`: Optuna storage URL for distributed optimization (optional)

The optimization will search over:
- Learning rate (log scale: 1e-4 to 1e-2)
- Dropout rate (0.1 to 0.5)
- Number of layers (4 to 10)
- Hidden layer size (1000, 1500, 2000, 2500)
- ResNet bottleneck factor (0.3 to 0.7)
- Mass power for loss (0.3 to 0.7)
- Decay scale for LR scheduler (500 to 2000)
- Batch size (32, 64, 128)
- Bidirectional prediction (True/False)
- Gate bidirectional prediction (True/False)

Results will be saved to:
- `best_params.json`: Best hyperparameters found
- `trials.csv`: All trial results

### Training with Best Hyperparameters

After hyperparameter optimization, train the final model using the best parameters:

```bash
python train.py \
  --metadata-path /path/to/metadata.tsv \
  --spectra-path /path/to/spectra.h5 \
  --splits-path /path/to/splits.tsv \
  --hidden-sizes 2000 2000 2000 2000 2000 2000 2000 2000 \
  --dropout 0.25 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --bidirectional \
  --output-dir outputs/neims_final
```

## Data Format

### Splits File (TSV)
The splits file should have the following columns:
- `mol_id`: Unique molecule identifier
- `inchi_key`: InChI key of the molecule
- `split`: One of "train", "val", or "test"

Example:
```
mol_id	inchi_key	split
UO000024	SJBLBJCIOBWHAC-UHFFFAOYSA-N	val
UO000003	PHHRZFRBFLDDTQ-UHFFFAOYSA-N	val
TT000166	GFKUCJKLCHSSJN-JRYVEOFUSA-N	train
TT000153	AVIRMQMUBGNCKS-RWCYGVJQSA-N	test
```

### Metadata File (TSV)
The metadata file should contain at least:
- `mol_id`: Unique molecule identifier (matching splits file)
- `smiles` or `standardized_smiles`: SMILES string of the molecule

### Spectra File (HDF5)
The HDF5 file should have groups keyed by `mol_id`, each containing:
- `masses` (or `mz`): m/z values array
- `intensities` (or `intensity`): Intensity values array

## Model Architecture

The NEIMS model consists of:
1. **Input layer**: Linear projection from fingerprint size to first hidden size
2. **Residual blocks**: Multiple residual blocks with batch normalization, ReLU, dropout, and bottleneck layers
3. **Prediction head**: 
   - Standard: Single linear layer to output spectrum
   - Bidirectional: Separate forward and backward predictors with optional learned gating

Key features:
- Batch normalization for stable training
- Residual connections for better gradient flow
- Dropout for regularization
- Mass-based output masking (predictions beyond molecule mass + tolerance are zeroed)
- Generalized MSE loss with mass-weighted error

## Output

Training produces:
- `best_model.pt`: Best model checkpoint based on validation loss
- `config.json`: Training configuration
- `results.json`: Final metrics (best validation loss, test loss)
- Wandb logs: Training metrics, losses, and hyperparameters logged to Weights & Biases

## Evaluation

After training a NEIMS model, you can evaluate it using the ICICLE evaluation pipeline. This allows you to compute the same metrics used for ICICLE models: similarity metrics, retrieval with formula, and retrieval with retention index.

### Using the ICICLE Evaluation Pipeline

The trained NEIMS model can be evaluated using the main ICICLE evaluation script with the NEIMS configuration:

```bash
cd /path/to/icicle-dev
python -m icicle.eval eval=neims
```

This will:
1. Load your trained NEIMS model from `baselines/neims/outputs/best_model.pt`
2. Run all enabled evaluations (similarity, retrieval_with_formula, retrieval_with_ri)
3. Save results to the output directory with the same format as ICICLE models

### Custom Evaluation Configuration

You can customize the evaluation by modifying `examples/configs/eval/neims.yaml` or overriding parameters:

```bash
# Evaluate on a subset of test data
python -m icicle.eval eval=neims eval.fraction_of_spectra_to_compute=0.1

# Use a different checkpoint
python -m icicle.eval eval=neims \
  eval.model.architecture.checkpoint_path=baselines/neims/outputs/my_model.pt

# Enable/disable specific evaluations
python -m icicle.eval eval=neims \
  eval.similarity.enable=true \
  eval.retrieval_with_formula.enable=true \
  eval.retrieval_with_ri.enable=false
```

### Output Format

The evaluation produces:
- `similarity_results.csv`: Per-molecule similarity metrics (cosine, Jensen-Shannon, etc.)
- `similarity_summary_metrics.txt`: Aggregated statistics across all molecules
- `all_evaluation_spectra.hdf5`: HDF5 file with predictions, ground truth, and metrics
- `retrieval_with_formula_results.csv`: Retrieval performance with formula filtering (if enabled)
- `retrieval_with_ri_results.csv`: Retrieval performance with RI filtering (if enabled)

All metrics are also logged to Weights & Biases for easy comparison with other models.

### Comparing with Other Baselines

The unified evaluation pipeline makes it easy to compare NEIMS with other baselines:

```bash
# Evaluate NEIMS
python -m icicle.eval eval=neims

# Evaluate average baseline
python -m icicle.eval eval=average

# Evaluate random baseline
python -m icicle.eval eval=random
```

All baselines produce outputs in the same format, making direct comparison straightforward.

## Notes

- The model uses the exact same data splits as specified in the splits TSV file, ensuring fair comparison with other methods
- Morgan fingerprints are computed on-the-fly during training
- The learning rate follows a square root decay schedule with a minimum multiplier
- Gradient clipping (norm=1.0) is applied for training stability
- Early stopping is used to prevent overfitting
- All metrics are logged to Weights & Biases for easy tracking and comparison
