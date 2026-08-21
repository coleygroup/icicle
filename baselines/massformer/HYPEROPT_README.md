The hyperparameter optimization script ([scripts/hyperopt.py](scripts/hyperopt.py)) uses Optuna's Tree-structured Parzen Estimator (TPE) sampler to efficiently search the hyperparameter space. It supports:

- **Flexible search spaces**: Integer, float, log-uniform, and categorical parameters
- **Pruning**: Early stopping of unpromising trials using median pruner
- **Resumability**: Continue optimization from saved studies
- **WandB integration**: Optional logging to Weights & Biases
- **Parallel execution**: Can run multiple trials in parallel (with separate GPU processes)

## Quick Start

### 1. Basic Usage

```bash
# Run hyperparameter optimization with default config
cd baselines/massformer
python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml
```

### 2. With Custom Settings

```bash
# Run with custom number of trials and study name
python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml \
    --n-trials 100 \
    --study-name my_hyperopt_study \
    --output-dir results/my_hyperopt
```

### 3. With WandB Logging

```bash
# Enable WandB logging for detailed tracking
python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml \
    --use-wandb \
    --n-trials 50
```

### 4. Resume Previous Study

```bash
# Resume a previous study (requires SQLite storage)
python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml \
    --resume \
    --storage sqlite:///results/hyperopt/study.db
```

## Configuration File Structure

The hyperopt configuration file (e.g., [config/hyperopt_nist23_gcms.yml](config/hyperopt_nist23_gcms.yml)) consists of several sections:

### 1. WandB Settings (optional)
```yaml
entity_name: "your_wandb_entity"
project_name: "massformer_nist23_hyperopt"
run_name: "nist23_gcms_hyperopt"
```

### 2. Optuna Configuration
```yaml
optuna:
  study_name: "massformer_nist23_scaffold"
  n_trials: 50
  sampler: "tpe"  # Options: tpe, random, cmaes
  pruner: "median"  # Options: median, hyperband, none
  direction: "maximize"  # Maximize cosine similarity
  storage: null  # Use SQLite for persistence: "sqlite:///study.db"
```

### 3. Fixed Configurations

These sections contain parameters that are **not optimized** but remain constant:

- `data`: Dataset and preprocessing settings
- `model`: Fixed model architecture components
- `run`: Fixed training settings

### 4. Search Space

Define the hyperparameter search space under `search_space`:

```yaml
search_space:
  model:
    ff_h_dim:
      type: "categorical"
      choices: [512, 768, 1000, 1280, 1536]

    dropout:
      type: "float"
      low: 0.0
      high: 0.5
      step: 0.05

  run:
    learning_rate:
      type: "loguniform"
      low: 1e-5
      high: 1e-2

    batch_size:
      type: "categorical"
      choices: [16, 32, 64, 128]
```

#### Parameter Types

- **`int`**: Integer parameter with `low`, `high`, and optional `step`
- **`float`**: Float parameter with `low`, `high`, and optional `step`
- **`loguniform`**: Log-scale parameter (good for learning rates, weight decay)
- **`categorical`**: Discrete choices from a list

## Command-Line Options

```
Usage: python scripts/hyperopt.py CONFIG_PATH [OPTIONS]

Arguments:
  CONFIG_PATH              Path to hyperopt configuration YAML file

Options:
  --output-dir TEXT        Directory to save results [default: results/hyperopt]
  --n-trials INTEGER       Number of trials (overrides config)
  --study-name TEXT        Study name (overrides config)
  --resume / --no-resume   Resume existing study [default: resume]
  --storage TEXT           Optuna storage URL (overrides config)
  --use-wandb / --no-wandb Use WandB for logging [default: no-wandb]
  --help                   Show this message and exit
```

## Output Files

After running hyperparameter optimization, the following files are created in the output directory:

```
results/hyperopt/
├── hyperopt_<study_name>.log              # Detailed log file
├── <study_name>_results.yaml              # Summary of results
├── <study_name>_study.pkl                 # Pickled Optuna study object
└── checkpoints/
    └── trial_N/                           # Checkpoints for each trial
        └── best_model.pt
```

### Results File Example

```yaml
best_trial: 15
best_value: 0.8523
best_params:
  model.ff_h_dim: 1000
  model.ff_num_layers: 4
  model.dropout: 0.25
  run.learning_rate: 0.0003162
  run.batch_size: 64
  run.weight_decay: 0.001
all_trials:
  - number: 0
    value: 0.7234
    params: {...}
    state: COMPLETE
  - number: 1
    value: 0.7891
    params: {...}
    state: COMPLETE
  ...
```

## Hyperparameters to Optimize

Based on the MassFormer architecture and training, here are recommended hyperparameters to optimize:

### Model Architecture
- **`ff_h_dim`**: Hidden dimension of feed-forward layers (512-1536)
- **`ff_num_layers`**: Number of feed-forward layers (2-5)
- **`ff_skip`**: Whether to use skip connections (True/False)
- **`dropout`**: Dropout rate (0.0-0.5)

### Training
- **`learning_rate`**: Initial learning rate (1e-5 to 1e-2, log-scale)
- **`batch_size`**: Training batch size (16, 32, 64, 128)
- **`weight_decay`**: L2 regularization (1e-5 to 1e-2, log-scale)
- **`scheduler`**: LR scheduler type (polynomial, plateau)
- **`scheduler_peak_lr`**: Peak learning rate for warmup (1e-5 to 1e-3)
- **`scheduler_warmup_frac`**: Fraction of training for warmup (0.0-0.2)

### Data Augmentation (FLAG)
- **`flag`**: Whether to use FLAG augmentation (True/False)
- **`flag_m`**: Number of FLAG ascent steps (1-5)
- **`flag_step_size`**: FLAG step size (1e-4 to 1e-2, log-scale)
- **`flag_mag`**: FLAG perturbation magnitude (1e-4 to 1e-2, log-scale)

## Advanced Usage

### Parallel Optimization

Run multiple trials in parallel on different GPUs:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml \
    --storage sqlite:///results/hyperopt/study.db \
    --study-name shared_study

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml \
    --storage sqlite:///results/hyperopt/study.db \
    --study-name shared_study
```

### Analyzing Results

```python
import pickle
import optuna

# Load study
with open("results/hyperopt/massformer_nist23_scaffold_study.pkl", "rb") as f:
    study = pickle.load(f)

# Get best trial
print(f"Best value: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")

# Plot optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# Plot parameter importances
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Plot parallel coordinate plot
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
```

## Tips and Best Practices

1. **Start with fewer epochs**: Use 20-50 epochs during hyperopt to iterate faster
2. **Use pruning**: Enable median or hyperband pruner to stop bad trials early
3. **Log-scale for rates**: Use `loguniform` for learning rates and weight decay
4. **Validate search ranges**: Ensure your search ranges make sense for the dataset
5. **Monitor progress**: Use `--use-wandb` to track trials in real-time
6. **Save frequently**: Use SQLite storage to avoid losing progress
7. **Parallel execution**: Run multiple processes with shared storage for faster optimization
8. **Parameter importance**: Run at least 10-20 trials before analyzing parameter importance

## Differences from RASSP Hyperopt

The MassFormer hyperopt script differs from RASSP's in several ways:

1. **Configuration style**: Uses YAML config files like MassFormer's training script
2. **Model architecture**: Optimizes MassFormer-specific parameters (Graphormer, FLAG)
3. **Integration**: Directly calls `train_and_eval` from MassFormer's runner
4. **Metrics**: Uses validation cosine similarity as the optimization objective
5. **WandB support**: Optional WandB logging (off by default)

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in the search space
- Increase `grad_acc_interval` in fixed config
- Reduce `ff_h_dim` search range

### Slow Optimization
- Reduce `num_epochs` in fixed config (e.g., 20-30 epochs)
- Enable `amp: True` for mixed precision training
- Use multiple GPUs in parallel

### All Trials Pruned
- Check that data paths are correct
- Verify that the model can train on a single trial first
- Adjust pruner settings (increase `n_warmup_steps`)

### Storage Issues
- Use absolute paths for SQLite storage
- Ensure write permissions for output directory
- Use `--no-resume` to start fresh if study is corrupted

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MassFormer Paper](https://www.nature.com/articles/s42256-023-00708-3)
- [Hyperparameter Optimization Best Practices](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
