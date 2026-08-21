"""Hyperparameter optimization for RASSP using Optuna with TPE sampler.

Usage:
    python hyperopt.py expconfig/hyperopt.yaml [--output-dir results/hyperopt]

Example:
    python hyperopt.py expconfig/hyperopt.yaml --n-trials 100 --study-name my_study
"""

import logging
import os
import pickle
import resource as res
import traceback
from typing import Any

import click
import numpy as np
import optuna
import torch
import yaml
from torch import nn
from tqdm import tqdm

from rassp import dataset, netutil, util
from rassp.metrics import convert_svect_to_sdict, dp, sdp
from rassp.model import (
    formulaenets,  # noqa: F401
    losses,  # noqa: F401
    nets,  # noqa: F401
    subsetnets,  # noqa: F401
)
from rassp.msutil import binutils

USE_CUDA = os.environ.get("USE_CUDA", "1") == "1"
DATALOADER_PIN_MEMORY = False


def sample_hyperparameters(
    trial: optuna.Trial, search_space: dict, prefix: str = ""
) -> dict:
    """Recursively sample hyperparameters from search space definition."""
    params = {}

    for key, value in search_space.items():
        param_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            if "type" in value:
                # This is a parameter definition
                param_type = value["type"]

                if param_type == "int":
                    params[key] = trial.suggest_int(
                        param_name,
                        value["low"],
                        value["high"],
                        step=value.get("step", 1),
                    )
                elif param_type == "float":
                    params[key] = trial.suggest_float(
                        param_name,
                        value["low"],
                        value["high"],
                        step=value.get("step"),
                    )
                elif param_type == "loguniform":
                    params[key] = trial.suggest_float(
                        param_name, value["low"], value["high"], log=True
                    )
                elif param_type == "categorical":
                    params[key] = trial.suggest_categorical(
                        param_name, value["choices"]
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            else:
                # This is a nested dict, recurse
                params[key] = sample_hyperparameters(
                    trial, value, prefix=f"{param_name}."
                )
        else:
            # Fixed value
            params[key] = value

    return params


def build_exp_config(
    hyperopt_config: dict, sampled_params: dict, trial_number: int
) -> dict:
    """Build a complete experiment config from hyperopt config and sampled params."""
    fixed = hyperopt_config["fixed"]
    training = hyperopt_config["training"]
    data_config = hyperopt_config["data"]

    # Start with fixed parameters
    exp_config = {
        "seed": training["seed"],
        "tgt_max_n": fixed["tgt_max_n"],
        "bin_config": fixed["bin_config"],
        "loss_params": fixed["loss_params"],
        "featurize_config": fixed["featurize_config"],
        "pred_config": fixed["pred_config"],
        "validate_config": fixed["validate_config"],
        "net_name": fixed["net_name"],
        "max_epochs": training["max_epochs"],
        "validate_every": training["validate_every"],
        "checkpoint_every_n_epochs": training["checkpoint_every_n_epochs"],
        "epoch_size": training["epoch_size"],
        "DATALOADER_NUM_WORKERS": 0,
        "tblogdir": f"tblogs.hyperopt/trial_{trial_number}",
    }

    # Add cluster config for data paths
    exp_config["cluster_config"] = {
        "data_dir": data_config["data_dir"],
        "checkpoint_dir": f"checkpoints/hyperopt/trial_{trial_number}",
        "using_cluster": False,
    }

    # Add data configuration
    exp_config["exp_data"] = {
        "data": [
            {
                "db_filename": data_config["train_file"],
                "phase": "train",
                "filter_max_mass": fixed["filter_max_mass"],
                "filter_max_unique_formulae": fixed[
                    "filter_max_unique_formulae"
                ],
                "filter_max_n": fixed["filter_max_n"],
            },
            {
                "db_filename": data_config["val_file"],
                "phase": "test",
                "filter_max_mass": fixed["filter_max_mass"],
                "filter_max_unique_formulae": fixed[
                    "filter_max_unique_formulae"
                ],
                "filter_max_n": fixed["filter_max_n"],
            },
        ],
        "cv_split": {
            "how": "morgan_fingerprint_mod",
            "mod": 10,
            "test": [0, 1],
        },
    }

    # Add sampled hyperparameters
    if "net_params" in sampled_params:
        net_params = sampled_params["net_params"].copy()
        # Ensure required fixed params (will be set during training)
        net_params["g_feature_n"] = -1

        # formula_oh_sizes and formula_oh_accum go ONLY in spect_out_config, not top-level
        if "spect_out_config" in net_params:
            net_params["spect_out_config"]["formula_oh_sizes"] = [
                50,
                46,
                30,
                30,
                30,
                30,
                30,
                30,
            ]
            net_params["spect_out_config"]["formula_oh_accum"] = True

        exp_config["net_params"] = net_params

    if "opt_params" in sampled_params:
        exp_config["opt_params"] = sampled_params["opt_params"]

    if "batch_size" in sampled_params:
        exp_config["batch_size"] = sampled_params["batch_size"]

    if "accumulate_steps" in sampled_params:
        exp_config["accumulate_steps"] = sampled_params["accumulate_steps"]

    return exp_config


def compute_validation_metrics(
    net: nn.Module,
    dl_test: torch.utils.data.DataLoader,
    spect_bin_config: Any,
    use_cuda: bool = True,
) -> dict:
    """Compute validation metrics (SDP, DP) on test set."""
    net.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Validating", leave=False):
            # Handle sparse data
            for k, v in batch.items():
                if "sparse" in k:
                    batch[k] = torch.sparse.FloatTensor(
                        v["inds"],
                        v["vals"],
                        v["shape"],
                    )

            batch_t = {k: util.move(v, use_cuda) for k, v in batch.items()}
            res = net(**batch_t)

            pred_spect = res["spect"].cpu().numpy()
            true_spect = batch_t["spect"].cpu().numpy()

            all_pred.append(pred_spect)
            all_true.append(true_spect)

    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)

    # Compute metrics
    sdp_scores = []
    dp_scores = []

    for i in range(len(all_pred)):
        pred_dict = convert_svect_to_sdict(all_pred[i])
        true_dict = convert_svect_to_sdict(all_true[i])

        sdp_scores.append(sdp(true_dict, pred_dict))
        dp_scores.append(dp(true_dict, pred_dict))

    return {
        "sdp": np.mean(sdp_scores),
        "dp": np.mean(dp_scores),
        "sdp_std": np.std(sdp_scores),
        "dp_std": np.std(dp_scores),
    }


def train_single_trial(
    exp_config: dict,
    trial: optuna.Trial,
    use_cuda: bool = True,
    early_stop_patience: int = 5,
) -> float:
    """Train a single trial and return the validation metric."""
    data_dir = exp_config["cluster_config"]["data_dir"]
    checkpoint_dir = exp_config["cluster_config"]["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    np.random.seed(exp_config["seed"])
    torch.manual_seed(exp_config["seed"])

    MAX_N = exp_config["tgt_max_n"]
    BATCH_SIZE = exp_config["batch_size"]

    spect_bin_config = binutils.create_spectrum_bins(
        **exp_config["bin_config"]
    )

    featurize_config_update = exp_config["featurize_config"]
    featurize_config = netutil.DEFAULT_FEATURIZE_CONFIG.copy()
    util.recursive_update(featurize_config, featurize_config_update)

    pred_config_update = exp_config["pred_config"]
    pred_config = netutil.DEFAULT_PRED_CONFIG.copy()
    util.recursive_update(pred_config, pred_config_update)

    exp_data = exp_config["exp_data"]
    cv_func = netutil.CVSplit(**exp_data["cv_split"])

    # Load datasets
    datasets_dict = {}
    for dataset_config in exp_data["data"]:
        dataset_config["db_filename"] = os.path.join(
            data_dir, dataset_config["db_filename"]
        )

        if not os.path.exists(dataset_config["db_filename"]):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_config['db_filename']}"
            )

        basename = os.path.basename(dataset_config["db_filename"])
        ext = basename.split(".")[-1]

        if ext in ["parquet", "pq"]:
            ds = dataset.load_pq_dataset(
                dataset_config,
                spect_bin_config,
                featurize_config,
                pred_config,
            )
        else:
            raise ValueError(f"Unsupported dataset extension: {ext}")

        phase = dataset_config["phase"]
        if phase not in datasets_dict:
            datasets_dict[phase] = []
        datasets_dict[phase].append(ds)

    ds_train = (
        datasets_dict["train"][0]
        if len(datasets_dict["train"]) == 1
        else torch.utils.data.ConcatDataset(datasets_dict["train"])
    )
    ds_test = (
        datasets_dict["test"][0]
        if len(datasets_dict["test"]) == 1
        else torch.utils.data.ConcatDataset(datasets_dict["test"])
    )

    logging.info(
        f"Training with {len(ds_train)} samples, validating with {len(ds_test)}"
    )

    # Create data loaders
    epoch_size = exp_config.get("epoch_size", 8192)
    train_sampler = netutil.SubsetSampler(
        epoch_size, len(ds_train), shuffle=True
    )
    test_sampler = netutil.SubsetSampler(
        min(epoch_size, len(ds_test)), len(ds_test), shuffle=True
    )

    DATALOADER_NUM_WORKERS = exp_config.get("DATALOADER_NUM_WORKERS", 0)

    # timeout only valid when num_workers > 0
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "pin_memory": DATALOADER_PIN_MEMORY,
        "num_workers": DATALOADER_NUM_WORKERS,
    }
    if DATALOADER_NUM_WORKERS > 0:
        dataloader_kwargs["timeout"] = 60 * 4

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        sampler=train_sampler,
        **dataloader_kwargs,
    )

    dl_test = torch.utils.data.DataLoader(
        ds_test,
        sampler=test_sampler,
        **dataloader_kwargs,
    )

    # Build network
    net_params = exp_config["net_params"]
    net_name = exp_config["net_name"]

    n_atom_feats = ds_test[0]["vect_feat"].shape[-1]
    net_params["g_feature_n"] = n_atom_feats
    net_params["GS"] = ds_test[0]["adj"].shape[0]
    net_params["spect_bin"] = spect_bin_config

    net = eval(net_name)(**net_params)
    net = util.move(net, use_cuda)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # Loss and optimizer
    loss_params = exp_config["loss_params"]
    criterion = netutil.create_loss(loss_params, use_cuda)

    opt_params = exp_config["opt_params"]
    optimizer = netutil.create_optimizer(opt_params, net.parameters())

    # Training loop
    n_epochs = exp_config["max_epochs"]
    validate_every = exp_config.get("validate_every", 1)
    accumulate_steps = exp_config.get("accumulate_steps", 1)

    best_metric = -float("inf")
    epochs_without_improvement = 0

    for epoch_i in tqdm(range(n_epochs), desc="Training"):
        net.train()

        # Training epoch
        train_res = netutil.run_epoch(
            net,
            optimizer,
            criterion,
            dl_train,
            pred_only=False,
            USE_CUDA=use_cuda,
            return_pred=False,
            progress_bar=True,  # Enable to see batch progress
            desc=f"Epoch {epoch_i}",
            accumulate_steps=accumulate_steps,
        )

        # Validation
        if epoch_i % validate_every == 0:
            metrics = compute_validation_metrics(
                net, dl_test, spect_bin_config, use_cuda
            )

            # Report to Optuna for pruning
            trial.report(metrics["sdp"], epoch_i)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping check
            if metrics["sdp"] > best_metric:
                best_metric = metrics["sdp"]
                epochs_without_improvement = 0

                # Save best checkpoint
                torch.save(
                    net.state_dict(),
                    os.path.join(checkpoint_dir, "best_model.state"),
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch_i}")
                break

            logging.info(
                f"Epoch {epoch_i}: train_loss={train_res['mean_loss']:.4f}, "
                f"val_sdp={metrics['sdp']:.4f}, val_dp={metrics['dp']:.4f}"
            )

    return best_metric


def objective(
    trial: optuna.Trial, hyperopt_config: dict, use_cuda: bool
) -> float:
    """Optuna objective function."""
    # Sample hyperparameters
    search_space = hyperopt_config["search_space"]
    sampled_params = sample_hyperparameters(trial, search_space)

    # Build experiment config
    exp_config = build_exp_config(
        hyperopt_config, sampled_params, trial.number
    )

    # Log sampled parameters
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Trial {trial.number}")
    logging.info(f"{'=' * 60}")
    logging.info("Sampled parameters:")
    for key, value in trial.params.items():
        logging.info(f"  {key}: {value}")

    try:
        metric = train_single_trial(exp_config, trial, use_cuda)
        return metric
    except optuna.TrialPruned:
        raise  # Re-raise pruning exceptions without modification
    except Exception as e:
        error_msg = (
            f"Trial {trial.number} failed with error: {type(e).__name__}: {e}"
        )
        logging.error(error_msg)
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        raise optuna.TrialPruned()


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    default="results/hyperopt",
    help="Directory to save results",
)
@click.option(
    "--n-trials",
    default=None,
    type=int,
    help="Number of trials (overrides config)",
)
@click.option(
    "--study-name",
    default=None,
    help="Study name (overrides config)",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume existing study if it exists",
)
@click.option(
    "--storage", default=None, help="Optuna storage URL (overrides config)"
)
def main(
    config_path: str,
    output_dir: str,
    n_trials: int,
    study_name: str,
    storage: str,
    resume: bool,
):
    """Run hyperparameter optimization for RASSP."""
    # Increase file limits
    file_limit_soft, file_limit_hard = res.getrlimit(res.RLIMIT_NOFILE)
    res.setrlimit(res.RLIMIT_NOFILE, (file_limit_hard, file_limit_hard))

    # Load config
    with open(config_path, "r") as f:
        hyperopt_config = yaml.load(f, Loader=yaml.FullLoader)

    optuna_config = hyperopt_config["optuna"]

    # Override config with CLI args if provided
    if n_trials is not None:
        optuna_config["n_trials"] = n_trials
    if study_name is not None:
        optuna_config["study_name"] = study_name
    if storage is not None:
        optuna_config["storage"] = storage

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    log_filename = os.path.join(
        output_dir, f"hyperopt_{optuna_config['study_name']}.log"
    )

    # Configure logging properly by getting the root logger directly
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters and handlers
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(
        log_filename, mode="a" if resume else "w"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Set up sampler
    sampler_name = optuna_config.get("sampler", "tpe")
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=hyperopt_config["training"]["seed"]
        )
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(
            seed=hyperopt_config["training"]["seed"]
        )
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(
            seed=hyperopt_config["training"]["seed"]
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Set up pruner
    pruner_name = optuna_config.get("pruner", "median")
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10
        )
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")

    # Determine direction
    direction = optuna_config.get("direction", "maximize")

    # Create or load study
    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        storage=optuna_config.get("storage"),
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=resume,
    )

    # Run optimization
    use_cuda = USE_CUDA and torch.cuda.is_available()
    logging.info(f"Using CUDA: {use_cuda}")

    study.optimize(
        lambda trial: objective(trial, hyperopt_config, use_cuda),
        n_trials=optuna_config["n_trials"],
        show_progress_bar=True,
    )

    # Save results
    logging.info("\n" + "=" * 60)
    logging.info("Optimization complete!")
    logging.info("=" * 60)

    # Check if any trials completed successfully
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if len(completed_trials) == 0:
        logging.info("\nWARNING: No trials completed successfully!")
        logging.info(
            "All trials were pruned or failed. Check the error messages above."
        )

        # Still save what we have for debugging
        results = {
            "best_trial": None,
            "best_value": None,
            "best_params": None,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }
    else:
        logging.info(f"\nBest trial: {study.best_trial.number}")
        logging.info(f"Best value: {study.best_trial.value:.4f}")
        logging.info("\nBest parameters:")
        for key, value in study.best_trial.params.items():
            logging.info(f"  {key}: {value}")

        # Save study results
        results = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }

    results_path = os.path.join(
        output_dir, f"{optuna_config['study_name']}_results.yaml"
    )
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    logging.info(f"\nResults saved to: {results_path}")

    # Also save as pickle for more detailed analysis
    pickle_path = os.path.join(
        output_dir, f"{optuna_config['study_name']}_study.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(study, f)

    logging.info(f"Study object saved to: {pickle_path}")

    # Generate importance analysis if enough completed trials
    if len(completed_trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            logging.info("\nParameter importance:")
            for param, imp in importance.items():
                logging.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            logging.info(f"Could not compute parameter importance: {e}")


if __name__ == "__main__":
    main()
