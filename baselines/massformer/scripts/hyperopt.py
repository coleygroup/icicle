"""Hyperparameter optimization for MassFormer using Optuna with TPE sampler.

Usage:
    python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml [--output-dir results/hyperopt]

Example:
    python scripts/hyperopt.py config/hyperopt_nist23_gcms.yml --n-trials 100 --study-name massformer_nist23
"""

import logging
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import click
import numpy as np
import optuna
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from massformer.misc_utils import np_temp_seed
from massformer.runner import load_config, train_and_eval


def sample_hyperparameters(
    trial: optuna.Trial, search_space: dict, prefix: str = ""
) -> dict:
    """Recursively sample hyperparameters from search space definition.

    Args:
        trial: Optuna trial object
        search_space: Dictionary defining the hyperparameter search space
        prefix: Prefix for nested parameter names

    Returns:
        Dictionary of sampled hyperparameters
    """
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
                        float(value["low"]),
                        float(value["high"]),
                        step=float(value["step"]) if value.get("step") is not None else None,
                    )
                elif param_type == "loguniform":
                    params[key] = trial.suggest_float(
                        param_name, float(value["low"]), float(value["high"]), log=True
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


def update_config_from_params(
    base_config: dict, sampled_params: dict, section: str
) -> dict:
    """Update a config section with sampled parameters.

    Args:
        base_config: Base configuration dictionary
        sampled_params: Sampled hyperparameters
        section: Config section name ('data', 'model', or 'run')

    Returns:
        Updated configuration dictionary
    """
    config = base_config.copy()

    for key, value in sampled_params.items():
        if isinstance(value, dict):
            # Handle nested updates
            if key not in config:
                config[key] = {}
            config[key].update(value)
        else:
            config[key] = value

    return config


def build_trial_configs(
    hyperopt_config: dict,
    sampled_params: dict,
    trial_number: int,
    output_dir: str
) -> "tuple[dict, dict, dict]":
    """Build data, model, and run configs from hyperopt config and sampled params.

    Args:
        hyperopt_config: Full hyperopt configuration
        sampled_params: Sampled hyperparameters for this trial
        trial_number: Trial number
        output_dir: Output directory for results

    Returns:
        Tuple of (data_config, model_config, run_config)
    """
    # Start from template defaults so all required keys are present
    template_fp = hyperopt_config.get("template_fp", os.path.join(
        os.path.dirname(__file__), "..", "config", "template.yml"
    ))
    with open(template_fp) as f:
        template = yaml.load(f, Loader=yaml.FullLoader)

    data_config = template["data"].copy()
    model_config = template["model"].copy()
    run_config = template["run"].copy()

    # Apply hyperopt config overrides on top of template
    for k, v in hyperopt_config.get("data", {}).items():
        data_config[k] = v
    for k, v in hyperopt_config.get("model", {}).items():
        model_config[k] = v
    for k, v in hyperopt_config.get("run", {}).items():
        run_config[k] = v

    # Update with sampled parameters
    if "data" in sampled_params:
        data_config = update_config_from_params(
            data_config, sampled_params["data"], "data"
        )
    if "model" in sampled_params:
        model_config = update_config_from_params(
            model_config, sampled_params["model"], "model"
        )
    if "run" in sampled_params:
        run_config = update_config_from_params(
            run_config, sampled_params["run"], "run"
        )

    # Set trial-specific settings
    trial_checkpoint_dir = os.path.join(output_dir, "checkpoints", f"trial_{trial_number}")
    os.makedirs(trial_checkpoint_dir, exist_ok=True)

    # Update data config with checkpoint directory
    data_config["checkpoint_dp"] = trial_checkpoint_dir

    return data_config, model_config, run_config


def train_single_trial(
    entity_name: str,
    project_name: str,
    run_name: str,
    data_config: dict,
    model_config: dict,
    run_config: dict,
    trial: optuna.Trial,
    use_wandb: bool = False,
) -> float:
    """Train a single trial and return the validation metric.

    Args:
        entity_name: WandB entity name
        project_name: WandB project name
        run_name: WandB run name
        data_config: Data configuration
        model_config: Model configuration
        run_config: Run configuration
        trial: Optuna trial object
        use_wandb: Whether to use WandB

    Returns:
        Validation metric (to be maximized)
    """
    # Import here to avoid circular imports
    from massformer.runner import init_wandb_run

    # Initialize WandB if needed
    if use_wandb:
        trial_run_name = f"{run_name}_trial_{trial.number}"
        group_name = f"{run_name}_hyperopt"

        init_wandb_run(
            entity_name=entity_name,
            project_name=project_name,
            run_name=trial_run_name,
            data_d=data_config,
            model_d=model_config,
            run_d=run_config,
            wandb_meta_dp=data_config.get("checkpoint_dp", "."),
            group_name=group_name,
            wandb_mode="online",
        )

    # Train and evaluate
    try:
        results = train_and_eval(data_config, model_config, run_config, use_wandb)

        # Extract validation metric
        # MassFormer returns metrics in a dict - use the validation cosine similarity
        if results and "val_metrics" in results:
            val_metric = results["val_metrics"].get("mol_sim_obj_mean", 0.0)
        else:
            # Fallback: try to get from last epoch results
            val_metric = results.get("best_val_sim", 0.0)

        # Report intermediate values for pruning
        if use_wandb:
            import wandb
            # Get validation metric from wandb history if available
            if wandb.run:
                history = wandb.run.history()
                if "mol_sim_obj_mean" in history.columns:
                    for epoch, metric in enumerate(history["mol_sim_obj_mean"]):
                        if not np.isnan(metric):
                            trial.report(metric, epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                wandb.finish()

        return val_metric

    except optuna.TrialPruned:
        if use_wandb:
            import wandb
            if wandb.run:
                wandb.finish()
        raise
    except Exception as e:
        if use_wandb:
            import wandb
            if wandb.run:
                wandb.finish()
        raise e


def objective(
    trial: optuna.Trial,
    hyperopt_config: dict,
    output_dir: str,
    use_wandb: bool = False
) -> float:
    """Optuna objective function.

    Args:
        trial: Optuna trial object
        hyperopt_config: Hyperopt configuration
        output_dir: Output directory
        use_wandb: Whether to use WandB

    Returns:
        Validation metric to maximize
    """
    # Sample hyperparameters
    search_space = hyperopt_config["search_space"]
    sampled_params = sample_hyperparameters(trial, search_space)

    # Build configs
    data_config, model_config, run_config = build_trial_configs(
        hyperopt_config, sampled_params, trial.number, output_dir
    )

    # Log sampled parameters
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Trial {trial.number}")
    logging.info(f"{'=' * 60}")
    logging.info("Sampled parameters:")
    for key, value in trial.params.items():
        logging.info(f"  {key}: {value}")

    try:
        # Get entity/project/run names
        entity_name = hyperopt_config.get("entity_name", "")
        project_name = hyperopt_config.get("project_name", "massformer_hyperopt")
        run_name = hyperopt_config.get("run_name", "hyperopt")

        metric = train_single_trial(
            entity_name=entity_name,
            project_name=project_name,
            run_name=run_name,
            data_config=data_config,
            model_config=model_config,
            run_config=run_config,
            trial=trial,
            use_wandb=use_wandb,
        )

        logging.info(f"Trial {trial.number} completed with metric: {metric:.4f}")
        return metric

    except optuna.TrialPruned:
        logging.info(f"Trial {trial.number} pruned")
        raise
    except Exception as e:
        error_msg = (
            f"Trial {trial.number} failed with error: {type(e).__name__}: {e}"
        )
        logging.error(error_msg)
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        # Prune failed trials
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
    "--storage",
    default=None,
    help="Optuna storage URL (overrides config)",
)
@click.option(
    "--use-wandb/--no-wandb",
    default=False,
    help="Use WandB for logging",
)
def main(
    config_path: str,
    output_dir: str,
    n_trials: int,
    study_name: str,
    storage: str,
    resume: bool,
    use_wandb: bool,
):
    """Run hyperparameter optimization for MassFormer."""
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

    # Configure logging
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
    seed = hyperopt_config.get("run", {}).get("train_seed", 112233)

    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Set up pruner
    pruner_name = optuna_config.get("pruner", "median")
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5
        )
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")

    # Determine direction (maximize cosine similarity)
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
    use_cuda = torch.cuda.is_available()
    logging.info(f"Using CUDA: {use_cuda}")
    logging.info(f"Using WandB: {use_wandb}")

    study.optimize(
        lambda trial: objective(trial, hyperopt_config, output_dir, use_wandb),
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
