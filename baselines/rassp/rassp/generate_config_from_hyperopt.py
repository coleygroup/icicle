"""Generate a forward_train.py compatible config from hyperopt results.

Usage:
    python generate_config_from_hyperopt.py results.yaml hyperopt_config.yaml -o best_config.yaml
"""

import click
import yaml


def unflatten_params(flat_params: dict) -> dict:
    """Convert dot-notation params to nested dict.

    E.g., {"net_params.int_d": 512} -> {"net_params": {"int_d": 512}}
    """
    result = {}
    for key, value in flat_params.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


@click.command()
@click.argument("results_yaml", type=click.Path(exists=True))
@click.argument("hyperopt_config", type=click.Path(exists=True))
@click.option("-o", "--output", default="best_config.yaml", help="Output config path")
@click.option("--max-epochs", default=200, help="Max epochs for final training")
def main(results_yaml: str, hyperopt_config: str, output: str, max_epochs: int):
    """Generate training config from hyperopt results."""
    # Load results
    with open(results_yaml) as f:
        results = yaml.safe_load(f)

    if results.get("best_params") is None:
        raise ValueError("No best_params found in results - no trials completed?")

    # Load hyperopt config for fixed params
    with open(hyperopt_config) as f:
        hyperopt_cfg = yaml.safe_load(f)

    # Unflatten the best params
    best_params = unflatten_params(results["best_params"])

    fixed = hyperopt_cfg["fixed"]
    data_cfg = hyperopt_cfg["data"]

    # Build the training config
    config = {
        "cluster_config": {
            "data_dir": data_cfg["data_dir"],
            "checkpoint_dir": "checkpoints",
            "using_cluster": False,
        },
        "exp_data": {
            "data": [
                {
                    "db_filename": data_cfg["train_file"],
                    "phase": "train",
                    "filter_max_mass": fixed["filter_max_mass"],
                    "filter_max_unique_formulae": fixed["filter_max_unique_formulae"],
                    "filter_max_n": fixed["filter_max_n"],
                },
                {
                    "db_filename": data_cfg["val_file"],
                    "phase": "test",
                    "filter_max_mass": fixed["filter_max_mass"],
                    "filter_max_unique_formulae": fixed["filter_max_unique_formulae"],
                    "filter_max_n": fixed["filter_max_n"],
                },
            ],
            "cv_split": {
                "how": "morgan_fingerprint_mod",
                "mod": 10,
                "test": [0, 1],
            },
        },
        "tblogdir": "tblogs.best",
        "checkpoint_every_n_epochs": 50,
        "DATALOADER_NUM_WORKERS": 4,
        "validate_every": 10,
        "pred_config": fixed["pred_config"],
        "validate_config": fixed["validate_config"],
        "net_name": fixed["net_name"],
        "bin_config": fixed["bin_config"],
        "automatic_mixed_precision": False,
        "epoch_size": hyperopt_cfg["training"]["epoch_size"],
        "tgt_max_n": fixed["tgt_max_n"],
        "seed": hyperopt_cfg["training"]["seed"],
        "featurize_config": fixed["featurize_config"],
        "loss_params": fixed["loss_params"],
        "max_epochs": max_epochs,
    }

    # Add best hyperparameters
    if "batch_size" in best_params:
        config["batch_size"] = best_params["batch_size"]

    if "accumulate_steps" in best_params:
        config["accumulate_steps"] = best_params["accumulate_steps"]

    if "opt_params" in best_params:
        config["opt_params"] = best_params["opt_params"]

    if "net_params" in best_params:
        net_params = best_params["net_params"]
        net_params["g_feature_n"] = -1  # Set at runtime
        # Add formula_oh config to spect_out_config
        if "spect_out_config" in net_params:
            net_params["spect_out_config"]["formula_oh_sizes"] = [50, 46, 30, 30, 30, 30, 30, 30]
            net_params["spect_out_config"]["formula_oh_accum"] = True
        config["net_params"] = net_params

    # Write output
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated config: {output}")
    print(f"Best trial: {results['best_trial']}")
    print(f"Best value (SDP): {results['best_value']:.4f}")
    print(f"\nRun with: python forward_train.py {output}")


if __name__ == "__main__":
    main()
