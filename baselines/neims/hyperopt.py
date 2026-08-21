"""Hyperparameter optimization for NEIMS and NEIMS-GNN using Optuna."""

import argparse
import json
import math
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from neims.model import generalized_mse_loss
from optuna.trial import Trial
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    mass_power,
    model_type="neims",
    loss_type="generalized_mse",
    max_batches=None,
):
    """Train model for one epoch (or subset of batches)."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(
        tqdm(train_loader, desc="Training", leave=False)
    ):
        if max_batches and batch_idx >= max_batches:
            break

        if model_type == "neims_gnn":
            batch = batch.to(device)
            masses = batch.mass
            spectra = batch.y
            predictions = model(batch, masses)
        else:
            fingerprints = batch["fingerprint"].to(device)
            masses = batch["mass"].to(device)
            spectra = batch["spectrum"].to(device)
            predictions = model(fingerprints, masses)

        if loss_type == "mse":
            loss = nn.functional.mse_loss(predictions, spectra)
        else:
            loss = generalized_mse_loss(predictions, spectra, mass_power)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def validate_model(
    model, val_loader, device, mass_power, model_type="neims", loss_type="generalized_mse"
):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if model_type == "neims_gnn":
                batch = batch.to(device)
                masses = batch.mass
                spectra = batch.y
                predictions = model(batch, masses)
            else:
                fingerprints = batch["fingerprint"].to(device)
                masses = batch["mass"].to(device)
                spectra = batch["spectrum"].to(device)
                predictions = model(fingerprints, masses)

            if loss_type == "mse":
                loss = nn.functional.mse_loss(predictions, spectra)
            else:
                loss = generalized_mse_loss(predictions, spectra, mass_power)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def objective_neims(trial: Trial, config: dict):
    """Optuna objective function for NEIMS (fingerprint-based)."""
    from neims.data import create_dataloaders
    from neims.model import NEIMS

    print(f"\n{'=' * 60}")
    print(f"Trial {trial.number} [NEIMS]: Starting...")
    print(f"{'=' * 60}")

    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 4, 10)
    hidden_size = trial.suggest_categorical(
        "hidden_size", [1000, 1500, 2000, 2500]
    )
    resnet_bottleneck = trial.suggest_float("resnet_bottleneck", 0.3, 0.7)
    mass_power = trial.suggest_float("mass_power", 0.3, 0.7)
    decay_scale = trial.suggest_float("decay_scale", 500.0, 2000.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    gate_bidirectional = (
        trial.suggest_categorical("gate_bidirectional", [True, False])
        if bidirectional
        else False
    )

    hidden_sizes = [hidden_size] * num_layers

    print("Sampled hyperparameters:")
    print(
        f"  lr={learning_rate:.2e}, dropout={dropout:.2f}, batch_size={batch_size}"
    )
    print(f"  num_layers={num_layers}, hidden_size={hidden_size}")

    # Create dataloaders with sampled batch size
    train_loader, val_loader, _, output_size = create_dataloaders(
        metadata_path=config["metadata_path"],
        spectra_path=config["spectra_path"],
        splits_path=config["splits_path"],
        batch_size=batch_size,
        num_workers=config["num_workers"],
        fp_radius=config["fp_radius"],
        fp_length=config["fp_length"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        bin_width=config["bin_width"],
    )

    # Create model
    model = NEIMS(
        input_size=config["fp_length"],
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        bidirectional=bidirectional,
        gate_bidirectional=gate_bidirectional,
        resnet_bottleneck=resnet_bottleneck,
        max_mass_offset=config["max_mass_offset"],
        fp_radius=config["fp_radius"],
        fp_length=config["fp_length"],
    ).to(config["device"])

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, eps=1e-7
    )

    def lr_lambda(step):
        return max(1.0 / math.sqrt(1.0 + step / decay_scale), 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train for a few epochs
    for epoch in range(config["n_epochs"]):
        train_loss = train_model(
            model,
            train_loader,
            optimizer,
            scheduler,
            config["device"],
            mass_power,
            model_type="neims",
        )
        val_loss = validate_model(
            model, val_loader, config["device"], mass_power,
            model_type="neims",
        )

        print(
            f"  Epoch {epoch + 1}/{config['n_epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        trial.report(val_loss, epoch)

        if trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch + 1}")
            raise optuna.TrialPruned()

    return val_loss


def objective_neims_gnn(trial: Trial, config: dict):
    """Optuna objective function for NEIMS-GNN (graph-based).

    Search space informed by Zhu et al. 2020 (GAT wins, GLU wins, max pool wins)
    and empirical over-smoothing observed at 270k scale (layers capped at 6).
    """
    from neims.gnn_data import create_gnn_dataloaders
    from neims.gnn_model import NEIMSGNN

    print(f"\n{'=' * 60}")
    print(f"Trial {trial.number} [NEIMS-GNN]: Starting...")
    print(f"{'=' * 60}")

    # Optimizer
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    decay_scale = trial.suggest_float("decay_scale", 500.0, 3000.0)
    # L2=1.0 (paper) collapses at 270k scale; cap at 1e-2
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # GNN encoder — GAT only (paper: GAT >> GCN)
    # Layers 2–6: over-smoothing observed with >=7 on 270k
    gnn_num_layers = trial.suggest_int("gnn_num_layers", 2, 6)
    gnn_hidden_size = trial.suggest_categorical("gnn_hidden_size", [64, 128, 256, 512])
    gnn_num_heads = trial.suggest_categorical("gnn_num_heads", [4, 8])
    gnn_dropout = trial.suggest_float("gnn_dropout", 0.0, 0.4)
    pool_type = "max"
    use_edge_features = trial.suggest_categorical("use_edge_features", [True, False])

    # Loss — plain MSE as in paper (Zhu 2020 Section 2.4)
    mass_power = 1.0

    # Prediction head — GLU + max pool fixed (paper best, bidir spreads density)
    bidirectional = False
    gate_bidirectional = False

    # Ensure hidden_size divisible by num_heads
    if gnn_hidden_size % gnn_num_heads != 0:
        gnn_hidden_size = (gnn_hidden_size // gnn_num_heads) * gnn_num_heads
        if gnn_hidden_size == 0:
            gnn_hidden_size = gnn_num_heads

    print("Sampled hyperparameters:")
    print(f"  lr={learning_rate:.2e}, batch_size={batch_size}, wd={weight_decay:.2e}")
    print(
        f"  gnn_hidden={gnn_hidden_size}, gnn_layers={gnn_num_layers}, "
        f"heads={gnn_num_heads}, pool={pool_type}, edge_feat={use_edge_features}"
    )

    # Create dataloaders
    train_loader, val_loader, _, output_size = create_gnn_dataloaders(
        metadata_path=config["metadata_path"],
        spectra_path=config["spectra_path"],
        splits_path=config["splits_path"],
        batch_size=batch_size,
        num_workers=config["num_workers"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        bin_width=config["bin_width"],
        use_edge_features=use_edge_features,
    )

    # Create model — GAT + GLU fixed (paper best); other params searched
    model = NEIMSGNN(
        output_size=output_size,
        gnn_type="GAT",
        gnn_hidden_size=gnn_hidden_size,
        gnn_num_layers=gnn_num_layers,
        gnn_num_heads=gnn_num_heads,
        gnn_dropout=gnn_dropout,
        pool_type=pool_type,
        use_edge_features=use_edge_features,
        ffnn_hidden_sizes=[],
        ffnn_dropout=0.25,
        resnet_bottleneck=0.5,
        use_glu=True,
        bidirectional=bidirectional,
        gate_bidirectional=gate_bidirectional,
        max_mass_offset=config["max_mass_offset"],
        max_mz=config["max_mz"],
    ).to(config["device"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, eps=1e-7,
        weight_decay=weight_decay,
    )

    def lr_lambda(step):
        return max(1.0 / math.sqrt(1.0 + step / decay_scale), 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train for a few epochs
    for epoch in range(config["n_epochs"]):
        train_loss = train_model(
            model,
            train_loader,
            optimizer,
            scheduler,
            config["device"],
            mass_power,
            model_type="neims_gnn",
            loss_type="generalized_mse",
        )
        val_loss = validate_model(
            model, val_loader, config["device"], mass_power,
            model_type="neims_gnn",
            loss_type="generalized_mse",
        )

        print(
            f"  Epoch {epoch + 1}/{config['n_epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        trial.report(val_loss, epoch)

        if trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch + 1}")
            raise optuna.TrialPruned()

    return val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for NEIMS / NEIMS-GNN"
    )

    # Model type
    parser.add_argument(
        "--model-type",
        type=str,
        default="neims",
        choices=["neims", "neims_gnn"],
        help="Model type: neims (fingerprint) or neims_gnn (graph)",
    )

    # Data arguments
    parser.add_argument(
        "--metadata-path", type=str, required=True, help="Path to metadata TSV"
    )
    parser.add_argument(
        "--spectra-path", type=str, required=True, help="Path to spectra HDF5"
    )
    parser.add_argument(
        "--splits-path", type=str, required=True, help="Path to splits TSV"
    )

    # Fixed NEIMS model arguments (only used when model-type=neims)
    parser.add_argument(
        "--fp-radius", type=int, default=2, help="Morgan fingerprint radius"
    )
    parser.add_argument(
        "--fp-length", type=int, default=4096, help="Morgan fingerprint length"
    )

    # Shared fixed arguments
    parser.add_argument(
        "--max-mass-offset", type=int, default=5, help="Maximum mass offset"
    )

    # Spectrum arguments
    parser.add_argument(
        "--min-mz", type=float, default=0.0, help="Minimum m/z"
    )
    parser.add_argument(
        "--max-mz", type=float, default=750.0, help="Maximum m/z"
    )
    parser.add_argument(
        "--bin-width", type=float, default=1.0, help="Bin width"
    )

    # Optimization arguments
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=20, help="Number of epochs per trial"
    )
    parser.add_argument(
        "--study-name", type=str, default=None, help="Study name (defaults to model type)"
    )
    parser.add_argument(
        "--storage", type=str, default=None, help="Optuna storage URL"
    )

    # System arguments
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of data workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hyperopt_results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Default study name based on model type
    if args.study_name is None:
        args.study_name = f"{args.model_type}_hyperopt"

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config dict for objective
    config = {
        "metadata_path": args.metadata_path,
        "spectra_path": args.spectra_path,
        "splits_path": args.splits_path,
        "max_mass_offset": args.max_mass_offset,
        "min_mz": args.min_mz,
        "max_mz": args.max_mz,
        "bin_width": args.bin_width,
        "n_epochs": args.n_epochs,
        "num_workers": args.num_workers,
        "device": args.device,
    }

    # Add NEIMS-specific fixed params
    if args.model_type == "neims":
        config["fp_radius"] = args.fp_radius
        config["fp_length"] = args.fp_length

    # Select objective function based on model type
    if args.model_type == "neims_gnn":
        objective_fn = objective_neims_gnn
    else:
        objective_fn = objective_neims

    # Build fixed_params for result saving
    fixed_params = {
        "model_type": args.model_type,
        "max_mass_offset": args.max_mass_offset,
        "min_mz": args.min_mz,
        "max_mz": args.max_mz,
        "bin_width": args.bin_width,
        "n_epochs": args.n_epochs,
    }
    if args.model_type == "neims":
        fixed_params["fp_radius"] = args.fp_radius
        fixed_params["fp_length"] = args.fp_length

    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=3
        ),
    )

    # Create callback to log trial completion
    def log_trial_callback(study, trial):
        """Callback to log each trial's completion."""
        print(
            f"\n[Trial {trial.number}] Finished with value: {trial.value:.4f}"
        )
        print(f"  Parameters: {trial.params}")

        # Save intermediate results after each trial
        if trial.number % 5 == 0 or trial.number == args.n_trials - 1:
            results = {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "fixed_params": fixed_params,
            }
            with open(output_dir / "best_params.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save study trials dataframe
            study_df = study.trials_dataframe()
            study_df.to_csv(output_dir / "trials.csv", index=False)
            print(f"  Saved intermediate results to {output_dir}")

    # Run optimization
    print(
        f"Starting {args.model_type} hyperparameter optimization "
        f"with {args.n_trials} trials..."
    )
    study.optimize(
        lambda trial: objective_fn(trial, config),
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[log_trial_callback],
    )

    # Print results
    print("\n" + "=" * 50)
    print(f"OPTIMIZATION RESULTS ({args.model_type.upper()})")
    print("=" * 50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "fixed_params": fixed_params,
    }

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save study
    study_df = study.trials_dataframe()
    study_df.to_csv(output_dir / "trials.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
