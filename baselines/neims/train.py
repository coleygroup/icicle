"""Training script for NEIMS and NEIMS-GNN."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from neims.data import create_dataloaders
from neims.model import NEIMS, generalized_mse_loss


def train_epoch(model, train_loader, optimizer, scheduler, device, mass_power, epoch, model_type="neims", loss_type="generalized_mse"):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Extract data based on model type
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

        # Compute loss
        if loss_type == "mse":
            loss = nn.functional.mse_loss(predictions, spectra)
        else:
            loss = generalized_mse_loss(predictions, spectra, mass_power)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Log to wandb
        if batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": global_step
            })

    return total_loss / num_batches


def validate(model, val_loader, device, mass_power, model_type="neims", loss_type="generalized_mse"):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
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

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train NEIMS or NEIMS-GNN model")

    # Model type
    parser.add_argument("--model-type", type=str, default="neims",
                        choices=["neims", "neims_gnn"],
                        help="Model type: neims (fingerprint) or neims_gnn (graph)")

    # Data arguments
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to metadata TSV")
    parser.add_argument("--spectra-path", type=str, required=True, help="Path to spectra HDF5")
    parser.add_argument("--splits-path", type=str, required=True, help="Path to splits TSV")

    # NEIMS model arguments (fingerprint-based)
    parser.add_argument("--fp-radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--fp-length", type=int, default=4096, help="Morgan fingerprint length")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[2000] * 8, help="Hidden layer sizes (1 input + N-1 residual blocks)")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--bidirectional", type=lambda x: str(x).lower() == 'true', default=True, help="Use bidirectional prediction")
    parser.add_argument("--gate-bidirectional", type=lambda x: str(x).lower() == 'true', default=False, help="Gate bidirectional prediction")
    parser.add_argument("--resnet-bottleneck", type=float, default=0.5, help="ResNet bottleneck factor")
    parser.add_argument("--max-mass-offset", type=int, default=5, help="Maximum mass offset")

    # NEIMS-GNN model arguments
    parser.add_argument("--gnn-type", type=str, default="GAT", choices=["GAT", "GCN"],
                        help="GNN type (paper: GAT)")
    parser.add_argument("--gnn-hidden-size", type=int, default=64,
                        help="GNN hidden dimension (paper: 64)")
    parser.add_argument("--gnn-num-layers", type=int, default=10,
                        help="Number of GNN layers (paper: 10)")
    parser.add_argument("--gnn-num-heads", type=int, default=8,
                        help="Number of GAT attention heads (paper: 8)")
    parser.add_argument("--gnn-dropout", type=float, default=0.5,
                        help="GNN dropout rate (paper: 0.5)")
    parser.add_argument("--pool-type", type=str, default="max",
                        choices=["max", "mean", "attention"],
                        help="Global pooling type (paper: max)")
    parser.add_argument("--use-edge-features", type=lambda x: str(x).lower() == 'true',
                        default=False, help="Include bond features in GNN")
    parser.add_argument("--use-glu", type=lambda x: str(x).lower() == 'true',
                        default=True, help="Use GLU output (paper: true)")
    parser.add_argument("--ffnn-hidden-sizes", type=int, nargs="*", default=None,
                        help="FFNN hidden sizes between GNN and output (empty=paper-style)")
    parser.add_argument("--ffnn-dropout", type=float, default=0.25,
                        help="FFNN dropout rate")

    # Spectrum arguments
    parser.add_argument("--min-mz", type=float, default=0.0, help="Minimum m/z")
    parser.add_argument("--max-mz", type=float, default=750.0, help="Maximum m/z")
    parser.add_argument("--bin-width", type=float, default=1.0, help="Bin width")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="L2 regularization / weight decay (paper GNN: 1.0)")
    parser.add_argument("--mass-power", type=float, default=1.0, help="Mass power for loss")
    parser.add_argument("--loss-type", type=str, default="generalized_mse",
                        choices=["generalized_mse", "mse"],
                        help="Loss function: generalized_mse (NEIMS) or mse (paper)")
    parser.add_argument("--decay-scale", type=float, default=1000.0, help="LR decay scale")
    parser.add_argument("--min-lr-multiplier", type=float, default=0.05, help="Minimum LR multiplier")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default="neims", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"], help="Wandb mode")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize wandb
    run_name = args.wandb_run_name or Path(args.output_dir).name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        mode=args.wandb_mode,
        name=run_name,
        dir=str(output_dir)
    )

    # Create dataloaders and model based on model type
    print(f"Loading data (model_type={args.model_type})...")

    if args.model_type == "neims_gnn":
        from neims.gnn_data import create_gnn_dataloaders
        from neims.gnn_model import NEIMSGNN

        train_loader, val_loader, test_loader, output_size = create_gnn_dataloaders(
            metadata_path=args.metadata_path,
            spectra_path=args.spectra_path,
            splits_path=args.splits_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            min_mz=args.min_mz,
            max_mz=args.max_mz,
            bin_width=args.bin_width,
            use_edge_features=args.use_edge_features,
        )

        print("Creating NEIMS-GNN model...")
        model = NEIMSGNN(
            output_size=output_size,
            gnn_type=args.gnn_type,
            gnn_hidden_size=args.gnn_hidden_size,
            gnn_num_layers=args.gnn_num_layers,
            gnn_num_heads=args.gnn_num_heads,
            gnn_dropout=args.gnn_dropout,
            pool_type=args.pool_type,
            use_edge_features=args.use_edge_features,
            ffnn_hidden_sizes=args.ffnn_hidden_sizes or [],
            ffnn_dropout=args.ffnn_dropout,
            resnet_bottleneck=args.resnet_bottleneck,
            use_glu=args.use_glu,
            bidirectional=args.bidirectional,
            gate_bidirectional=args.gate_bidirectional,
            max_mass_offset=args.max_mass_offset,
            max_mz=args.max_mz,
        ).to(args.device)

        model_config = {
            "model_type": "neims_gnn",
            "output_size": output_size,
            "gnn_type": args.gnn_type,
            "gnn_hidden_size": args.gnn_hidden_size,
            "gnn_num_layers": args.gnn_num_layers,
            "gnn_num_heads": args.gnn_num_heads,
            "gnn_dropout": args.gnn_dropout,
            "pool_type": args.pool_type,
            "use_edge_features": args.use_edge_features,
            "ffnn_hidden_sizes": args.ffnn_hidden_sizes or [],
            "ffnn_dropout": args.ffnn_dropout,
            "resnet_bottleneck": args.resnet_bottleneck,
            "use_glu": args.use_glu,
            "bidirectional": args.bidirectional,
            "gate_bidirectional": args.gate_bidirectional,
            "max_mass_offset": args.max_mass_offset,
            "min_mz": args.min_mz,
            "max_mz": args.max_mz,
            "bin_width": args.bin_width,
        }
    else:
        train_loader, val_loader, test_loader, output_size = create_dataloaders(
            metadata_path=args.metadata_path,
            spectra_path=args.spectra_path,
            splits_path=args.splits_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fp_radius=args.fp_radius,
            fp_length=args.fp_length,
            min_mz=args.min_mz,
            max_mz=args.max_mz,
            bin_width=args.bin_width,
        )

        print("Creating NEIMS model...")
        model = NEIMS(
            input_size=args.fp_length,
            output_size=output_size,
            hidden_sizes=args.hidden_sizes,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            gate_bidirectional=args.gate_bidirectional,
            resnet_bottleneck=args.resnet_bottleneck,
            max_mass_offset=args.max_mass_offset,
            fp_radius=args.fp_radius,
            fp_length=args.fp_length,
        ).to(args.device)

        model_config = {
            "model_type": "neims",
            "fp_length": args.fp_length,
            "fp_radius": args.fp_radius,
            "output_size": output_size,
            "hidden_sizes": args.hidden_sizes,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "gate_bidirectional": args.gate_bidirectional,
            "resnet_bottleneck": args.resnet_bottleneck,
            "max_mass_offset": args.max_mass_offset,
            "min_mz": args.min_mz,
            "max_mz": args.max_mz,
            "bin_width": args.bin_width,
        }

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1e-7,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step):
        return max(
            1.0 / math.sqrt(1.0 + step / args.decay_scale),
            args.min_lr_multiplier,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, args.device,
            args.mass_power, epoch,
            model_type=args.model_type, loss_type=args.loss_type,
        )

        # Validate
        val_loss = validate(
            model, val_loader, args.device, args.mass_power,
            model_type=args.model_type, loss_type=args.loss_type,
        )

        # Log
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss
        })

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                    "model_config": model_config,
                },
                output_dir / "best_model.pt",
            )
            print(f"Saved best model with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    # Test
    print("Testing...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = validate(
        model, test_loader, args.device, args.mass_power,
        model_type=args.model_type, loss_type=args.loss_type,
    )
    print(f"Test loss: {test_loss:.4f}")

    wandb.log({
        "test/loss": test_loss,
        "best_val_loss": best_val_loss
    })

    # Save final metrics
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
            },
            f,
            indent=2,
        )

    wandb.finish()
    print(f"Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
