"""Checkpoint utilities."""

from pathlib import Path


def load_ckpt(ckpt_dir: Path):
    """Load checkpoint from directory."""
    ckpt_dir = ckpt_dir / "checkpoints"  # This line creates a new path
    ckpt = list(Path(ckpt_dir).glob("*.ckpt"))[0]

    return ckpt
