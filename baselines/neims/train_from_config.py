"""Train NEIMS from a YAML config file."""

import argparse
import yaml
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train NEIMS from config file")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Build command - use absolute path to train.py
    train_script = Path(__file__).parent / "train.py"
    cmd = [sys.executable, str(train_script)]

    # Add all config parameters as command-line arguments
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            # Convert bool to string for parsing
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        elif isinstance(value, list):
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.extend([str(v) for v in value])
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    # Run training
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
