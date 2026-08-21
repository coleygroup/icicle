"""Predict spectra for a list of SMILES using MassFormer. Outputs JSON to stdout.

Usage (from baselines/massformer/):
    python scripts/predict_smiles_to_json.py \
        --smiles "CCO" "CCC" \
        --checkpoint checkpoints/massformer_random_s2/chkpt.pkl
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

# Redirect stdout to stderr during imports/model loading so only JSON hits stdout
_real_stdout = sys.stdout
sys.stdout = sys.stderr

from massformer.runner import load_config, get_ds_model
from scripts.run_inference import init_from_smiles, run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--template", default="config/template.yml")
    parser.add_argument("--custom", default="config/train_nist23_gcms_massformer_random_s2.yml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    _, _, _, data_d, model_d, run_d = load_config(args.template, args.custom, None, None)
    run_d["device"] = args.device
    run_d["log_tqdm"] = False

    ds, model, _, _, _ = get_ds_model(data_d, model_d, run_d)

    chkpt = th.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(chkpt["best_model_sd"])
    model.eval()

    n_bins = int(data_d["mz_max"])

    # Batch all SMILES in one forward pass
    smiles_df = pd.DataFrame(
        [{"smiles": smi, "mol_id": i} for i, smi in enumerate(args.smiles)]
    )
    new_ds, new_dl = init_from_smiles(ds, smiles_df, prec_types=[], nces=[], run_d=run_d)
    out_d = run_inference(model, new_dl, data_d, model_d, run_d, transform="none", normalization="none")

    results = {}
    for i, smi in enumerate(args.smiles):
        spec = out_d["spec"][i]
        spec = np.maximum(spec, 0).astype(float)
        if spec.max() > 0:
            spec /= spec.max()
        results[smi] = spec[:n_bins].tolist()

    # Restore stdout and print JSON
    sys.stdout = _real_stdout
    print(json.dumps(results))


if __name__ == "__main__":
    main()
