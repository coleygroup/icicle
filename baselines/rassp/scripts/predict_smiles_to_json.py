#!/usr/bin/env python
"""Run RASSP inference on arbitrary SMILES, print {smiles: spectrum} as JSON.

Mirrors baselines/massformer/scripts/predict_smiles_to_json.py so both
baselines can be driven from a notebook via the same subprocess pattern
(conda env with its own Python, stdout captured and json.loads'd).

Usage (from repo root, in the `rassp` conda env):
    conda run -n rassp python baselines/rassp/scripts/predict_smiles_to_json.py \\
        --smiles "CCO" "CCC" \\
        --checkpoint baselines/rassp/rassp/checkpoints/rassp_random_s1.rassp_random_s1.00000050.model \\
        --meta baselines/rassp/rassp/checkpoints/rassp_random_s1.rassp_random_s1.meta \\
        --device cuda:0
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rassp import netutil
from rassp.run_rassp import RenameUnpickler

sys.path.insert(0, os.path.dirname(__file__))
from run_inference_for_eval import _sparse_to_dense, validate_smiles_parallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-mz", type=float, default=0.0)
    parser.add_argument("--max-mz", type=float, default=750.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    real_stdout = sys.stdout
    sys.stdout = io.StringIO()  # rassp/run_inference_for_eval print() noise -> discard, keep stdout JSON-only

    with open(args.meta, "rb") as f:
        meta = RenameUnpickler(f).load()
    feat_config = meta["featurize_config"]

    use_gpu = args.device.startswith("cuda") and torch.cuda.is_available()
    predictor = netutil.PredModel(
        args.meta,
        args.checkpoint,
        USE_CUDA=use_gpu,
        data_parallel=False,
        featurize_config_update=feat_config,
    )

    valid_mols, validity = validate_smiles_parallel(
        args.smiles,
        feat_config["MAX_N"],
        feat_config["explicit_formulae_config"]["max_formulae"],
        n_workers=1,
    )
    valid_mol_list = [m for m, ok in zip(valid_mols, validity) if ok]

    preds = predictor.pred(
        valid_mol_list,
        progress_bar=False,
        normalize_pred=True,
        output_hist_bins=True,
        batch_size=args.batch_size,
        dataloader_config={
            "pin_memory": False,
            "num_workers": 0,
            "persistent_workers": False,
        },
    )
    pred_binned = preds["pred_binned"]
    n_bins = int((args.max_mz - args.min_mz) / args.bin_width)

    out = {}
    pred_idx = 0
    for smi, ok in zip(args.smiles, validity):
        if not ok:
            out[smi] = np.zeros(n_bins).tolist()
            continue
        dense = _sparse_to_dense(
            pred_binned[pred_idx], n_bins, args.bin_width, args.min_mz
        )
        pred_idx += 1
        if dense.max() > 0:
            dense = dense / dense.max()
        out[smi] = dense.tolist()

    sys.stdout = real_stdout
    print(json.dumps(out))


if __name__ == "__main__":
    main()
