#!/usr/bin/env python
"""Run NEIMS inference and save predictions as HDF5 for unified ICICLE evaluation.

Loads a trained NEIMS (or NEIMS-GNN) checkpoint, runs inference on a given
split or PubChem candidate set, and writes an HDF5 file that
eval_from_predictions.py can consume.
split or PubChem candidate set, and writes an HDF5 file that
eval_from_predictions.py can consume.

Output HDF5 format
------------------
/<mol_id>/
    attrs:
        smiles    (str)
        inchi_key (str)
    datasets:
        predicted_intensities  float32[num_bins]
        mz_bins                float32[num_bins]
file-level attrs:
    min_mz, max_mz, bin_width, model, eval_split

Usage
-----
cd baselines/neims

# Predict test-split molecules (for similarity eval)
python predict.py \\
    --checkpoint outputs/neims_run/best_model.pt \\
    --metadata   ../../data/NIST2023_GCMS_main/metadata.tsv \\
    --spectra    ../../data/NIST2023_GCMS_main/spectra.hdf5 \\
    --splits     ../../data/NIST2023_GCMS_main/splits/scaffold.tsv \\
    --output     outputs/neims_run/predictions_scaffold_test.hdf5 \\
    --eval-split test

# Predict all PubChem isomers (for formula retrieval eval)
python predict.py \\
    --checkpoint outputs/neims_run/best_model.pt \\
    --output     results/predictions/neims_pubchem_cands_None.hdf5 \\
    --candidates-pickle ../../data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_None_compat.pkl

Then from the repo root:
    uv run src/icicle/eval_from_predictions.py \\
        --predictions baselines/neims/outputs/neims_run/predictions_scaffold_test.hdf5 \\
        --ground-truth data/NIST2023_GCMS_main/spectra.hdf5 \\
        --labels  data/NIST2023_GCMS_main/metadata.tsv \\
        --splits  data/NIST2023_GCMS_main/splits/scaffold.tsv \\
        --output  results/eval/neims_scaffold \\
        --mode all
"""

import argparse
import os
import pickle
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import InchiToInchiKey, rdFingerprintGenerator
from rdkit.Chem.inchi import MolToInchi
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def _smiles_to_row(smi: str):
    """Convert a SMILES string to (smiles, inchi_key). Returns (smi, None) on failure."""
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, None
        inchi = MolToInchi(mol)
        if inchi is None:
            return smi, None
        return smi, InchiToInchiKey(inchi)
    except Exception:
        return smi, None


def load_smiles_from_candidates_pickle(candidates_pickle_path: str) -> tuple:
    """Extract all unique candidate SMILES from a retrieval candidates pickle.

     Returns
    -------
     tuple of (mol_ids, smiles_list, inchikey_list)
    """
    with open(candidates_pickle_path, "rb") as f:
        cands_dict = pickle.load(f)

    unique_smiles = set()
    for entry in cands_dict.values():
        cands = entry["cands"]
        unique_smiles.update(
            cands.tolist() if hasattr(cands, "tolist") else cands
        )

    smiles_list = sorted(unique_smiles)
    n_workers = cpu_count()
    print(
        f"Computing InChIKeys for {len(smiles_list)} unique SMILES ({n_workers} workers)…"
    )

    chunk_size = max(1000, len(smiles_list) // (n_workers * 10))
    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_smiles_to_row, smiles_list, chunksize=chunk_size),
                total=len(smiles_list),
                desc="Computing InChIKeys",
            )
        )

    rows = [
        (i, smi, ik) for i, (smi, ik) in enumerate(results) if ik is not None
    ]
    skipped = len(results) - len(rows)
    if skipped:
        print(f"Skipped {skipped} unparseable SMILES")
    print(f"  {len(rows)} candidate molecules ready for inference")

    mol_ids = [str(r[0]) for r in rows]
    smiles_out = [r[1] for r in rows]
    inchikeys = [r[2] for r in rows]
    return mol_ids, smiles_out, inchikeys


def _smiles_to_inchikey(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        inchi = MolToInchi(mol)
        if inchi is None:
            return ""
        return InchiToInchiKey(inchi) or ""
    except Exception:
        return ""


class InferenceDataset(Dataset):
    """Minimal dataset for NEIMS inference (no spectra loading needed)."""

    def __init__(
        self,
        mol_ids: list,
        smiles_list: list,
        fp_radius: int = 2,
        fp_length: int = 4096,
        output_size: int = 750,
    ):
        self.mol_ids = mol_ids
        self.smiles_list = smiles_list
        self.output_size = output_size
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_length
        )

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp = np.zeros(
                self.mfpgen.GetFingerprintAsNumPy(
                    Chem.MolFromSmiles("C")
                ).shape,
                dtype=np.float32,
            )
            mass = np.float32(0.0)
        else:
            fp = self.mfpgen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
            from rdkit.Chem import Descriptors

            mass = np.float32(Descriptors.ExactMolWt(mol))

        return {
            "mol_id": self.mol_ids[idx],
            "fingerprint": torch.from_numpy(fp),
            "mass": torch.tensor(mass),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEIMS inference -> HDF5 (ICICLE-compatible format)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_model.pt saved by train.py",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to ICICLE metadata.tsv (required when not using --candidates-pickle)",
    )
    parser.add_argument(
        "--spectra",
        default=None,
        help="Path to ICICLE spectra.hdf5 (required when not using --candidates-pickle)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Path to ICICLE splits TSV. Required for nist/mona unless --all-test is used. "
            "Not needed for vgwd (all molecules are test) or --candidates-pickle mode."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output .hdf5 file path",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to run inference on (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (default: 256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--candidates-pickle",
        type=str,
        default=None,
        help=(
            "Path to prebuilt candidates pickle "
            "(e.g. data/NIST2023_GCMS_main/retrieval/cands_pickled_scaffold_None_compat.pkl). "
            "When set, runs inference on all unique PubChem candidate SMILES instead of a split."
        ),
    )
    parser.add_argument(
        "--all-test",
        action="store_true",
        default=False,
        help=(
            "Run inference on all molecules in --metadata without filtering by --splits. "
            "Use for datasets where every molecule is a test molecule (MoNA, xeno-AAs)."
        ),
    )
    parser.add_argument(
        "--metadata-format",
        type=str,
        default="nist",
        choices=["nist", "mona", "vgwd"],
        help=(
            "Metadata format to use when loading molecules from --metadata. "
            "'nist' and 'mona' share the same TSV schema (mol_id, standardized_smiles, inchi_key). "
            "'vgwd' uses columns spec, smiles, inchikey from data/VGWD2023_GCMS/raw/labels.tsv."
        ),
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint["model_config"]
    model_type = model_config.get("model_type", "neims")
    print(f"  Model type: {model_type}")

    if model_type == "neims_gnn":
        from neims.gnn_model import NEIMSGNN

        model = NEIMSGNN(
            output_size=model_config["output_size"],
            gnn_type=model_config.get("gnn_type", "GAT"),
            gnn_hidden_size=model_config.get("gnn_hidden_size", 64),
            gnn_num_layers=model_config.get("gnn_num_layers", 10),
            gnn_num_heads=model_config.get("gnn_num_heads", 8),
            gnn_dropout=model_config.get("gnn_dropout", 0.5),
            pool_type=model_config.get("pool_type", "max"),
            use_edge_features=model_config.get("use_edge_features", False),
            ffnn_hidden_sizes=model_config.get("ffnn_hidden_sizes", []),
            ffnn_dropout=model_config.get("ffnn_dropout", 0.25),
            resnet_bottleneck=model_config.get("resnet_bottleneck", 0.5),
            use_glu=model_config.get("use_glu", True),
            bidirectional=model_config.get("bidirectional", True),
            gate_bidirectional=model_config.get("gate_bidirectional", False),
            max_mass_offset=model_config.get("max_mass_offset", 5),
            max_mz=model_config.get("max_mz", 750.0),
        )
    else:
        from neims.model import NEIMS

        model = NEIMS(
            input_size=model_config.get("fp_length", 4096),
            output_size=model_config["output_size"],
            hidden_sizes=model_config.get("hidden_sizes", [2000] * 8),
            dropout=model_config.get("dropout", 0.25),
            bidirectional=model_config.get("bidirectional", True),
            gate_bidirectional=model_config.get("gate_bidirectional", False),
            resnet_bottleneck=model_config.get("resnet_bottleneck", 0.5),
            max_mass_offset=model_config.get("max_mass_offset", 5),
            fp_radius=model_config.get("fp_radius", 2),
            fp_length=model_config.get("fp_length", 4096),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    print(f"  Model loaded on {args.device}")

    min_mz = model_config.get("min_mz", 0.0)
    max_mz = model_config.get("max_mz", 750.0)
    bin_width = model_config.get("bin_width", 1.0)
    output_size = model_config["output_size"]

    if args.candidates_pickle:
        print(
            f"Running inference on PubChem candidates from {args.candidates_pickle}"
        )
        mol_ids, smiles_list, inchikey_list = (
            load_smiles_from_candidates_pickle(args.candidates_pickle)
        )
        eval_split_label = "candidates"
    else:
        metadata_df = pd.read_csv(args.metadata, sep="\t")

        if args.metadata_format == "vgwd":
            # VGWD: spec col = mol_id, smiles col = smiles, inchikey col = inchikey
            metadata_df = metadata_df.rename(
                columns={"spec": "mol_id", "inchikey": "inchi_key"}
            )
            if args.splits:
                splits_df = pd.read_csv(args.splits, sep="\t")
                split_mol_ids = set(
                    splits_df[splits_df["split"] == args.eval_split]["mol_id"].astype(str)
                )
                mol_df = metadata_df[metadata_df["mol_id"].astype(str).isin(split_mol_ids)].copy()
                eval_split_label = args.eval_split
                print(f"  VGWD: {len(mol_df)} molecules in {args.eval_split} split")
            else:
                mol_df = metadata_df.copy()
                eval_split_label = "test"
                print(f"  VGWD: using all {len(mol_df)} molecules (no split)")
            smiles_col = "smiles"
        else:
            # nist / mona: same schema (mol_id, standardized_smiles, inchi_key)
            if args.all_test or args.splits is None:
                # treat everything as test (e.g. MoNA, xeno-AAs)
                mol_df = metadata_df.copy()
                eval_split_label = "test"
                print(f"  {len(mol_df)} molecules (all-as-test, no split file)")
            else:
                print(f"Loading {args.eval_split} split from {args.splits}")
                splits_df = pd.read_csv(args.splits, sep="\t")
                split_mol_ids = set(
                    splits_df[splits_df["split"] == args.eval_split]["mol_id"].astype(str)
                )
                mol_df = metadata_df[
                    metadata_df["mol_id"].astype(str).isin(split_mol_ids)
                ].copy()
                eval_split_label = args.eval_split
                print(f"  {len(mol_df)} molecules in {args.eval_split} split")

            smiles_col = (
                "standardized_smiles"
                if "standardized_smiles" in mol_df.columns
                else "smiles"
            )

        mol_ids = mol_df["mol_id"].astype(str).tolist()
        smiles_list = mol_df[smiles_col].tolist()
        ik_col = "inchi_key" if "inchi_key" in mol_df.columns else "inchikey"
        inchikey_list = (
            mol_df[ik_col].tolist()
            if ik_col in mol_df.columns
            else [_smiles_to_inchikey(s) for s in smiles_list]
        )

    if model_type == "neims_gnn":
        from neims.gnn_data import create_inference_dataloader

        dl = create_inference_dataloader(
            mol_ids=mol_ids,
            smiles_list=smiles_list,
            output_size=output_size,
            use_edge_features=model_config.get("use_edge_features", False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        fp_radius = model_config.get("fp_radius", 2)
        fp_length = model_config.get("fp_length", 4096)
        ds = InferenceDataset(
            mol_ids=mol_ids,
            smiles_list=smiles_list,
            fp_radius=fp_radius,
            fp_length=fp_length,
            output_size=output_size,
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    all_mol_ids = []
    all_preds = []

    print("Running inference …")
    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            if model_type == "neims_gnn":
                batch = batch.to(args.device)
                masses = batch.mass
                preds = model(batch, masses)
                batch_mol_ids = batch.mol_id
            else:
                fps = batch["fingerprint"].to(args.device)
                masses = batch["mass"].to(args.device)
                preds = model(fps, masses)
                batch_mol_ids = batch["mol_id"]

            all_preds.append(preds.cpu().numpy())
            all_mol_ids.extend(
                batch_mol_ids.tolist()
                if hasattr(batch_mol_ids, "tolist")
                else batch_mol_ids
            )

    all_preds = np.concatenate(all_preds, axis=0)  # [N, output_size]

    # Normalise to max=1 (ICICLE convention)
    row_max = all_preds.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    all_preds = all_preds / row_max

    n_bins = output_size
    mz_bins = (
        np.arange(n_bins) * bin_width + min_mz + 0.5 * bin_width
    ).astype(np.float32)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Writing HDF5 -> {out_path}")

    # Build mol_id -> index mapping for fast lookup
    mol_id_to_idx = {mid: i for i, mid in enumerate(mol_ids)}

    with h5py.File(out_path, "w") as hf:
        hf.attrs["min_mz"] = min_mz
        hf.attrs["max_mz"] = max_mz
        hf.attrs["bin_width"] = bin_width
        hf.attrs["model"] = "neims"
        hf.attrs["eval_split"] = eval_split_label
        hf.attrs["n_samples"] = len(mol_ids)

        for i, mol_id in enumerate(tqdm(mol_ids, desc="Writing samples")):
            pred_idx = mol_id_to_idx[mol_id]
            dense = all_preds[pred_idx].astype(np.float32)
            smiles = smiles_list[i]
            inchi_key = inchikey_list[i]

            grp = hf.create_group(mol_id)
            grp.attrs["smiles"] = smiles
            grp.attrs["inchi_key"] = (
                inchi_key
                if inchi_key
                and not (isinstance(inchi_key, float) and np.isnan(inchi_key))
                else ""
            )
            grp.create_dataset(
                "predicted_intensities", data=dense, compression="gzip"
            )
            grp.create_dataset("mz_bins", data=mz_bins, compression="gzip")

    print(f"Done. {len(mol_ids)} predictions saved to {out_path}")


if __name__ == "__main__":
    main()
