"""Train a simple heavy-atom-count predictor from a binned EI-MS spectrum.

Motivation: the existing MW filter in pubchem_global_retrieval.py uses the
highest observed peak as a molecular-weight proxy, which is a noisy signal
(the molecular ion is often weak/absent; isotope peaks confuse it further).
Heavy-atom count is a different, complementary structural descriptor (it
distinguishes a small dense molecule from a large sparse one at the same
MW) and is directly derivable from a predicted spectrum without needing
ground-truth structure — exactly the same setting as the MW proxy, so it
can be used as a candidate-set filter the same way (see
add_true_heavy_atom_index_to_hdf5.py + evaluate_heavy_atom_window in
pubchem_global_retrieval.py).

Trained on NIST2023_GCMS_main's true spectra (ground truth available).
Split defaults to random (random_no_xeno_aas_deduplicated); pass
--split-path/--output-model-path to train a scaffold-split counterpart
instead — required when evaluating heavy-atom filtering against
scaffold-split PubChem predictions, since applying the random-split
model there leaks scaffold information the split is meant to withhold
(the model's train set may share scaffolds with the scaffold-split test
set). At inference on PubChem candidates, the model is applied to each
candidate's own *predicted* spectrum (from ICICLE/NEIMS/MassFormer)
instead of a true one — same setup as MW.

Usage
-----
uv run examples/scripts/evaluation/train_heavy_atom_predictor.py
uv run examples/scripts/evaluation/train_heavy_atom_predictor.py \
    --split-path data/NIST2023_GCMS_main/splits/scaffold_no_xeno_aas_deduplicated.tsv \
    --output-model-path checkpoints/heavy_atom_predictor_scaffold_split.joblib
"""

import argparse
import logging
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data/NIST2023_GCMS_main")
METADATA_PATH = DATA_DIR / "metadata.tsv"
SPECTRA_PATH = DATA_DIR / "spectra.hdf5"

MIN_MZ = 0
MAX_MZ = 750
BIN_WIDTH = 1.0
N_BINS = int((MAX_MZ - MIN_MZ) / BIN_WIDTH)


def _bin_spectrum(masses: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """Same binning as pubchem_global_retrieval.py's _bin_spectrum, so a model
    trained here applies directly to predicted spectra scanned from the PubChem
    prediction HDF5s (same bin edges, same max-normalization)."""
    spec = np.zeros(N_BINS, dtype=np.float32)
    bins = np.floor((masses - MIN_MZ) / BIN_WIDTH).astype(int)
    valid = (bins >= 0) & (bins < N_BINS)
    np.add.at(spec, bins[valid], intensities[valid])
    if spec.max() > 0:
        spec /= spec.max()
    return spec


def load_split_data(
    split_name: str,
    split_df: pd.DataFrame,
    metadata: pd.DataFrame,
    spectra_file: h5py.File,
):
    mol_ids = split_df.loc[split_df["split"] == split_name, "mol_id"]
    meta_sub = metadata[metadata["mol_id"].isin(mol_ids)]

    X, y = [], []
    for mol_id, num_atoms in zip(meta_sub["mol_id"], meta_sub["num_atoms"]):
        key = str(mol_id)
        if key not in spectra_file:
            continue
        grp = spectra_file[key]
        masses = np.array(grp["masses"])
        intensities = np.array(grp["intensities"])
        spec = _bin_spectrum(masses, intensities)
        if spec.max() <= 0:
            continue
        X.append(spec)
        y.append(num_atoms)

    return np.stack(X), np.array(y, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split-path",
        type=Path,
        default=DATA_DIR / "splits/random_no_xeno_aas_deduplicated.tsv",
    )
    p.add_argument(
        "--output-model-path",
        type=Path,
        default=Path("checkpoints/heavy_atom_predictor_random_split.joblib"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info(f"Loading metadata from {METADATA_PATH}")
    metadata = pd.read_csv(
        METADATA_PATH,
        sep="\t",
        usecols=["mol_id", "num_atoms", "has_spectrum"],
    )
    metadata = metadata[metadata["has_spectrum"]]

    log.info(f"Loading split from {args.split_path}")
    split_df = pd.read_csv(args.split_path, sep="\t")

    log.info(f"Loading spectra from {SPECTRA_PATH}")
    with h5py.File(SPECTRA_PATH, "r") as spectra_file:
        log.info("Building train set...")
        X_train, y_train = load_split_data(
            "train", split_df, metadata, spectra_file
        )
        log.info("Building val set...")
        X_val, y_val = load_split_data("val", split_df, metadata, spectra_file)
        log.info("Building test set...")
        X_test, y_test = load_split_data(
            "test", split_df, metadata, spectra_file
        )

    log.info(
        f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=0,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    log.info("Fitting GradientBoostingRegressor...")
    model.fit(X_train, y_train)

    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        within_1 = float(np.mean(np.abs(pred - y) <= 1.0))
        within_2 = float(np.mean(np.abs(pred - y) <= 2.0))
        log.info(
            f"[{name}] MAE={mae:.3f}  R2={r2:.4f}  "
            f"within_1_atom={within_1:.3f}  within_2_atoms={within_2:.3f}"
        )

    args.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model_path)
    log.info(f"Saved model → {args.output_model_path}")


if __name__ == "__main__":
    main()
