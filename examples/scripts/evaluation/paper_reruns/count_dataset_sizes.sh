#!/usr/bin/env bash
# Produces: SI Table tab:split-sizes (every dataset/split count) -- provenance
# only, no side effects (stdout only, does not write any file).
#
# Reports the exact dataset/split sizes used in the SI, with the file each
# number comes from, so every count in supplementary.tex is traceable.
# Run: bash examples/scripts/evaluation/paper_reruns/count_dataset_sizes.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

DATA=data/NIST2023_GCMS_main

echo "=== Full extracted+filtered NIST metadata (output of extract_spectra_from_sdf.py) ==="
echo "file: $DATA/metadata.tsv"
n=$(($(wc -l < "$DATA/metadata.tsv") - 1))
echo "rows (excl. header): $n"

echo
echo "=== Split sizes, before dedup, xeno-AAs already excluded (output of create_splits.py + xeno_amino_acids/filter_splits.py) ==="
for f in random_no_xeno_aas.tsv scaffold_no_xeno_aas.tsv; do
  n=$(($(wc -l < "$DATA/splits/$f") - 1))
  echo "$f: $n"
done

echo
echo "=== Split sizes, after InChIKey-14 dedup (deduplicated splits actually used for training/eval) ==="
for f in random_no_xeno_aas_deduplicated.tsv scaffold_no_xeno_aas_deduplicated.tsv; do
  n=$(($(wc -l < "$DATA/splits/$f") - 1))
  echo "$f: $n"
done

echo
echo "=== Per-split train/val/test breakdown ==="
for f in random_no_xeno_aas_deduplicated.tsv scaffold_no_xeno_aas_deduplicated.tsv; do
  echo "-- $f --"
  tail -n +2 "$DATA/splits/$f" | cut -f3 | sort | uniq -c
done

echo
echo "=== RASSP-filtered subset sizes, per split, train/val/test ==="
for f in random_no_xeno_aas_deduplicated_no_qcxms2_rassp.tsv scaffold_no_xeno_aas_deduplicated_rassp.tsv; do
  echo "-- $f --"
  tail -n +2 "$DATA/splits/$f" | cut -f3 | sort | uniq -c
done

echo
echo "=== AIRI RI-predictor train/val/test sizes, per column type (random split) ==="
for t in stdnp stdpolar semistdnp; do
  d="$DATA/airi_data_${t}_random"
  echo "-- $t --"
  for split in train valid test; do
    f="$d/airi_${split}.parquet"
    if [ -f "$f" ]; then
      python3 -c "import pandas as pd; print('$split:', len(pd.read_parquet('$f')))"
    fi
  done
done
