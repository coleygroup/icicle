#!/bin/bash
# Self-healing wrapper around batch_infer_pubchem_neims.py for the scaffold-split
# PubChem run. The underlying script has a recurring HDF5 corruption bug: a mid-flush
# crash can leave the `intensities` dataset resized ahead of `smiles`/`inchikey14`/
# `valid` (write order in _flush is intensities first), and once that happens the
# file's internal free-space manager is broken -- even a plain resize() to truncate
# it back fails with the same "addr overflow" OSError. Salvaging the "good" prefix
# isn't safe either: a partial-repair attempt separately hit the same error reading
# rows well before the crash boundary. So the only reliable recovery is a full
# restart from row 0 whenever corruption is detected -- cheap here since NEIMS runs
# in well under 2 hours end-to-end at steady state.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CKPT="${CKPT:-outputs/neims_scaffold_s1/best_model.pt}"
INPUT="${INPUT:-/tmp/PubChem_filtered.tsv}"
OUTPUT="${OUTPUT:-results/pubchem_predictions/neims_scaffold_s1_pubchem_full.hdf5}"
PROGRESS="${OUTPUT}.progress.json"

check_consistent() {
    uv run --no-sync python -c "
import sys, h5py
try:
    with h5py.File('$OUTPUT', 'r') as f:
        lens = {k: f[k].shape[0] for k in ('smiles', 'inchikey14', 'valid', 'intensities')}
        n = f['smiles'].shape[0]
        _ = f['smiles'][n-1]; _ = f['inchikey14'][n-1]; _ = f['intensities'][n-1]
except Exception as e:
    print(f'BAD: {e}')
    sys.exit(1)
if len(set(lens.values())) != 1:
    print(f'BAD: length mismatch {lens}')
    sys.exit(1)
print(f'OK: {n} consistent rows')
" 2>&1
}

while true; do
    if [ -f "$OUTPUT" ]; then
        result=$(check_consistent)
        echo "[$(date)] Consistency check: $result"
        if [[ "$result" != OK:* ]]; then
            echo "[$(date)] Corruption detected -- deleting output and progress file, restarting from row 0."
            rm -f "$OUTPUT" "$PROGRESS"
        fi
    fi

    echo "[$(date)] Launching batch_infer_pubchem_neims.py..."
    CKPT="$CKPT" INPUT="$INPUT" OUTPUT="$OUTPUT" bash batch_infer_pubchem_neims.sh
    exit_code=$?
    echo "[$(date)] Exited with code $exit_code."

    result=$(check_consistent)
    if [[ "$result" == OK:* ]]; then
        total_rows=$(echo "$result" | grep -oE '[0-9]+' | head -1)
        if [ "$total_rows" -ge 93661074 ]; then
            echo "[$(date)] All rows processed. Done."
            break
        fi
    fi

    echo "[$(date)] Restarting in 5s..."
    sleep 5
done
