#!/bin/bash
# Resilient wrapper around run_inference_for_eval.py's chunked PubChem
# inference for one GPU's assigned row range. The underlying script's
# --chunk-start/--chunk-end loop has no exception handling: any failure on
# any chunk after the first kills the whole process, with nothing to notice
# or restart it. Each already-written chunk is skipped on resume (the
# script's own `if chunk_path.exists(): skip` check), so a plain restart
# loop is safe and just continues from wherever it died.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPU="${GPU:?set GPU=<n>}"
CHUNK_START="${CHUNK_START:?set CHUNK_START=<n>}"
CHUNK_END="${CHUNK_END:?set CHUNK_END=<n>}"
CONFIG="${CONFIG:-config/inference_massformer_scaffold_s1_pubchem_full.yml}"
OUTPUT="${OUTPUT:-results/predictions/massformer_scaffold_s1_pubchem_full.hdf5}"
SMILES_TSV="${SMILES_TSV:-/tmp/PubChem_filtered.tsv}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
LOG="${LOG:-../../results/pubchem_retrieval_pipeline_logs/massformer_scaffold_gpu${GPU}_resilient.log}"

while true; do
    echo "[$(date)] Launching chunk range [$CHUNK_START, $CHUNK_END) on GPU $GPU..." >> "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" /home/magled/miniconda3/envs/MF-GPU/bin/python scripts/run_inference_for_eval.py \
        --config "$CONFIG" \
        --output "$OUTPUT" \
        --smiles-tsv "$SMILES_TSV" \
        --smiles-tsv-smiles-col 1 \
        --smiles-tsv-id-col 0 \
        --chunk-size "$CHUNK_SIZE" \
        --chunk-start "$CHUNK_START" \
        --chunk-end "$CHUNK_END" \
        >> "$LOG" 2>&1
    exit_code=$?
    echo "[$(date)] Exited with code $exit_code." >> "$LOG"

    # Find the highest chunk index already written in this range, resume there.
    last_row=$CHUNK_START
    idx=$((CHUNK_START / CHUNK_SIZE))
    while [ -f "${OUTPUT%.hdf5}.chunk${idx}.hdf5" ]; do
        idx=$((idx + 1))
        last_row=$((idx * CHUNK_SIZE))
    done

    if [ "$last_row" -ge "$CHUNK_END" ]; then
        echo "[$(date)] Range [$CHUNK_START, $CHUNK_END) complete." >> "$LOG"
        break
    fi

    echo "[$(date)] Resuming from row $last_row (was $CHUNK_START). Restarting in 5s..." >> "$LOG"
    CHUNK_START=$last_row
    sleep 5
done
