#!/bin/bash
set -euo pipefail

DATA_DIR=data
DATA_SET_DIR=data/NIST2023_GCMS_main

# Download PubChem SMILES
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz

# Unzip
gunzip CID-SMILES.gz

mkdir -p "$DATA_DIR/PubChem"
mkdir -p "$DATA_SET_DIR/retrieval"

mv CID-SMILES "$DATA_DIR/PubChem/pubchem_full.txt"

# Filter PubChem (MW, atom types, valid SMILES, no stereochemistry, dedup by
# InChIKey-14). filter_pubchem.py has no CLI - it reads INPUT_FILE/OUTPUT_DIR
# constants at the top of the file. Edit those to point at
# $DATA_DIR/PubChem/pubchem_full.txt / $DATA_DIR/PubChem/ before running:
uv run examples/scripts/retention_index/filter_pubchem.py

# FORMULA MATCH CANDIDATE SET
# Create formula subsets
uv run examples/scripts/data_processing/retrieval_candidate_set/retrieval_build_general_map.py \
    --pubchem-file "$DATA_DIR/PubChem/PubChem_filtered.tsv" \
    --output-file "$DATA_DIR/PubChem/pubchem_formula_map.p" \
    --n-jobs 32

# Subset dataset
uv run examples/scripts/data_processing/retrieval_candidate_set/retrieval_create_subset_map.py \
    --full-map "$DATA_DIR/PubChem/pubchem_formula_map.p" \
    --labels-file "$DATA_SET_DIR/metadata.tsv" \
    --output-file "$DATA_SET_DIR/retrieval/pubchem_formula_map_subset.p"

# Make retrieval lists (both splits used in the paper)
uv run examples/scripts/data_processing/retrieval_candidate_set/retrieval_create_retrieval_lists.py \
    --input-map "$DATA_SET_DIR/retrieval/pubchem_formula_map_subset.p" \
    --labels-file "$DATA_SET_DIR/metadata.tsv" \
    --split-file "$DATA_SET_DIR/splits/random_no_xeno_aas_deduplicated.tsv" \
    --output-dir "$DATA_SET_DIR/retrieval/" \
    --max-k 50

uv run examples/scripts/data_processing/retrieval_candidate_set/retrieval_create_retrieval_lists.py \
    --input-map "$DATA_SET_DIR/retrieval/pubchem_formula_map_subset.p" \
    --labels-file "$DATA_SET_DIR/metadata.tsv" \
    --split-file "$DATA_SET_DIR/splits/scaffold_no_xeno_aas_deduplicated.tsv" \
    --output-dir "$DATA_SET_DIR/retrieval/" \
    --max-k 50

# RI MATCH CANDIDATE SET
# Train the AIRI retention-index model first (per RI column type):
#   uv run examples/scripts/retention_index/train_airi_models.py \
#       --data-dir $DATA_SET_DIR/airi_data_stdnp_random/ \
#       --output-dir airi_models/stdnp
# See README.md's "Top-N RI Retrieval" section for the full command and the
# other two column types.

# Inference on PubChem (all molecules), one model checkpoint per RI type:
uv run examples/scripts/retention_index/infer_airi.py \
    --input "$DATA_DIR/PubChem/PubChem_filtered.tsv" \
    --output "$DATA_DIR/PubChem/PubChem_filtered_with_ri_new.tsv" \
    --smiles-column SMILES \
    --model-path airi_models/stdnp/<checkpoint>.ckpt \
    --ri-type StdNP

# Build the RI-window candidate set (random split - the only RI dataset
# currently prepared under $DATA_SET_DIR/retention_index_airi/; a scaffold-
# split RI dataset would need to be prepared separately before this can be
# run for the scaffold split):
uv run examples/scripts/retention_index/create_retrieval_candidates.py \
    --ri-predictions-file "$DATA_DIR/PubChem/PubChem_filtered_with_ri_new.tsv" \
    --ri-dataset-file "$DATA_SET_DIR/retention_index_airi/ri_dataset_random_split_no_xeno_aas.tsv" \
    --output-dir "$DATA_SET_DIR/retrieval/" \
    --output-prefix pubchem_ri_candidates_top_n \
    --mode top-n \
    --top-n 150
