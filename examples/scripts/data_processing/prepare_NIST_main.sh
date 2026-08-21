#!/bin/bash

SDF_PATH="/home/runzhong/ms_collaborators/nist2023_gcms/gcms_nist23.SDF"
DATA_DIR="/home/magled/icicle-dev/data/NIST2023_GCMS_main"

# Create directory structure
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/splits

# Reformat SDF to labels file & extracted spectra
uv run examples/scripts/data_processing/extract_spectra_from_sdf.py --sdf-path $SDF_PATH --output-dir $DATA_DIR

# Create splits
uv run examples/scripts/data_processing/create_splits.py --metadata-path $DATA_DIR/metadata.tsv --output-dir $DATA_DIR/splits --split-types random
uv run examples/scripts/data_processing/create_splits.py --metadata-path $DATA_DIR/metadata.tsv --output-dir $DATA_DIR/splits --split-types scaffold
uv run examples/scripts/data_processing/create_splits.py --metadata-path $DATA_DIR/metadata.tsv --output-dir $DATA_DIR/splits --split-types butina

# Remove xeno AAs?
# TODO

# Deduplicate stereoisomers
uv run examples/scripts/data_processing/deduplicate_stereoisomers.py \
   --split-file $DATA_DIR/splits/random_no_xeno_aas_deduplicated.tsv \
   --metadata-file $DATA_DIR/metadata.tsv

uv run examples/scripts/data_processing/deduplicate_stereoisomers.py \
   --split-file $DATA_DIR/splits/scaffold_no_xeno_aas_deduplicated.tsv \
   --metadata-file $DATA_DIR/metadata.tsv

# Run MAGMa and assign subformulae to peaks
uv run examples/scripts/data_processing/label_ground_truth_dags.py --data-dir $DATA_DIR --num-h-shifts 1
uv run examples/scripts/data_processing/label_ground_truth_dags.py --data-dir $DATA_DIR --num-h-shifts 6
uv run examples/scripts/data_processing/label_ground_truth_dags.py --data-dir $DATA_DIR --num-h-shifts 1 --detect-isotope-patterns

# [OPTIONAL]Get NPClassifier classifications
mkdir -p $DATA_DIR/np_classes
uv run examples/scripts/data_processing/get_np_class.py --labels-file $DATA_DIR/metadata.tsv --splits-file $DATA_DIR/splits/random_no_xeno_aas_deduplicated.tsv --output-dir $DATA_DIR/np_classes/

# [OPTIONAL] Get exmol classifications