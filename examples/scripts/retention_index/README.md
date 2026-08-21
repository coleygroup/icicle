# Retention Index (RI) Scripts

Scripts for training RI prediction models, running large-scale inference on PubChem, and evaluating RI-based candidate selection for GC-MS retrieval.

Three RI column types are supported throughout: **StdNP** (standard non-polar), **SemiStdNP** (semi-standard non-polar), **StdPolar** (standard polar).

---

## Data Pipeline: PubChem Raw --> AIRI Reference Database

This section documents the full lineage of files on disk, including how many molecules
are lost at each step.

### Download raw PubChem SMILES

```
data/PubChem/pubchem_full.txt          119,314,794 lines
```

Downloaded from PubChem (CID --> Isomeric SMILES). One SMILES per line, no header,
no InChIKeys. Raw, unfiltered.

---

### Filter to AIRI-compatible molecules

**Script:** `examples/scripts/retention_index/filter_pubchem.py` (logged to `data/PubChem/filter_pubchem.log`)

```
Input:  pubchem_full.txt             119,314,794

  Invalid SMILES:                         41,048   (0.03%)  — RDKit can't parse
  Invalid elements:                    1,355,951   (1.14%)  — contains atoms outside
                                                              VALID_ELEMENTS (src/icicle/utils/
                                                              chem/constants.py):
                                                              {C,N,P,O,S,Si,I,H,Cl,F,Br,B,Se,
                                                               Fe,Co,As,Na,K}
                                                              i.e. heavy metals (Pb,Hg,Pt,Sn,…),
                                                              rare transition metals, radioactive
                                                              isotopes, etc.
                                                              Si IS valid and NOT filtered here.
                                                              AIRI (masskit_ai mol_features.py)
                                                              also supports Si natively.
  MW > 750 Da:                         6,828,640   (5.72%)  — out of AIRI training domain
  RDKit errors:                               36   (~0%)    — sanitization failures
  ─────────────────────────────────────────────
  Subtotal removed before dedup:       8,225,675   (6.89%)

  Duplicates (same 2D structure):     17,428,045  (15.71%)  — removed by InChIKey dedup

  Total removed:                      25,653,720  (21.50%)

Output: PubChem_filtered.tsv              93,661,074  (78.50%)
        columns: ID, SMILES, InChIKey, MW
```

**Also created:** `PubChem_filtered_for_AIRI.csv` — same 93.7M molecules, SMILES-only
(no header), ready to feed into AIRI inference.

---

## AIRI Training (from scratch)

To train the AIRI models:

```bash
bash examples/scripts/retention_index/train_airi_batch.sh
```
---

### AIRI inference (batched across multiple GPUs)

**Script:** `predict_ri_airi.py` | **Orchestrated by:** `inference_airi_pubchem.sh`

AIRI inference is GPU-intensive and takes hours for 93M molecules, so the input is split into row-range chunks run in parallel on different machines.

**Environment:** requires `masskit_ai` conda environment.

```bash
conda activate masskit_ai
```

### Add 2D InChIKeys and molecular descriptors

**Script:** `add_inchikeys_descriptors_pubchem.py`

Adds five columns computed from SMILES using RDKit (parallel, 16 workers):

| Column             | Description |
|-------||
| `InChIKey`         | **2D InChIKey** (stereochemistry stripped via `RemoveStereochemistry`) |
| `mw`               | Exact monoisotopic mass |
| `molecular_formula`| e.g. `C10H18O` |
| `dbe`              | Degree of unsaturation: `1 + (2C + N − H − X) / 2` |
| `is_aromatic`      | Bool, has ≥ 1 aromatic atom |

```bash
uv run examples/scripts/retention_index/add_inchikeys_descriptors_pubchem.py \
    --input  data/PubChem/260303_full_PubChem_AIRI_with_RI.tsv \
    --output data/PubChem/260303_full_PubChem_AIRI_with_RI_inchikey.tsv \
    --workers 16
```

```
Output: 260303_full_PubChem_AIRI_with_RI_inchikey.tsv     95,313,806 rows
        ri_StdNP non-null:     95,232,139  (99.9%)
        ri_SemiStdNP non-null: 95,232,139  (99.9%)
        ri_StdPolar non-null:  95,232,139  (99.9%)
```


**Parquet cache:** `analyze_ri_recall.py` automatically caches this file to
`260303_full_PubChem_AIRI_with_RI_inchikey_cache.parquet` on first load for faster
subsequent access.


## Candidate Set Creation

After producing a RI-annotated reference database, create
candidate sets for retrieval evaluation:

```bash
# Realistic scenario: candidates ranked by predicted RI of the query
uv run examples/scripts/retention_index/create_retrieval_candidates.py \
    --ri-predictions-file data/PubChem/260303_full_PubChem_AIRI_with_RI_inchikey.tsv \
    --ri-dataset-file     data/NIST2023_GCMS_main/retention_index/ri_dataset_random_split_no_xeno_aas.tsv \
    --output-dir          data/NIST2023_GCMS_main/retrieval/ \
    --output-prefix       pubchem_airi_candidates \
    --ri-source from_pred \ # from_exp: candidates ranked by experimental RI of the query (less realistic, oracle upper bound)
    --mode range --ri-margin 100
```

`--mode top-n` returns the N closest molecules; `--mode range` returns all molecules
within `--ri-margin` RI units.

---

## Recall Analysis

`analyze_ri_recall.py` evaluates oracle upper bounds for RI-based candidate selection
across 10 scenarios (RI only, RI + formula, RI + mass window, etc.) against the full
95M PubChem reference.

Run as a script or open in VS Code / Jupyter (has `# %%` cell markers):

```bash
uv run examples/scripts/evaluation/analyze_ri_recall.py
```

Outputs per scenario (saved after each scenario completes):
- `examples/notebooks/ri_recall_<scenario>.svg` — recall vs top-N curve
- `examples/notebooks/ri_ranks_hist_<scenario>.svg` — log₁₀(rank) distribution
- `examples/notebooks/ri_recall_<scenario>.csv` — numerical recall values
- `examples/notebooks/ri_recall_all_scenarios.csv` — all scenarios combined
- `examples/notebooks/ri_recall_all_scenarios_<RI_TYPE>.svg` — combined grid plot


---

## File Inventory

```
data/PubChem/
  pubchem_full.txt                          raw download, 119.3M SMILES
  PubChem_filtered.tsv                      93.7M rows: ID, SMILES, InChIKey, MW
  PubChem_filtered_for_AIRI.csv             93.7M SMILES only (AIRI input format)
  PubChem_filtered_for_AIRI_without_260302.csv   molecules not yet predicted in round 1
  260302_*.tsv                              round-1 AIRI chunk predictions
  260303_*.tsv                              round-2 AIRI chunk predictions
  260303_full_PubChem_AIRI_with_RI.tsv      merged predictions, 95.3M rows, no InChIKey
  260303_full_PubChem_AIRI_with_RI_inchikey.tsv  <- MAIN REFERENCE FILE
                                            95.3M rows, +InChIKey/MW/formula/dbe/aromatic
  260303_full_PubChem_AIRI_with_RI_inchikey_cache.parquet  auto-cached by analyze script
```
