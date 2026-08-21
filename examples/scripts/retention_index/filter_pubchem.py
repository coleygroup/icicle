"""Script to filter PubChem dataset based on criteria:
- Valid SMILES
- Molecular weight <= 750 Da
- Contains only valid elements
- Remove stereochemistry
- Deduplicate by InChIKey connectivity (first 14-char block, keeps first occurrence)
Saves filtered dataset and logs RDKit errors."""

import logging
import multiprocessing as mp
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from icicle.utils.chem.constants import VALID_ELEMENTS

# Configuration
INPUT_FILE = "data/PubChem/pubchem_full.txt"
OUTPUT_DIR = Path("data/PubChem")
OUTPUT_FILE = OUTPUT_DIR / "PubChem_filtered.tsv"
LOG_FILE = OUTPUT_DIR / "filter_pubchem.log"
RDKIT_ERRORS_FILE = OUTPUT_DIR / "rdkit_errors.tsv"
INVALID_SMILES_FILE = OUTPUT_DIR / "invalid_smiles.tsv"
CHUNK_SIZE = 10000  # Lines per chunk for parallel processing


def setup_logging(log_path):
    """Setup logging to file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    return logging.getLogger(__name__)


def has_valid_elements(mol):
    """Checks if all atoms in the mol are in the VALID_ELEMENTS set."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in VALID_ELEMENTS:
            return False
    return True


def process_line(line):
    """Process a single line.

    Returns (inchikey, result_str, status, error_line, invalid_smiles_line).
    inchikey is used for deduplication in the main process.
    """
    line = line.strip()

    if not line:
        return None, None, "empty_line", None, None

    parts = line.split()
    if len(parts) < 2:
        return None, None, "malformed", None, None

    cid = parts[0]
    smiles = parts[1]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_line = f"{cid}\t{smiles}\n"
            return None, None, "invalid_smiles", None, invalid_line

        if not has_valid_elements(mol):
            return None, None, "invalid_elements", None, None

        mw = Descriptors.MolWt(mol)
        if mw > 750:
            return None, None, "mw_too_high", None, None

        Chem.RemoveStereochemistry(mol)
        clean_smiles = Chem.MolToSmiles(mol)
        inchikey = Chem.MolToInchiKey(mol)
        # First block of InChIKey (14 chars) = connectivity layer (same for stereoisomers)
        inchikey_connectivity = inchikey.split("-")[0]

        result = f"{cid}\t{clean_smiles}\t{inchikey}\t{mw:.4f}\n"
        return inchikey_connectivity, result, "passed_filters", None, None

    except Exception as e:
        # Return original line info for logging
        error_line = f"{cid}\t{smiles}\t{type(e).__name__}: {e}\n"
        return None, None, "rdkit_error", error_line, None


def process_chunk(lines):
    """Process a chunk of lines.

    Returns (results_with_keys, error_lines, invalid_smiles_lines,
    chunk_stats). results_with_keys is a list of (inchikey_connectivity,
    result_str) tuples.
    """
    chunk_stats = {
        "total": 0,
        "empty_line": 0,
        "malformed": 0,
        "invalid_smiles": 0,
        "invalid_elements": 0,
        "mw_too_high": 0,
        "rdkit_error": 0,
        "passed_filters": 0,
    }
    results = []
    error_lines = []
    invalid_smiles_lines = []

    for line in lines:
        chunk_stats["total"] += 1
        inchikey_conn, result, status, error_line, invalid_line = process_line(
            line
        )
        chunk_stats[status] += 1
        if result is not None:
            results.append((inchikey_conn, result))
        if error_line is not None:
            error_lines.append(error_line)
        if invalid_line is not None:
            invalid_smiles_lines.append(invalid_line)

    return results, error_lines, invalid_smiles_lines, chunk_stats


def count_lines(filepath):
    """Count total lines in file for progress bar."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def chunked_reader(filepath, chunk_size):
    """Read file in chunks of lines."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def process_pubchem(
    input_path,
    output_path,
    log_path,
    errors_path,
    invalid_smiles_path,
    num_workers=None,
):
    """Process PubChem file with multiprocessing."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Setup logging
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("PubChem Filtering Started")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"RDKit errors: {errors_path}")
    logger.info(f"Invalid SMILES: {invalid_smiles_path}")
    logger.info(f"Workers: {num_workers}")
    logger.info("")

    # Count total lines for progress bar
    logger.info("Counting lines...")
    total_lines = count_lines(input_path)
    logger.info(f"Total lines: {total_lines:,}")

    # Calculate number of chunks
    num_chunks = (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Aggregate stats
    stats = {
        "total": 0,
        "empty_line": 0,
        "malformed": 0,
        "invalid_smiles": 0,
        "invalid_elements": 0,
        "mw_too_high": 0,
        "rdkit_error": 0,
        "passed_filters": 0,
        "duplicate": 0,
        "saved": 0,
    }

    # Track seen InChIKey connectivity blocks for deduplication
    seen_inchikeys = set()

    with (
        open(output_path, "w", encoding="utf-8") as outfile,
        open(errors_path, "w", encoding="utf-8") as errfile,
        open(invalid_smiles_path, "w", encoding="utf-8") as invalid_file,
    ):
        # Write headers
        outfile.write("\t".join(["ID", "SMILES", "InChIKey", "MW"]) + "\n")
        errfile.write("\t".join(["ID", "SMILES", "Error"]) + "\n")
        invalid_file.write("\t".join(["ID", "SMILES"]) + "\n")

        # Process with multiprocessing
        with mp.Pool(num_workers) as pool:
            chunks = chunked_reader(input_path, CHUNK_SIZE)

            for results, error_lines, invalid_lines, chunk_stats in tqdm(
                pool.imap(process_chunk, chunks),
                total=num_chunks,
                desc=f"Filtering ({num_workers} cores)",
            ):
                # Aggregate stats from chunk
                for key in chunk_stats:
                    stats[key] += chunk_stats[key]

                # Write results (deduplicate by InChIKey connectivity)
                for inchikey_conn, result in results:
                    if inchikey_conn not in seen_inchikeys:
                        seen_inchikeys.add(inchikey_conn)
                        outfile.write(result)
                        stats["saved"] += 1
                    else:
                        stats["duplicate"] += 1

                # Write error lines
                for error_line in error_lines:
                    errfile.write(error_line)

                # Write invalid SMILES lines
                for invalid_line in invalid_lines:
                    invalid_file.write(invalid_line)

    # Log stats
    logger.info("")
    logger.info("=" * 60)
    logger.info("FILTERING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Total lines:        {stats['total']:>12,}")
    logger.info(f"  Empty lines:        {stats['empty_line']:>12,}")
    logger.info(f"  Malformed lines:    {stats['malformed']:>12,}")
    logger.info(f"  Invalid SMILES:     {stats['invalid_smiles']:>12,}")
    logger.info(f"  Invalid elements:   {stats['invalid_elements']:>12,}")
    logger.info(f"  MW > 750:           {stats['mw_too_high']:>12,}")
    logger.info(f"  RDKit errors:       {stats['rdkit_error']:>12,}")
    logger.info("-" * 60)
    logger.info(f"  Passed filters:     {stats['passed_filters']:>12,}")
    pct_dup = (
        100 * stats["duplicate"] / stats["passed_filters"]
        if stats["passed_filters"] > 0
        else 0
    )
    logger.info(
        f"  Duplicates:         {stats['duplicate']:>12,} ({pct_dup:.1f}%)"
    )
    logger.info("-" * 60)
    filtered_out = stats["total"] - stats["saved"]
    pct_filtered = (
        100 * filtered_out / stats["total"] if stats["total"] > 0 else 0
    )
    pct_saved = (
        100 * stats["saved"] / stats["total"] if stats["total"] > 0 else 0
    )
    logger.info(
        f"  Filtered out:       {filtered_out:>12,} ({pct_filtered:.1f}%)"
    )
    logger.info(
        f"  Unique saved:       {stats['saved']:>12,} ({pct_saved:.1f}%)"
    )
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Log file: {log_path}")
    logger.info(f"RDKit errors file: {errors_path}")
    logger.info(f"Invalid SMILES file: {invalid_smiles_path}")
    logger.info("Done!")

    # Also print summary to console
    print(f"\nDone! Saved {stats['saved']:,} unique molecules.")
    print(
        f"  (Removed {stats['duplicate']:,} duplicates by InChIKey connectivity)"
    )
    print(f"Log: {log_path}")
    print(f"RDKit errors: {errors_path}")
    print(f"Invalid SMILES: {invalid_smiles_path}")


if __name__ == "__main__":
    process_pubchem(
        INPUT_FILE,
        OUTPUT_FILE,
        LOG_FILE,
        RDKIT_ERRORS_FILE,
        INVALID_SMILES_FILE,
    )
