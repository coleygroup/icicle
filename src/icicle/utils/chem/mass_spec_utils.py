"""Mass spectrometry-related utilities."""

from itertools import groupby
from pathlib import Path, PosixPath
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def parse_spectra(
    spectra_file: Union[str, List[str], Path],
) -> Tuple[Dict[str, str], List[Tuple[str, NDArray[np.float64]]]]:
    """Parse spectra in the SIRIUS format.

    Args:
        spectra_file: Name of spectra file to parse or lines of parsed spectra.
                     Can be a string path, Path object, or list of lines.

    Returns
    -------
        Tuple containing:
            - Dictionary of metadata
            - List of tuples, each containing:
                - String: spectrum header
                - ndarray: peak data array
    """
    # Handle different input types
    if isinstance(spectra_file, (str, PosixPath)):
        lines = [
            i.strip() for i in Path(spectra_file).read_text().splitlines()
        ]
    elif isinstance(spectra_file, list):
        lines = [i.strip() for i in spectra_file]
    else:
        raise ValueError(
            f"Unsupported type for spectra_file: {type(spectra_file)}"
        )

    group_num = 0
    metadata: Dict[str, str] = {}
    spectras: List[Tuple[str, np.ndarray]] = []

    # Create iterator for grouping lines
    line_groups: Iterator[Tuple[bool, Iterator[str]]] = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, group_lines) in enumerate(line_groups):
        # Convert iterators to lists immediately to avoid exhaustion
        current_lines = list(group_lines)
        try:
            next_group = next(line_groups)
            subject_lines = list(next_group[1])
        except StopIteration:
            break

        # Process spectrum data
        if group_num > 0:
            spectra_header = current_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]

            if peak_data:  # Check if we have any peak data
                peak_array = np.array(peak_data)
                spectras.append((spectra_header, peak_array))

        # Process metadata
        else:
            entries: Dict[str, str] = {}
            for line in current_lines:
                if " " not in line:
                    continue
                elif line.startswith("#INSTRUMENT TYPE"):
                    key = "INSTRUMENT TYPE"  # Remove # from key
                    val = line.split("#INSTRUMENT TYPE")[1].strip()
                    entries[key] = val
                else:
                    start, end = line.split(" ", 1)
                    key = start[1:]  # Remove # or > from key
                    while key in entries:
                        key = f"{key}'"
                    entries[key] = end.strip()

            metadata.update(entries)

        group_num += 1

    # Add file information to metadata if input was a file
    if isinstance(spectra_file, (str, PosixPath)):
        path = Path(spectra_file)
        metadata["_FILE_PATH"] = str(path)
        metadata["_FILE"] = path.stem

    return metadata, spectras


def process_common_spec_file(
    meta: Dict[str, Union[str, float]],
    tuples: List[Tuple[str, NDArray[np.float64]]],
    precision: int = 4,
    merge_specs: bool = True,
    exclude_parent: bool = False,
) -> Optional[Union[NDArray[np.float64], Dict[str, NDArray[np.float64]]]]:
    """Process and normalize mass spectrometry data.

    This function processes mass spectrometry data by optionally merging multiple spectra
    and normalizing intensities. It can handle parent mass exclusion and intensity
    normalization with square root transformation.

    Parameters
    ----------
    meta : Dict[str, Union[str, float]]
        Metadata dictionary containing at least 'parentmass' key
    tuples : List[Tuple[str, NDArray[np.float64]]]
        List of tuples containing (collision energy, spectrum array)
    precision : int, optional
        Decimal precision for m/z values rounding, by default 4
    merge_specs : bool, optional
        Whether to merge multiple spectra into one, by default True
    exclude_parent : bool, optional
        Whether to exclude peaks above parent mass - 1, by default False

    Returns
    -------
    Optional[Union[NDArray[np.float64], Dict[str, NDArray[np.float64]]]]
        If merge_specs is True:
            Returns numpy array of shape (n, 2) with columns [m/z, intensity]
        If merge_specs is False:
            Returns dictionary mapping collision energies to spectrum arrays
        Returns None if no valid spectra are found
    """
    parent_mass = float(meta.get("parentmass", 1000000))

    # First norm spectra
    fused_tuples = {ce: x for ce, x in tuples if x.size > 0}

    if len(fused_tuples) == 0:
        return None

    if merge_specs:
        mz_to_inten_pair: Dict[float, NDArray[np.float64]] = {}
        new_tuples: List[NDArray[np.float64]] = []

        for spec_array in fused_tuples.values():
            for tup in spec_array:
                mz, inten = tup
                mz_ind = np.round(mz, precision)
                cur_pair = mz_to_inten_pair.get(mz_ind)
                if cur_pair is None:
                    pair_array = np.array([mz, inten], dtype=np.float64)
                    mz_to_inten_pair[mz_ind] = pair_array
                    new_tuples.append(pair_array)
                elif inten > cur_pair[1]:
                    cur_pair[1] = inten  # max merging

        if not new_tuples:
            return None

        if not new_tuples:
            return None

        merged_spec = np.vstack(new_tuples)
        if exclude_parent:
            merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass - 1)]
        else:
            merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass + 1)]
        merged_spec = merged_spec[merged_spec[:, 1] > 0]

        if len(merged_spec) == 0:
            return None

        merged_spec[:, 1] = merged_spec[:, 1] / np.max(merged_spec[:, 1])
        merged_spec[:, 1] = np.sqrt(merged_spec[:, 1])
        return merged_spec
    else:
        new_specs: Dict[str, NDArray[np.float64]] = {}
        for k, v in fused_tuples.items():
            # Convert v to a list of arrays before vstacking
            if isinstance(v, np.ndarray):
                new_spec = v  # If it's already a 2D array, use it directly
            else:
                # If it's a sequence of arrays, vstack them
                new_spec = np.vstack([arr for arr in v])

            new_spec = new_spec[new_spec[:, 0] <= (parent_mass + 1)]
            new_spec = new_spec[new_spec[:, 1] > 0]

            if len(new_spec) == 0:
                continue

            new_spec[:, 1] = new_spec[:, 1] / np.max(new_spec[:, 1])
            new_spec[:, 1] = np.sqrt(new_spec[:, 1])
            new_specs[k] = new_spec

        return new_specs


def filter_spectra_by_intensity(
    spec: NDArray[np.float64], max_num_inten: int = 60, inten_thresh: float = 0
) -> NDArray[np.float64]:
    """Normalize and filter spectra so that the top max_num_inten peaks are
    returned.

    Args:
        spec: 2D spectra array
        max_num_inten: Max number of peaks
        inten_thresh: Min intensity to alloow in returned peak

    Return:
        Spec filtered down
    """
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    # Make sure to only take max of each formula
    # Sort by intensity and select top subpeaks
    new_sort_order = np.argsort(spec_intens)[::-1]
    if max_num_inten is not None:
        new_sort_order = new_sort_order[:max_num_inten]

    spec_masses = spec_masses[new_sort_order]
    spec_intens = spec_intens[new_sort_order]

    spec_mask = spec_intens > inten_thresh
    spec_masses = spec_masses[spec_mask]
    spec_intens = spec_intens[spec_mask]
    spec = np.vstack([spec_masses, spec_intens]).transpose(1, 0)
    return spec
