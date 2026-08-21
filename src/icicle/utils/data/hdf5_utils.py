# hdf5_utils.py
"""HDF5 file handling utilities."""

from typing import Any, Dict, List

import h5py
import numpy as np


class HDF5Dataset:
    """A dataset as a HDF5 file."""

    def __init__(self, path: Any, mode: str = "r"):
        self.path = path
        self.h5_obj = h5py.File(path, mode=mode)
        self.attrs = self.h5_obj.attrs

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and clean up the resource."""
        self.close()

    def __getitem__(self, idx: int) -> Any:
        return self.h5_obj[idx]

    def __setitem__(self, key: Any, value: Any) -> Any:
        self.h5_obj[key] = value

    def __contains__(self, idx: int) -> Any:
        return idx in self.h5_obj

    def get_all_names(self) -> Any:
        return self.h5_obj.keys()

    def read_str(self, name: str, encoding: str = "utf-8") -> str:
        """Read a string from the HDF5 file."""

        if "/" in name:
            groupname, name = name.rsplit("/", 1)
            grp = self.h5_obj[groupname]
        else:
            grp = self.h5_obj
        str_obj = grp[name][0]

        if type(str_obj) is not bytes:
            raise TypeError(f"Wrong type of {name}")
        return str_obj.decode(encoding)

    def write_str(self, name: str, data: Any) -> Any:
        """Write a string to the HDF5 file."""
        if "/" in name:
            groupname, name = name.rsplit("/", 1)
            grp = self.h5_obj.require_group(groupname)
        else:
            grp = self.h5_obj
        dt = h5py.special_dtype(vlen=str)
        ds = grp.create_dataset(name, (1,), dtype=dt, compression="gzip")
        ds[0] = data

    def write_dict(self, dict: Dict[str, str]) -> Any:
        """Write a dictionary to the HDF5 file."""
        for filename, data in dict.items():
            self.write_str(filename, data)

    def write_list_of_tuples(self, list_of_tuples: List[Any]) -> Any:
        """Write a list of tuples to the HDF5 file."""
        for tup in list_of_tuples:
            if tup is None:
                continue
            self.write_str(tup[0], tup[1])

    def read_data(self, name: str) -> np.ndarray:
        """Read a numpy array object from the HDF5 file."""
        return self.h5_obj[name][:]

    def write_data(self, name: str, data: Any) -> Any:
        """Write a numpy array object to the HDF5 file."""
        self.h5_obj.create_dataset(name, data=data)

    def read_attr(self, name: str) -> dict:
        """Read attribute of name as a dict from the HDF5 file."""
        return {k: v for k, v in self.h5_obj[name].attrs.items()}

    def update_attr(self, name: str, inp_dict: Dict[Any, Any]) -> Any:
        """Write inp_dict to name's attribute in the HDF5 file."""
        cur_obj = self.h5_obj[name].attrs
        for k, v in inp_dict.items():
            cur_obj[k] = v

    def close(self) -> Any:
        """Close the HDF5 file."""
        self.h5_obj.close()

    def flush(self) -> Any:
        """Flush the HDF5 file."""
        self.h5_obj.flush()
