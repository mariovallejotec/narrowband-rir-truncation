# -*- coding: utf-8 -*-
"""
Dataset-specific RIR ingestion adapters. Deliberately kept OUT of
truncation_core_v19.py: that file is the generic, dataset-agnostic
truncation algorithm, and should never have to know that a dataset called
"IKS" stores its RIRs as MATLAB structs, or that "dEchorate" uses a
particular HDF5 layout. Each loader here converts one dataset's raw file
format into a plain (rir, fs) array/int pair, using the SAME DC-offset
correction as the rest of the pipeline via `remove_dc_offset`, so there is
one single definition of that correction, not one per format.

Add a new dataset's loader here, not in produce_dataset.py and not in
truncation_core_v19.py.
"""
import os
import sys
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truncation_core_v19 import remove_dc_offset


def load_iks_mat(path):
    """Load a single-channel RIR from a MATLAB .mat file in the IKS/Aachen
    AIR format (struct `air_info.fs`, array `h_air`). No resampling -- the
    dataset's own rate is kept."""
    data = sio.loadmat(path)
    info = data["air_info"][0, 0]
    fs = int(info["fs"][0, 0])
    rir = np.asarray(data["h_air"]).astype(float).ravel()
    rir = remove_dc_offset(rir)
    return rir, fs


def load_dechorate_h5(h5_file, room, src, mic):
    """Load a single-channel RIR from an already-open dEchorate HDF5 file
    (`h5py.File`), given room/source/mic keys. Caller owns opening and
    closing the file, since callers typically iterate many room/src/mic
    combinations from the same open handle."""
    fs = int(h5_file.attrs["sampling_rate"])
    rir = np.asarray(h5_file["rir"][room][src][:, mic], dtype=np.float64)
    rir = remove_dc_offset(rir)
    return rir, fs
