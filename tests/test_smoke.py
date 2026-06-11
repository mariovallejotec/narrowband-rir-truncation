"""Smoke test: runs both pipelines on a synthetic RIR (no external data).

A synthetic RIR is built as exponentially decaying noise. The repeated
pipeline gets two noisy realisations of the same decay; the single pipeline
gets one. The test checks that both return finite outputs of the right shape.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truncate_repeated_v16 import truncate_covariance_band_expanded
from truncate_single_v16 import truncate_lundeby_band_expanded

FS = 48000
N = FS  # 1 s


def synthetic_rir(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    decay = np.exp(-t / 0.15)              # RT-like decay
    noise_floor = 1e-4 * rng.standard_normal(N)
    return decay * rng.standard_normal(N) + noise_floor


def main():
    rir_a, rir_b = synthetic_rir(1), synthetic_rir(2)

    out = truncate_covariance_band_expanded([rir_a, rir_b], FS, do_align=False)
    rir_clean = out[0]
    assert rir_clean.shape == rir_a.shape, "repeated: wrong output shape"
    assert np.all(np.isfinite(rir_clean)), "repeated: non-finite output"
    print("repeated pipeline: OK")

    out = truncate_lundeby_band_expanded(rir_a, FS)
    rir_clean = out[0]
    assert rir_clean.shape == rir_a.shape, "single: wrong output shape"
    assert np.all(np.isfinite(rir_clean)), "single: non-finite output"
    print("single pipeline: OK")

    print("smoke test passed")


if __name__ == "__main__":
    main()
