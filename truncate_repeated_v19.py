# -*- coding: utf-8 -*-
"""
truncate_repeated_v17 - Covariance pipeline (repeated measurements) with
band-expanded outlier correction, same as the single-RIR pipeline
(truncate_single_v17) but with the initial truncation times estimated from
the covariance of measurement pairs instead of per-bin Lundeby. Onset
trimming was already present here since v16; renamed to v17 to pair with
the single-RIR pipeline, which only gained onset trimming in this version.

Why the detector works on a truncation-time excess RATIO (scale-invariant)
rather than on tail energy in dB:
  A tail-energy detector breaks when RIR duration is much larger than the
  room RT: the last frames are already in the noise floor for every bin, so
  no bin sticks out. Empirical: a 5 s recording of York Guildhall Council
  Chamber (RT ~1.2 s) had a clear ~516-586 Hz narrowband artefact that a
  tail-energy detector missed.

  The "trunc_time ratio vs the 1/3-oct neighbourhood median" detector is
  invariant to recording duration and room RT. The replacement
  neighbourhood is widened to a full octave and excludes flagged bins so
  multi-bin strips don't pollute the median used to rebuild the strip.
  Detection restricted to a configurable audible range (default 100-18000
  Hz) to skip ultrasonic resampling tails.
"""

import numpy as np
from scipy import signal
from scipy.signal import ShortTimeFFT
import soundfile as sf

from utils import unimodal_regression, time_align_rirs
from truncation_core_v19 import trunc_time_excess_ratio, band_expand_replace

API_VERSION = "v19-repeated-trunc-excess-outlier-self-contained"


def load_rir_raw(path, target_fs=None, channel=0):
    rir_raw, fs_file = sf.read(path)
    n_channels = 1 if rir_raw.ndim == 1 else rir_raw.shape[1]
    if not 0 <= channel < n_channels:
        raise ValueError(f"channel {channel} out of range: "
                         f"file has {n_channels} channel(s).")
    if rir_raw.ndim > 1:
        rir_raw = rir_raw[:, channel]
    if rir_raw.size == 0:
        raise ValueError("file contains no samples.")
    if not np.all(np.isfinite(rir_raw)):
        raise ValueError("file contains non-finite samples (NaN or Inf).")
    # Remove DC offset (constant hardware offset never decays and blocks
    # the noise-threshold crossing at the 0 Hz bin; see load_rir).
    rir_raw = rir_raw - np.mean(rir_raw)
    if target_fs is not None and fs_file != target_fs:
        rir = signal.resample(rir_raw, int(len(rir_raw) * target_fs / fs_file))
        return rir, target_fs
    return rir_raw.astype(float).copy(), fs_file


def align_pair(rir_a, rir_b, up_factor=10):
    L = max(len(rir_a), len(rir_b))
    a = np.zeros(L); b = np.zeros(L)
    a[:len(rir_a)] = rir_a; b[:len(rir_b)] = rir_b
    stacked = np.stack([a, b], axis=1)
    up = signal.resample(stacked, L * up_factor, axis=0)
    up_aligned = time_align_rirs(up, referenceID=0)
    aligned = signal.resample(up_aligned, L, axis=0)
    return aligned[:, 0], aligned[:, 1]


def truncate_covariance_band_expanded(rirs, fs, winLen=1024, hop=512,
                                      do_align=True, trunc_excess_ratio=1.25,
                                      band_hz=150.0, neighbour_fraction=3,
                                      replacement_fraction=1,
                                      f_lo=100.0, f_hi=18000.0):
    if len(rirs) < 2:
        raise ValueError("Need at least 2 RIRs.")
    if do_align:
        rirs_aligned = [rirs[0]]
        for r in rirs[1:]:
            _, r_aligned = align_pair(rirs[0], r)
            rirs_aligned.append(r_aligned)
        rirs = rirs_aligned
    else:
        L = max(len(r) for r in rirs)
        rirs = [np.pad(r, (0, L - len(r))) for r in rirs]

    rir_len = len(rirs[0])
    if rir_len < winLen:
        raise ValueError(f"RIR too short: {rir_len} samples, "
                         f"need at least one STFT window (winLen={winLen}).")
    nfft_stft = winLen
    SFT = ShortTimeFFT.from_window('hamming', fs=fs, nperseg=winLen,
                                    noverlap=winLen - hop, fft_mode='onesided',
                                    mfft=2 * nfft_stft - 1)
    Zs = [SFT.stft(r) for r in rirs]
    freqs = SFT.f
    times = SFT.t(rir_len)
    n_freqs, n_frames = Zs[0].shape

    n_pairs = len(Zs) - 1
    Cov_uni_sum = np.zeros((n_freqs, n_frames))
    for p in range(n_pairs):
        Cov = np.abs(Zs[p] * np.conj(Zs[p + 1]))
        for fr in range(n_freqs):
            smooth, _ = unimodal_regression(Cov[fr, :])
            Cov_uni_sum[fr, :] += smooth
    Cov_uni_avg = Cov_uni_sum / n_pairs

    nL = max(int(0.2 * fs / winLen), 1)
    nv = np.sum(Cov_uni_avg[:, -nL:], axis=1)
    noise_var = np.tile(nv[:, None], (1, n_frames))
    init_mask = (Cov_uni_avg > noise_var).astype(float)

    init_trunc = np.full(n_freqs, times[-1])
    init_onset = np.full(n_freqs, times[0])
    for fr in range(n_freqs):
        kept = np.where(init_mask[fr, :] > 0)[0]
        if len(kept) > 0:
            init_trunc[fr] = times[kept[-1]]
            init_onset[fr] = times[kept[0]]

    # Outlier detection on truncation-time excess RATIO (scale-invariant).
    excess_ratio = trunc_time_excess_ratio(init_trunc, freqs,
                                           fraction=neighbour_fraction)
    audible = (freqs >= f_lo) & (freqs <= f_hi)
    flagged_central = (excess_ratio > trunc_excess_ratio) & audible

    final_trunc, expanded_mask = band_expand_replace(
        init_trunc, freqs, flagged_central,
        band_hz=band_hz, replacement_fraction=replacement_fraction)

    mask = np.ones((n_freqs, n_frames), dtype=float)
    for fr in range(n_freqs):
        mask[fr, times > final_trunc[fr]] = 0
        mask[fr, times < init_onset[fr]] = 0

    Z_masked = Zs[0] * mask
    rir_trunc = SFT.istft(Z_masked, k1=rir_len)
    if len(rir_trunc) > rir_len:
        rir_trunc = rir_trunc[:rir_len]
    else:
        rir_trunc = np.pad(rir_trunc, (0, rir_len - len(rir_trunc)))

    return (rir_trunc, mask, final_trunc, init_trunc, flagged_central,
            expanded_mask, excess_ratio, (freqs, times), n_pairs)
