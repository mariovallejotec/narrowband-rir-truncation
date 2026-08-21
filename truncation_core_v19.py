"""Core truncation functions shared by both pipelines: generic wav RIR
loading (mono and multichannel), DC-offset correction, per-bin Lundeby,
truncation-excess ratio, healthy-neighbour median replacement (band
expansion), baseline Lundeby truncation. Format-agnostic and
dataset-agnostic by design: no dataset-specific file formats or names
belong here. For dataset-specific ingestion (MATLAB .mat, HDF5, etc.) see
dataset_loaders_v19.py, which reuses `remove_dc_offset` from this file
instead of duplicating the correction.
"""
import numpy as np
from scipy import signal
from scipy.signal import ShortTimeFFT
import soundfile as sf


def load_rir(path, target_fs=48000, channel=0):
    """Load WAV, take one channel, resample to target_fs. No peak
    normalization: removed in v18, was an unnecessary music-production
    habit (export the normalized master to avoid clipping), not needed for
    this pipeline."""
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
    if fs_file != target_fs:
        rir = signal.resample(rir_raw, int(len(rir_raw) * target_fs / fs_file))
    else:
        rir = rir_raw.copy()
    # Remove DC offset: a constant hardware offset in the recording chain
    # never decays, so the 0 Hz bin's envelope stays flat and the noise
    # threshold can never be crossed, leaving that bin untruncated.
    rir = rir - np.mean(rir)
    return rir, target_fs


def load_rir_multichannel(path, target_fs=48000):
    """Load WAV, keep ALL channels, resample and DC-correct each
    independently. Added in v18 so multichannel RIRs (stereo, ambisonic,
    etc.) can be truncated channel by channel instead of picking one
    arbitrarily.

    Returns
    -------
    rir : 2-D array (n_samples, n_channels)
    target_fs : int
    n_channels : int
    """
    rir_raw, fs_file = sf.read(path, always_2d=True)
    n_channels = rir_raw.shape[1]
    if rir_raw.size == 0:
        raise ValueError("file contains no samples.")
    if not np.all(np.isfinite(rir_raw)):
        raise ValueError("file contains non-finite samples (NaN or Inf).")

    channels = []
    for ch in range(n_channels):
        x = rir_raw[:, ch]
        if fs_file != target_fs:
            x = signal.resample(x, int(len(x) * target_fs / fs_file))
        else:
            x = x.copy()
        x = x - np.mean(x)
        channels.append(x)

    rir = np.stack(channels, axis=1)
    return rir, target_fs, n_channels


def remove_dc_offset(rir):
    """Subtract the mean: a constant hardware offset in the recording chain
    never decays, so the 0 Hz bin's envelope stays flat and the noise
    threshold can never be crossed, leaving that bin untruncated. Shared
    helper so every format-specific loader (see dataset_loaders_v19.py)
    applies the exact same correction instead of duplicating this line."""
    return rir - np.mean(rir)


# ------------------------------------------------------------------------------
# Per-bin Lundeby on STFT energy envelopes
# ------------------------------------------------------------------------------


def _onset_from_noise_floor(env_dB, times, peak_idx, noise_dB):
    """Onset time: last upward crossing of the noise floor before the peak.

    Mirrors the truncation crossing (last downward crossing after the peak)
    but on the rising side, using the same noise_dB estimate. If the bin
    never dips below the noise floor before the peak (e.g. the direct sound
    is in the very first frame), the onset defaults to times[0] -- i.e. no
    front trimming for that bin, the safe/conservative fallback.
    """
    above = env_dB[:peak_idx + 1] > noise_dB
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(crossings) == 0:
        return times[0]
    onset_idx = crossings[-1] + 1
    return float(times[onset_idx])


def _lundeby_per_bin(energy_envelope, times, dB_above_noise=10,
                     max_iter=30, tol=0.01):
    """Apply Lundeby's algorithm to a single energy envelope.

    Parameters
    ----------
    energy_envelope : 1-D array
        |Z[f, :]|^2 for one frequency bin across all STFT frames.
    times : 1-D array
        Center time of each STFT frame.
    dB_above_noise : float
        Regression stops this many dB above the noise floor.
    max_iter : int
        Maximum number of Lundeby iterations.
    tol : float
        Convergence tolerance in seconds for the crossing point.

    Returns
    -------
    trunc_time : float
        Truncation time in seconds. If estimation fails, returns times[-1].
    onset_time : float
        Onset time in seconds (last upward noise-floor crossing before the
        peak). If estimation fails, returns times[0] (no front trimming).
    """
    n_frames = len(energy_envelope)
    dur = times[-1]

    # Avoid log of zero
    env = energy_envelope.copy()
    env[env <= 0] = 1e-30
    env_dB = 10 * np.log10(env)

    # Initial noise estimate: mean of last 10% of frames
    n_tail = max(int(np.round(n_frames * 0.1)), 1)
    noise_est = np.mean(env[-n_tail:])
    if noise_est <= 0:
        return dur, times[0]
    noise_dB = 10 * np.log10(noise_est)

    # Peak index
    peak_idx = np.argmax(env_dB)

    # Initial regression: from peak to dB_above_noise above noise
    above_noise = np.where(env_dB[peak_idx:] > noise_dB + dB_above_noise)[0]
    if len(above_noise) < 2:
        return dur, _onset_from_noise_floor(env_dB, times, peak_idx, noise_dB)
    stop_idx = peak_idx + above_noise[-1]
    start_idx = peak_idx

    if stop_idx <= start_idx:
        return dur, _onset_from_noise_floor(env_dB, times, peak_idx, noise_dB)

    # First regression
    t_reg = times[start_idx:stop_idx + 1]
    e_reg = env_dB[start_idx:stop_idx + 1]
    A = np.vstack([np.ones(len(t_reg)), t_reg]).T
    result = np.linalg.lstsq(A, e_reg, rcond=None)
    slope = result[0]  # [intercept, slope]

    if slope[1] >= 0 or np.any(np.isnan(slope)):
        return dur, _onset_from_noise_floor(env_dB, times, peak_idx, noise_dB)

    # Initial crossing point
    crossing = (noise_dB - slope[0]) / slope[1]

    # Iterative refinement
    for _ in range(max_iter):
        old_crossing = crossing

        # Re-estimate noise: mean from 10 dB below crossing to end
        rel_decay = 10.0 / slope[1]  # time span for 10 dB
        cutoff_time = crossing - rel_decay
        cutoff_idx = np.searchsorted(times, cutoff_time)
        cutoff_idx = max(cutoff_idx, 1)
        last_90 = int(np.round(n_frames * 0.9))
        noise_start = min(cutoff_idx, last_90)
        noise_est = np.mean(env[noise_start:])
        if noise_est <= 0:
            return dur, _onset_from_noise_floor(env_dB, times, peak_idx, noise_dB)
        noise_dB = 10 * np.log10(noise_est)

        # Re-regression
        above = np.where(env_dB[peak_idx:] > noise_dB + dB_above_noise)[0]
        if len(above) < 2:
            break
        stop_idx = peak_idx + above[-1]
        if stop_idx <= peak_idx:
            break

        t_reg = times[peak_idx:stop_idx + 1]
        e_reg = env_dB[peak_idx:stop_idx + 1]
        A = np.vstack([np.ones(len(t_reg)), t_reg]).T
        result = np.linalg.lstsq(A, e_reg, rcond=None)
        slope = result[0]

        if slope[1] >= 0 or np.any(np.isnan(slope)):
            break

        crossing = (noise_dB - slope[0]) / slope[1]

        if abs(old_crossing - crossing) < tol:
            break

    onset_time = _onset_from_noise_floor(env_dB, times, peak_idx, noise_dB)

    # Clamp to valid range
    if crossing < 0 or crossing > dur or np.isnan(crossing):
        return dur, onset_time

    return float(crossing), onset_time


def lundeby_per_stft_bin(Z, times):
    """Apply Lundeby's algorithm to each STFT frequency bin.

    Parameters
    ----------
    Z : 2-D complex array (n_freqs, n_frames)
        STFT coefficients.
    times : 1-D array
        Center time of each frame.

    Returns
    -------
    trunc_times : 1-D array (n_freqs,)
        Truncation time per frequency bin.
    onset_times : 1-D array (n_freqs,)
        Onset time per frequency bin.
    """
    n_freqs = Z.shape[0]
    trunc_times = np.zeros(n_freqs)
    onset_times = np.zeros(n_freqs)

    for i_f in range(n_freqs):
        energy_env = np.abs(Z[i_f, :]) ** 2
        trunc_times[i_f], onset_times[i_f] = _lundeby_per_bin(energy_env, times)

    return trunc_times, onset_times


# ------------------------------------------------------------------------------
# Main truncation function
# ------------------------------------------------------------------------------


def band_expand_replace(init_trunc, freqs, flagged_central, band_hz=100.0,
                        replacement_fraction=1):
    """Expand each flagged bin into a ±band_hz/2 strip and replace every bin
    inside the strip with the median truncation time of its healthy
    neighbours: bins inside the 1/replacement_fraction-octave neighbourhood
    around the flagged bin that lie outside the strip and are not flagged
    themselves (so multi-bin strips don't contaminate the median). Strips
    with fewer than 4 healthy neighbours are left untouched.

    Returns
    -------
    final_trunc : 1-D array (n_freqs,)
        Corrected truncation time per bin.
    expanded_mask : 1-D bool array (n_freqs,)
        True for every bin whose truncation time was replaced.
    """
    n_freqs = len(freqs)
    final_trunc = init_trunc.copy()
    expanded_mask = np.zeros(n_freqs, dtype=bool)
    half_replace_ratio = 2.0 ** (0.5 / replacement_fraction)
    half_band = band_hz / 2.0
    flagged_set = set(np.where(flagged_central)[0].tolist())

    for i in np.where(flagged_central)[0]:
        f0 = freqs[i]
        if f0 <= 0:
            continue
        # Strip = ±half_band Hz around f0
        strip_idx_lo = np.searchsorted(freqs, f0 - half_band, side='left')
        strip_idx_hi = np.searchsorted(freqs, f0 + half_band, side='right')
        nb_lo = np.searchsorted(freqs, f0 / half_replace_ratio, side='left')
        nb_hi = np.searchsorted(freqs, f0 * half_replace_ratio, side='right')
        nb_lo = max(nb_lo, 0); nb_hi = min(nb_hi, n_freqs)
        outside_strip = [j for j in range(nb_lo, nb_hi)
                         if (j < strip_idx_lo or j >= strip_idx_hi)
                         and j not in flagged_set]
        if len(outside_strip) < 4:
            continue
        replace_val = np.median(init_trunc[outside_strip])
        for j in range(strip_idx_lo, strip_idx_hi):
            final_trunc[j] = replace_val
            expanded_mask[j] = True

    return final_trunc, expanded_mask


def trunc_time_excess_ratio(trunc_times, freqs, fraction=3):
    """For each bin, return trunc_time[i] / median(trunc_time of 1/fraction-oct
    neighbours, excluding self). Bins with no usable neighbourhood: 1.0
    (no excess). Scale-invariant across rooms with very different RT."""
    half = 2.0 ** (0.5 / fraction)
    ratio = np.ones(len(trunc_times))
    for i, f0 in enumerate(freqs):
        if f0 <= 0:
            continue
        idx_lo = np.searchsorted(freqs, f0 / half, side='left')
        idx_hi = np.searchsorted(freqs, f0 * half, side='right')
        idx_lo = max(idx_lo, 0); idx_hi = min(idx_hi, len(freqs))
        idxs = [j for j in range(idx_lo, idx_hi) if j != i]
        if len(idxs) < 4:
            continue
        med = np.median(trunc_times[idxs])
        if med <= 0:
            continue
        ratio[i] = trunc_times[i] / med
    return ratio


def truncate_lundeby(rir, fs, winLen=1024, hop=512):
    """Truncate a single RIR using per-bin Lundeby in the STFT domain."""
    rir_len = len(rir)
    if rir_len < winLen:
        raise ValueError(f"RIR too short: {rir_len} samples, "
                         f"need at least one STFT window (winLen={winLen}).")

    # STFT
    nfft_stft = winLen
    SFT = ShortTimeFFT.from_window('hamming', fs=fs, nperseg=winLen,
                                    noverlap=winLen - hop, fft_mode='onesided',
                                    mfft=2 * nfft_stft - 1)
    Z = SFT.stft(rir)
    freqs = SFT.f
    times = SFT.t(rir_len)
    n_freqs, n_frames = Z.shape

    # Per-bin Lundeby truncation and onset times
    trunc_times, onset_times = lundeby_per_stft_bin(Z, times)

    # Build mask from per-bin truncation and onset times
    mask = np.ones((n_freqs, n_frames), dtype=float)
    for i_f in range(n_freqs):
        mask[i_f, times > trunc_times[i_f]] = 0
        mask[i_f, times < onset_times[i_f]] = 0

    # Apply mask and reconstruct via ISTFT
    Z_masked = Z * mask
    rir_trunc = SFT.istft(Z_masked, k1=rir_len)
    if len(rir_trunc) > rir_len:
        rir_trunc = rir_trunc[:rir_len]
    else:
        rir_trunc = np.pad(rir_trunc, (0, rir_len - len(rir_trunc)))

    return rir_trunc, mask, trunc_times, (freqs, times)
