# -*- coding: utf-8 -*-
"""
truncate_single_v18 - same as v17, minus peak normalization (removed from
load_rir and _save in truncation_core_v19 / this file).

truncate_single_v17 - SINGLE-RIR pipeline (no covariance) + truncation-time
outlier detection with band expansion + onset trimming.

Same band-expanded outlier filter as the covariance pipeline
(truncate_repeated_v17), ported to the baseline single-RIR Lundeby pipeline:

  1. STFT decomposition.
  2. Per-bin Lundeby truncation and onset times (init_trunc, init_onset) —
     onset added in v17, mirroring the covariance pipeline: the front of the
     RIR (silence/noise before the direct sound) is now trimmed here too,
     not just the noise tail.
  3. Outlier detection on truncation-time excess RATIO (scale-invariant):
       - ratio[f] = init_trunc[f] / median(init_trunc in 1/3-octave
         neighbours, excluding self).
       - Flag bins with ratio > trunc_excess_ratio (default 1.25) within the
         audible range [f_lo, f_hi].
  4. Band expansion: for each flagged bin f0, force every bin within
     ±band_hz/2 around f0 to take the median trunc_time of bins JUST OUTSIDE
     that strip but inside the replacement neighbourhood.
  5. Build mask from final_trunc AND init_onset, reconstruct via ISTFT.
"""

import os
import sys
import argparse
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal import spectrogram as scipy_spectrogram
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truncation_core_v19 import (
    load_rir, load_rir_multichannel, lundeby_per_stft_bin,
    trunc_time_excess_ratio, band_expand_replace, truncate_lundeby,
)


API_VERSION = "v19-single-band-expanded-outlier-onset-no-peak-norm-multichannel-self-contained"


def truncate_lundeby_band_expanded(rir, fs, winLen=1024, hop=512,
                                    trunc_excess_ratio=1.25, band_hz=150.0,
                                    neighbour_fraction=3,
                                    replacement_fraction=1,
                                    f_lo=100.0, f_hi=18000.0):
    """Single-RIR Lundeby truncation with band-expanded outlier correction
    and onset trimming.

    Detection: a bin is flagged when the ratio of its Lundeby truncation time
    to the median of its 1/3-octave neighbours exceeds `trunc_excess_ratio`,
    AND its centre freq is within [f_lo, f_hi] (audible range, no ultrasonic
    Nyquist artefacts).
    """
    rir_len = len(rir)
    if rir_len < winLen:
        raise ValueError(f"RIR too short: {rir_len} samples, "
                         f"need at least one STFT window (winLen={winLen}).")

    nfft_stft = winLen
    SFT = ShortTimeFFT.from_window('hamming', fs=fs, nperseg=winLen,
                                    noverlap=winLen - hop, fft_mode='onesided',
                                    mfft=2 * nfft_stft - 1)
    Z = SFT.stft(rir)
    freqs = SFT.f
    times = SFT.t(rir_len)
    n_freqs, n_frames = Z.shape

    init_trunc, init_onset = lundeby_per_stft_bin(Z, times)

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

    Z_masked = Z * mask
    rir_trunc = SFT.istft(Z_masked, k1=rir_len)
    if len(rir_trunc) > rir_len:
        rir_trunc = rir_trunc[:rir_len]
    else:
        rir_trunc = np.pad(rir_trunc, (0, rir_len - len(rir_trunc)))

    return (rir_trunc, mask, final_trunc, init_trunc, flagged_central,
            expanded_mask, excess_ratio, (freqs, times))


def truncate_lundeby_band_expanded_multichannel(rir_multi, fs, **kwargs):
    """Truncate every channel of a multichannel RIR independently (v18):
    each column of `rir_multi` (n_samples, n_channels) is run through
    `truncate_lundeby_band_expanded` on its own, instead of picking one
    channel arbitrarily for the whole file. Per-channel results are stacked
    back into a multichannel array so channels stay time-aligned in the
    output.

    Returns
    -------
    rir_trunc_multi : 2-D array (n_samples, n_channels)
    per_channel : list of the full per-channel result tuples from
        `truncate_lundeby_band_expanded`, in channel order (useful for
        per-channel diagnostics/plots).
    """
    n_channels = rir_multi.shape[1]
    per_channel = []
    trunc_channels = []
    for ch in range(n_channels):
        result = truncate_lundeby_band_expanded(rir_multi[:, ch].copy(), fs, **kwargs)
        per_channel.append(result)
        trunc_channels.append(result[0])
    rir_trunc_multi = np.stack(trunc_channels, axis=1)
    return rir_trunc_multi, per_channel


def plot_compare_2panel(rir_orig, rir_cleaned, fs, title=None, out_path=None,
                         nperseg=1024, noverlap=768):
    """2-panel comparison: original (as recorded) vs. cleaned. Used for the
    final dataset release plots, not for validation."""
    f, t, S_orig = scipy_spectrogram(rir_orig, fs=fs, nperseg=nperseg,
                                      noverlap=noverlap, window='hamming')
    _, _, S_clean = scipy_spectrogram(rir_cleaned, fs=fs, nperseg=nperseg,
                                       noverlap=noverlap, window='hamming')

    S_orig_dB = 10 * np.log10(S_orig + 1e-20)
    S_clean_dB = 10 * np.log10(S_clean + 1e-20)
    ref_max = S_orig_dB.max()
    S_orig_dB -= ref_max
    S_clean_dB -= ref_max

    norm = Normalize(vmin=-80, vmax=0)
    panels = [
        ('Raw measured RIR', S_orig_dB),
        ('Narrowband-truncated RIR', S_clean_dB),
    ]

    fig = plt.figure(figsize=(11, 6))
    for idx, (label, S) in enumerate(panels):
        ax = fig.add_subplot(2, 1, idx + 1)
        ax.pcolormesh(t, f, S, cmap='coolwarm', norm=norm, shading='gouraud',
                      rasterized=True)
        ax.set_ylim(20, 20000)
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(label, fontsize=10)
        if idx == 1:
            ax.set_xlabel('Time (s)')

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def _save(rir_trunc, fs, in_path, out_dir, suffix="_lundeby_v18_truncated"):
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{base}{suffix}.wav")
    sf.write(out_path, rir_trunc, fs)
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Single-RIR Lundeby truncation with band-expanded outlier filter (v18, no peak normalization).")
    p.add_argument("input", help="Path to input WAV file")
    p.add_argument("--out", default=".", help="Output directory")
    p.add_argument("--channel", type=int, default=0,
                   help="Ignored if --all-channels is set.")
    p.add_argument("--all-channels", action="store_true",
                   help="Truncate every channel independently (v18) instead "
                        "of picking one channel for the whole file. Output "
                        "wav keeps all channels, time-aligned.")
    p.add_argument("--winlen", type=int, default=1024)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--trunc-excess-ratio", type=float, default=1.25,
                   help="Bin flagged if its trunc_time / 1/3-oct median exceeds this ratio (default 1.25 = 25%% over median).")
    p.add_argument("--band-hz", type=float, default=150.0)
    p.add_argument("--neighbour-fraction", type=int, default=3,
                   help="1/N-octave for detection (default 3 = 1/3 octave).")
    p.add_argument("--replacement-fraction", type=int, default=1,
                   help="1/N-octave for replacement pool (default 1 = full octave).")
    p.add_argument("--f-lo", type=float, default=100.0)
    p.add_argument("--f-hi", type=float, default=18000.0)
    p.add_argument("--plot", action="store_true",
                   help="4-panel compare: original, Lundeby base, Lundeby+band filter, removed.")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        p.error(f"input file not found: {args.input}")
    if not 0 < args.hop <= args.winlen:
        p.error(f"--hop must be between 1 and --winlen "
                f"(got hop={args.hop}, winlen={args.winlen})")
    os.makedirs(args.out, exist_ok=True)

    if args.all_channels:
        try:
            rir_multi, fs, n_channels = load_rir_multichannel(args.input, target_fs=48000)
        except ValueError as e:
            p.error(str(e))
        if rir_multi.shape[0] < args.winlen:
            p.error(f"input too short: {rir_multi.shape[0]} samples after resampling to "
                    f"{fs} Hz, need at least one STFT window ({args.winlen} samples)")
        dur = rir_multi.shape[0] / fs
        print(f"Loaded: {os.path.basename(args.input)} | {dur:.2f}s @ {fs}Hz | {n_channels} channel(s), all truncated independently")

        rir_trunc_multi, per_channel = truncate_lundeby_band_expanded_multichannel(
            rir_multi, fs, winLen=args.winlen, hop=args.hop,
            trunc_excess_ratio=args.trunc_excess_ratio, band_hz=args.band_hz,
            neighbour_fraction=args.neighbour_fraction,
            replacement_fraction=args.replacement_fraction,
            f_lo=args.f_lo, f_hi=args.f_hi)

        for ch, result in enumerate(per_channel):
            flagged, expanded_mask, excess_ratio = result[4], result[5], result[6]
            freqs = result[7][0]
            n_flagged = int(flagged.sum())
            n_modified = int(expanded_mask.sum())
            print(f"  [ch {ch}] Flagged central bins: {n_flagged} | "
                  f"modified by band expansion: {n_modified}")

        out_v18 = _save(rir_trunc_multi, fs, args.input, args.out)
        print(f"  v18 all-channels output: {out_v18}")

        if args.plot:
            base = os.path.splitext(os.path.basename(args.input))[0]
            for ch in range(n_channels):
                rir_ch = rir_multi[:, ch]
                rir_trunc_ch = per_channel[ch][0]
                png_path = os.path.join(args.out, f"{base}_2panel_v18_ch{ch}.png")
                plot_compare_2panel(rir_ch, rir_trunc_ch, fs,
                                    title=f"{base} (channel {ch})", out_path=png_path)
                print(f"  Plot: {png_path}")
        return

    try:
        rir, fs = load_rir(args.input, target_fs=48000, channel=args.channel)
    except ValueError as e:
        p.error(str(e))
    if len(rir) < args.winlen:
        p.error(f"input too short: {len(rir)} samples after resampling to "
                f"{fs} Hz, need at least one STFT window ({args.winlen} samples)")
    dur = len(rir) / fs
    print(f"Loaded: {os.path.basename(args.input)} | {dur:.2f}s @ {fs}Hz | ch={args.channel}")

    (rir_v16, _, final_trunc, init_trunc, flagged, expanded_mask,
     excess_ratio, (freqs, _)) = truncate_lundeby_band_expanded(
        rir.copy(), fs, winLen=args.winlen, hop=args.hop,
        trunc_excess_ratio=args.trunc_excess_ratio, band_hz=args.band_hz,
        neighbour_fraction=args.neighbour_fraction,
        replacement_fraction=args.replacement_fraction,
        f_lo=args.f_lo, f_hi=args.f_hi)

    n_flagged = int(flagged.sum())
    n_modified = int(expanded_mask.sum())
    print(f"  Flagged central bins: {n_flagged}")
    print(f"  Bins modified by band expansion (±{args.band_hz/2:.0f} Hz): {n_modified}")
    if n_flagged > 0:
        flagged_freqs = freqs[flagged]
        for ff in flagged_freqs:
            i = int(np.argmin(np.abs(freqs - ff)))
            print(f"    bin {ff:.0f} Hz | ratio {excess_ratio[i]:.2f}x | "
                  f"init {init_trunc[i]:.3f} s -> final {final_trunc[i]:.3f} s")

    out_v16 = _save(rir_v16, fs, args.input, args.out)
    print(f"  v18 single output: {out_v16}")

    if args.plot:
        base = os.path.splitext(os.path.basename(args.input))[0]
        png_path = os.path.join(args.out, f"{base}_2panel_v18_single.png")
        plot_compare_2panel(rir, rir_v16, fs,
                            title=base, out_path=png_path)
        print(f"  Plot: {png_path}")


if __name__ == "__main__":
    main()
