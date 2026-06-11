# -*- coding: utf-8 -*-
"""
truncate_single_v16 - SINGLE-RIR pipeline (no covariance) + truncation-time
outlier detection with band expansion.

Same band-expanded outlier filter as the covariance pipeline
(truncate_repeated_v16), ported to the baseline single-RIR Lundeby pipeline:

  1. STFT decomposition.
  2. Per-bin Lundeby truncation times (init_trunc) — unchanged from the baseline.
  3. Outlier detection on truncation-time excess RATIO (scale-invariant):
       - ratio[f] = init_trunc[f] / median(init_trunc in 1/3-octave
         neighbours, excluding self).
       - Flag bins with ratio > trunc_excess_ratio (default 1.25) within the
         audible range [f_lo, f_hi].
  4. Band expansion: for each flagged bin f0, force every bin within
     ±band_hz/2 around f0 to take the median trunc_time of bins JUST OUTSIDE
     that strip but inside the replacement neighbourhood.
  5. Build mask from final_trunc and reconstruct via ISTFT.
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
from truncation_core_v16 import (
    load_rir, lundeby_per_stft_bin, trunc_time_excess_ratio,
    band_expand_replace, truncate_lundeby,
)


API_VERSION = "v16-single-band-expanded-outlier"


def truncate_lundeby_band_expanded(rir, fs, winLen=1024, hop=512,
                                    trunc_excess_ratio=1.25, band_hz=100.0,
                                    neighbour_fraction=3,
                                    replacement_fraction=1,
                                    f_lo=100.0, f_hi=18000.0):
    """Single-RIR Lundeby truncation with band-expanded outlier correction.

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

    init_trunc = lundeby_per_stft_bin(Z, times)

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

    Z_masked = Z * mask
    rir_trunc = SFT.istft(Z_masked, k1=rir_len)
    if len(rir_trunc) > rir_len:
        rir_trunc = rir_trunc[:rir_len]
    else:
        rir_trunc = np.pad(rir_trunc, (0, rir_len - len(rir_trunc)))

    return (rir_trunc, mask, final_trunc, init_trunc, flagged_central,
            expanded_mask, excess_ratio, (freqs, times))


def plot_compare_4panel(rir_orig, rir_lundeby_base, rir_lundeby_v16, fs,
                         title=None, out_path=None,
                         nperseg=1024, noverlap=768):
    """4-panel: Original / Lundeby base / Lundeby + band filter / Removed."""
    removed = rir_orig - rir_lundeby_v16

    f, t, S_orig    = scipy_spectrogram(rir_orig, fs=fs, nperseg=nperseg,
                                         noverlap=noverlap, window='hamming')
    _, _, S_base    = scipy_spectrogram(rir_lundeby_base, fs=fs, nperseg=nperseg,
                                         noverlap=noverlap, window='hamming')
    _, _, S_v16    = scipy_spectrogram(rir_lundeby_v16, fs=fs, nperseg=nperseg,
                                         noverlap=noverlap, window='hamming')
    _, _, S_removed = scipy_spectrogram(removed, fs=fs, nperseg=nperseg,
                                         noverlap=noverlap, window='hamming')

    S_orig_dB    = 10 * np.log10(S_orig    + 1e-20)
    S_base_dB    = 10 * np.log10(S_base    + 1e-20)
    S_v16_dB    = 10 * np.log10(S_v16    + 1e-20)
    S_removed_dB = 10 * np.log10(S_removed + 1e-20)

    ref_max = S_orig_dB.max()
    S_orig_dB    -= ref_max
    S_base_dB    -= ref_max
    S_v16_dB    -= ref_max
    S_removed_dB -= ref_max

    norm = Normalize(vmin=-80, vmax=0)
    panels = [
        ('Original RIR exactly as recorded', S_orig_dB),
        ('Per-bin Lundeby, no outlier filter', S_base_dB),
        ('Per-bin Lundeby + band-expanded outlier filter', S_v16_dB),
        ('Removed component (original minus filtered)', S_removed_dB),
    ]

    fig = plt.figure(figsize=(11, 10))
    for idx, (label, S) in enumerate(panels):
        ax = fig.add_subplot(4, 1, idx + 1)
        ax.pcolormesh(t, f, S, cmap='coolwarm', norm=norm, shading='gouraud',
                      rasterized=True)
        ax.set_ylim(20, 20000)
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(label, fontsize=10)
        if idx == 3:
            ax.set_xlabel('Time (s)')

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig


def _save(rir_trunc, fs, in_path, out_dir, suffix="_lundeby_v16_truncated"):
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{base}{suffix}.wav")
    rir_trunc = rir_trunc / (np.max(np.abs(rir_trunc)) + 1e-12)
    sf.write(out_path, rir_trunc, fs)
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Single-RIR Lundeby truncation with band-expanded outlier filter (v16 single).")
    p.add_argument("input", help="Path to input WAV file")
    p.add_argument("--out", default=".", help="Output directory")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--winlen", type=int, default=1024)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--trunc-excess-ratio", type=float, default=1.25,
                   help="Bin flagged if its trunc_time / 1/3-oct median exceeds this ratio (default 1.25 = 25%% over median).")
    p.add_argument("--band-hz", type=float, default=100.0)
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
    print(f"  v16 single output: {out_v16}")

    if args.plot:
        rir_base, _, _, _ = truncate_lundeby(
            rir.copy(), fs, winLen=args.winlen, hop=args.hop)
        base = os.path.splitext(os.path.basename(args.input))[0]
        png_path = os.path.join(args.out, f"{base}_4panel_v16_single.png")
        title = "Single-RIR example"
        plot_compare_4panel(rir, rir_base, rir_v16, fs,
                            title=title, out_path=png_path)
        print(f"  Plot: {png_path}")


if __name__ == "__main__":
    main()
