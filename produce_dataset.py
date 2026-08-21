# -*- coding: utf-8 -*-
# PRODUCTION: ONE script for all datasets. The only thing that changes per
# dataset is (a) how the raw RIRs are listed and (b) how their samples are
# loaded -- unavoidable, each dataset comes in a different file format (wav,
# .mat, hdf5); that per-format loading lives in dataset_loaders_v19.py, not
# here. Everything else -- which cleaning algorithm runs (single/Lundeby or
# repeated/covariance), saving the clean wav, the 2-panel plot, being
# resumable -- is the SAME code for all of them, once, below.
#
# Usage: python produce_dataset.py <dataset>
#   dataset in: openair, iks, dechorate, arni
import os, sys, re, glob, json, traceback
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import soundfile as sf

NB = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(NB, "narrowband-rir-truncation")
sys.path.insert(0, REPO)

from truncation_core_v19 import load_rir, load_rir_multichannel
from truncate_single_v19 import (
    truncate_lundeby_band_expanded,
    truncate_lundeby_band_expanded_multichannel,
    plot_compare_2panel,
)
from truncate_repeated_v19 import truncate_covariance_band_expanded, load_rir_raw
from dataset_loaders_v19 import load_iks_mat, load_dechorate_h5

import h5py


def list_and_load_single(dataset):
    """Yields (name, load_fn) for single/Lundeby datasets. load_fn() -> (rir, fs)."""
    if dataset == "openair":
        # v19: keep and truncate every channel independently instead of
        # picking channel 0 (OpenAIR has real stereo and up to 16-channel
        # ambisonic material, not a noise/signal split like MRTD).
        d = os.path.join(NB, "Assets", "Raw", "openair", "rirs")
        for w in sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)):
            if os.path.basename(w).startswith("._"):
                continue
            name = os.path.splitext(os.path.basename(w))[0]
            yield name, (lambda w=w: load_rir_multichannel(w, target_fs=48000)[:2])

    elif dataset == "iks":
        d = os.path.join(NB, "Assets", "Raw", "iks")
        for m in sorted(glob.glob(os.path.join(d, "**", "*.mat"), recursive=True)):
            if os.path.basename(m).startswith("._"):
                continue
            name = os.path.splitext(os.path.basename(m))[0]
            yield name, (lambda m=m: load_iks_mat(m))

    elif dataset == "dechorate":
        h5path = os.path.join(NB, "Assets", "Raw", "dechorate", "dEchorate_rir.h5")
        f = h5py.File(h5path, "r")
        n_mics = int(f.attrs["n_mics"])
        rooms = sorted(f["rir"].keys())
        for room in rooms:
            sources = sorted(f["rir"][room].keys(), key=int)
            for src in sources:
                for mic in range(n_mics):
                    name = f"room{room}_src{src}_mic{mic+1}"
                    yield name, (lambda f=f, room=room, src=src, mic=mic: load_dechorate_h5(f, room, src, mic))
    else:
        raise ValueError(f"unknown single dataset: {dataset}")


def list_and_load_repeated(dataset):
    """Yields (name, load_fn) for repeated/covariance datasets. load_fn() -> (list_of_rirs, fs)."""
    if dataset == "arni":
        d = os.path.join(NB, "Assets", "Raw", "arni", "rirs")
        pat = re.compile(r'IR_numClosed_(\d+)_numComb_(\d+)_mic_(\d+)_sweep_(\d+)')
        wavs = [w for w in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
                if not os.path.basename(w).startswith("._")]
        groups = defaultdict(list)
        for w in wavs:
            m = pat.search(os.path.basename(w))
            if m:
                nC, nComb, mic, sweep = m.groups()
                groups[(int(nC), int(nComb), int(mic))].append((int(sweep), w))
        for k in sorted(groups.keys()):
            name = "numClosed_%d_numComb_%d_mic_%d" % k
            sweeps = [w for _, w in sorted(groups[k])]
            yield name, sweeps
    else:
        raise ValueError(f"unknown repeated dataset: {dataset}")


DATASETS = {
    "openair":         {"pipeline": "single",   "out": "OpenAIR"},
    "iks":             {"pipeline": "single",   "out": "IKS"},
    "dechorate":       {"pipeline": "single",   "out": "dEchorate"},
    "arni":            {"pipeline": "repeated", "out": "Arni"},
}

# Which module version actually produced a given run. TAG goes into the
# output folder name itself (Assets/Cleaned_Dataset/<Name>_<TAG>/) so runs
# from different code versions physically CANNOT land in the same folder --
# update this alongside the imports above when bumping versions.
PIPELINE_TAG = "v19"
PIPELINE_VERSION = {
    "truncation_core": "v19",
    "truncate_single": "v19",
    "truncate_repeated": "v19",
    "dataset_loaders": "v19",
}


def write_version_stamp(out_base, dataset):
    with open(os.path.join(out_base, "PIPELINE_VERSION.json"), "w") as f:
        json.dump({
            "dataset": dataset,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **PIPELINE_VERSION,
        }, f, indent=2)


def run_single(dataset, out_base):
    out_wav = os.path.join(out_base, "wavs")
    out_png = os.path.join(out_base, "plots")
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_png, exist_ok=True)
    write_version_stamp(out_base, dataset)

    items = list(list_and_load_single(dataset))
    print(f"{dataset}: {len(items)} raw RIRs -> clean all", flush=True)

    done = 0
    for i, (name, load_fn) in enumerate(items, 1):
        wav_out = os.path.join(out_wav, f"{name}_truncated.wav")
        png_out = os.path.join(out_png, f"{name}_2panel.png")
        if os.path.exists(wav_out) and os.path.exists(png_out):
            done += 1
            print(f"[{i}/{len(items)}] already done {name}", flush=True)
            continue
        try:
            rir, fs = load_fn()
            if rir.shape[0] < 1024:
                print(f"[{i}/{len(items)}] short {name}", flush=True)
                continue

            if rir.ndim == 2 and rir.shape[1] > 1:
                # Multichannel (v19): truncate every channel independently,
                # save all channels time-aligned in one wav. Plot uses
                # channel 0 only, one PNG per file like the mono case.
                rir_trunc, _per_channel = truncate_lundeby_band_expanded_multichannel(rir, fs)
                peak = np.max(np.abs(rir_trunc)) + 1e-12
                out_norm = rir_trunc / peak * (10 ** (-18 / 20))
                sf.write(wav_out, out_norm, fs, subtype="FLOAT")
                plot_compare_2panel(rir[:, 0], rir_trunc[:, 0], fs, title=name, out_path=png_out)
            else:
                rir_1d = rir[:, 0] if rir.ndim == 2 else rir
                rir_v16 = truncate_lundeby_band_expanded(rir_1d.copy(), fs)[0]
                # Final render: peak at -18 dBFS (standard operating level,
                # EBU R68), float32. With impulsive material the operating level
                # is applied at peak: RMS -18 would give peaks of +7..+22 dBFS
                # (clipping).
                out_norm = rir_v16 / (np.max(np.abs(rir_v16)) + 1e-12) * (10 ** (-18 / 20))
                sf.write(wav_out, out_norm, fs, subtype="FLOAT")
                plot_compare_2panel(rir_1d, rir_v16, fs, title=name, out_path=png_out)
            done += 1
            print(f"[{i}/{len(items)}] ok {name}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(items)}] ERR {name}: {e}", flush=True)
            traceback.print_exc()

    print(f"\nDONE. {done} of {len(items)} {dataset} cleaned in: {out_base}", flush=True)


def run_repeated(dataset, out_base, limit=None):
    out_wav = os.path.join(out_base, "wavs")
    out_png = os.path.join(out_base, "plots")
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_png, exist_ok=True)
    write_version_stamp(out_base, dataset)

    items = list(list_and_load_repeated(dataset))
    total_real = len(items)
    if limit is not None:
        items = items[:limit]
        print(f"{dataset}: {total_real} total positions, limited to the first {len(items)}", flush=True)
    else:
        print(f"{dataset}: {len(items)} positions -> clean all", flush=True)

    done = 0
    for i, (name, sweeps) in enumerate(items, 1):
        wav_out = os.path.join(out_wav, f"{name}_truncated.wav")
        png_out = os.path.join(out_png, f"{name}_2panel.png")
        if os.path.exists(wav_out) and os.path.exists(png_out):
            done += 1
            print(f"[{i}/{len(items)}] already done {name}", flush=True)
            continue
        if len(sweeps) < 2:
            print(f"[{i}/{len(items)}] skip {name}: <2 repetitions", flush=True)
            continue
        # Per the JASA RIR-cropping paper (Prawda et al.): "the proposed
        # method works best with a pair of RIRs... in the case of Arni, we
        # used two consecutive measurements". The method is defined for a
        # PAIR, not for averaging over all available repetitions (Arni has
        # 4-5 per position). Use only the first 2.
        sweeps = sweeps[:2]
        try:
            # Arni is mono.
            rirs = [load_rir_raw(w, target_fs=48000, channel=0)[0] for w in sweeps]
            fs = 48000
            rir_v16 = truncate_covariance_band_expanded(rirs, fs)[0]
            rir_orig = rirs[0][:len(rir_v16)]
            if len(rir_orig) < len(rir_v16):
                rir_orig = np.pad(rir_orig, (0, len(rir_v16) - len(rir_orig)))
            # Final render: peak at -18 dBFS (standard operating level,
            # EBU R68), float32. With impulsive material the operating level
            # is applied at peak: RMS -18 would give peaks of +7..+22 dBFS
            # (clipping).
            out_norm = rir_v16 / (np.max(np.abs(rir_v16)) + 1e-12) * (10 ** (-18 / 20))
            sf.write(wav_out, out_norm, fs, subtype="FLOAT")
            plot_compare_2panel(rir_orig, rir_v16, fs, title=name, out_path=png_out)
            done += 1
            print(f"[{i}/{len(items)}] ok {name} ({len(sweeps)} repetitions)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(items)}] ERR {name}: {e}", flush=True)
            traceback.print_exc()

    print(f"\nDONE. {done} of {len(items)} {dataset} cleaned in: {out_base}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in DATASETS:
        print(f"Usage: python produce_dataset.py <{'|'.join(DATASETS.keys())}> [--pct N | --limit N]")
        sys.exit(1)
    dataset = sys.argv[1]
    cfg = DATASETS[dataset]
    out_base = os.path.join(NB, "Assets", "Cleaned_Dataset", f"{cfg['out']}_{PIPELINE_TAG}")

    limit = None
    if len(sys.argv) > 3 and sys.argv[2] == "--pct":
        pct = float(sys.argv[3])
        total = len(list(list_and_load_repeated(dataset))) if cfg["pipeline"] == "repeated" else len(list(list_and_load_single(dataset)))
        limit = round(total * pct / 100)
    elif len(sys.argv) > 3 and sys.argv[2] == "--limit":
        limit = int(sys.argv[3])

    if cfg["pipeline"] == "single":
        run_single(dataset, out_base)
    else:
        run_repeated(dataset, out_base, limit=limit)
