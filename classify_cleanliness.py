# -*- coding: utf-8 -*-
# Classifies each cleaned RIR of a dataset into 3 categories, comparing raw vs.
# truncated using per-frequency-band energy (not visual inspection):
#
#   1. NO_TRUNCATION_NEEDED -> truncation removed almost nothing (already clean).
#   2. CORRECTLY_TRUNCATED  -> noise removed proportionally across all bands.
#   3. INCOMPLETE_LOW_FREQUENCY_TRUNCATION -> the low band (<200 Hz) keeps a
#      "bar" of noise/decay that was barely truncated while the mid/high
#      bands were cut correctly.
#
# Method: spectrogram of raw and truncated, low band (<200 Hz) and high band
# (500-8000 Hz), active duration = last frame with energy > -60 dB relative
# to that band's own peak. reduction = 1 - (active_truncated/active_raw).
import os, sys, glob, json
import numpy as np
from scipy.signal import spectrogram
import soundfile as sf

NB = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(NB, "narrowband-rir-truncation")
sys.path.insert(0, REPO)
sys.path.insert(0, NB)

# Reuses the same "raw" loaders that produce_dataset.py uses to generate the
# cleaned dataset -- a single source of truth for how each format is loaded
# (wav, .mat, hdf5), instead of reimplementing it here.
from produce_dataset import list_and_load_single, list_and_load_repeated, DATASETS, PIPELINE_TAG
from truncate_repeated_v19 import load_rir_raw


def band_active_duration(sig, fs, f_lo, f_hi, thresh_db=-60, nperseg=1024, noverlap=768):
    # A truncated RIR can end up shorter than nperseg (e.g. OpenAIR anechoic/
    # direct-sound captures) -- scipy silently caps nperseg to len(sig) in
    # that case, which then makes the fixed noverlap=768 invalid ("noverlap
    # must be less than nperseg"). Scale both down together for short signals.
    n = min(nperseg, len(sig))
    if n < 8:
        return 0.0, 0.0
    ov = min(noverlap, n - 1)
    f, t, S = spectrogram(sig, fs=fs, nperseg=n, noverlap=ov, window='hamming')
    band = (f >= f_lo) & (f <= f_hi)
    if not np.any(band):
        return 0.0, 0.0
    band_energy = S[band, :].mean(axis=0)
    band_dB = 10 * np.log10(band_energy + 1e-20)
    peak = band_dB.max()
    above = np.where(band_dB - peak > thresh_db)[0]
    if len(above) == 0:
        return 0.0, peak
    return float(t[above[-1]]), peak


def band_main_and_islands(sig, fs, f_lo, f_hi, main_db=-60, faint_db=-75,
                          nperseg=1024, noverlap=768):
    """Contiguous main block end + faint 'islands' after it, one band.

    Calibrated against 51 manually reviewed labels on dEchorate room000000
    (36 good / 15 bad, 2026-08-20). What the eye flags as badly truncated is
    NOT a longer low band per se: it is faint patches of leftover noise that
    REAPPEAR after the main energy block ends (below -60 dB, so the old
    last-active-frame measure gave identical numbers for good and bad cases).
    main_end = end of the FIRST contiguous run above main_db.
    island_dur = total time above faint_db AFTER that block.
    """
    n = min(nperseg, len(sig))
    if n < 8:
        return 0.0, 0.0
    ov = min(noverlap, n - 1)
    f, t, S = spectrogram(sig, fs=fs, nperseg=n, noverlap=ov, window='hamming')
    band = (f >= f_lo) & (f <= f_hi)
    if not np.any(band):
        return 0.0, 0.0
    band_dB = 10 * np.log10(S[band, :].mean(axis=0) + 1e-20)
    band_dB -= band_dB.max()
    above = band_dB > main_db
    if not above.any():
        return 0.0, 0.0
    start = int(np.argmax(above))
    below = np.where(~above)[0]
    gaps = below[below > start]
    main_i = int(gaps[0] - 1) if len(gaps) else int(np.where(above)[0][-1])
    after = np.arange(len(t)) > main_i
    islands = after & (band_dB > faint_db)
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    return float(t[main_i]), float(islands.sum() * dt)


def classify_rir(name, load_raw_fn, wav_trunc_path):
    rir_raw, fs_raw = load_raw_fn()
    rir_trunc, fs_trunc = sf.read(wav_trunc_path)

    # v18+ truncates and saves every channel of a multichannel RIR (OpenAIR
    # b-format/ambisonic, up to 16ch) independently instead of picking one --
    # sf.read then returns (samples, channels). scipy.signal.spectrogram
    # defaults to the LAST axis, which here is the tiny channel axis, not
    # time -- nperseg=1024 against an axis of length 4 breaks immediately
    # ("noverlap must be less than nperseg"). Classify on channel 0 only,
    # same channel produce_dataset.py already uses for its comparison plot.
    if rir_raw.ndim == 2:
        rir_raw = rir_raw[:, 0]
    if rir_trunc.ndim == 2:
        rir_trunc = rir_trunc[:, 0]

    if fs_raw != fs_trunc:
        from scipy.signal import resample
        rir_raw = resample(rir_raw, int(len(rir_raw) * fs_trunc / fs_raw))
        fs_raw = fs_trunc

    low_raw, _ = band_active_duration(rir_raw, fs_raw, 20, 200)
    low_trunc, _ = band_active_duration(rir_trunc, fs_trunc, 20, 200)
    high_raw, _ = band_active_duration(rir_raw, fs_raw, 500, 8000)
    high_trunc, _ = band_active_duration(rir_trunc, fs_trunc, 500, 8000)

    low_reduction = 1 - (low_trunc / low_raw) if low_raw > 0 else 0.0
    high_reduction = 1 - (high_trunc / high_raw) if high_raw > 0 else 0.0
    low_high_ratio = low_trunc / high_trunc if high_trunc > 1e-6 else (
        float("inf") if low_trunc > 1e-6 else 1.0)

    # 2026-08-20: the old cat-3 rule (low_high_ratio > 1.6) collapsed on
    # dEchorate: manual review of 51 cat-3 cases (room000000) found 36 were
    # actually fine, with metric values IDENTICAL to bad ones (e.g. src1
    # mic11 good vs mic12 bad, both ratio 2.364 / low 0.277s). A longer low
    # band with a clean cut is just the room's own low-frequency decay,
    # correctly truncated. What the eye actually flags as bad is leftover
    # noise that REAPPEARS as faint patches after the main block ends --
    # below the -60 dB threshold, invisible to the ratio.
    #
    # New rule, fit jointly on those 51 dEchorate labels AND the 14 IKS
    # calibration cases below (48/51 + 14/14, vs 36/51 wrong before):
    #   cat 3 if faint islands after the main low block run longer than
    #   ISLAND_REL_MAX times the high band's contiguous block (relative,
    #   because on long RIRs like IKS stairway 0.4s of faint patches is
    #   normal decay, while on dEchorate's ~1s RIRs it is leftover noise),
    #   OR the contiguous low bar runs SOLID_RATIO_MIN times longer than the
    #   high band's while the low band was barely reduced from the raw
    #   (LOW_RED_MIN) -- the uncut-solid-bar case (IKS booth, dEchorate
    #   src1_mic31), distinct from a genuinely long low decay that WAS cut
    #   (dEchorate room000000 greens reduce low by 0.53-0.76).
    ISLAND_REL_MAX = 2.0
    SOLID_RATIO_MIN = 1.6
    LOW_RED_MIN = 0.45
    low_main, low_islands = band_main_and_islands(rir_trunc, fs_trunc, 20, 200)
    high_main, _ = band_main_and_islands(rir_trunc, fs_trunc, 500, 8000)
    solid_ratio = low_main / high_main if high_main > 1e-6 else (
        float("inf") if low_main > 1e-6 else 1.0)
    islands_rel = low_islands / high_main if high_main > 1e-6 else (
        float("inf") if low_islands > 1e-6 else 0.0)

    if low_reduction < 0.05 and high_reduction < 0.05:
        category = 1
    elif islands_rel > ISLAND_REL_MAX or (
            solid_ratio > SOLID_RATIO_MIN and low_reduction < LOW_RED_MIN):
        category = 3
    else:
        category = 2

    return {
        "name": name,
        "category": category,
        "low_reduction": round(low_reduction, 3),
        "high_reduction": round(high_reduction, 3),
        "low_high_ratio": round(low_high_ratio, 3) if low_high_ratio != float("inf") else "inf",
        "low_main_s": round(low_main, 3),
        "low_islands_s": round(low_islands, 3),
        "islands_rel": round(islands_rel, 3) if islands_rel != float("inf") else "inf",
        "solid_ratio": round(solid_ratio, 3) if solid_ratio != float("inf") else "inf",
        "low_raw_s": round(low_raw, 3),
        "low_trunc_s": round(low_trunc, 3),
        "high_raw_s": round(high_raw, 3),
        "high_trunc_s": round(high_trunc, 3),
    }


DATASET_OUT_NAME = {
    "openair": "OpenAIR", "iks": "IKS", "dechorate": "dEchorate",
    "mrtd_single": "MRTD_single",
    "arni": "Arni", "mrtd_covarianza": "MRTD_covarianza",
}


def _repeated_raw_loader(sweeps):
    # Same reference produce_dataset.py uses to compare against the clean
    # wav: the FIRST repetition of the pair, loaded the same way (mono,
    # channel 0 -- Arni's raw wavs are mono; see run_repeated).
    return load_rir_raw(sweeps[0], target_fs=48000, channel=0)


def run_dataset(dataset):
    out_name = DATASET_OUT_NAME[dataset]
    wav_dir = os.path.join(NB, "Assets", "Cleaned_Dataset", f"{out_name}_{PIPELINE_TAG}", "wavs")
    if dataset in ("arni", "mrtd_covarianza"):
        loaders = {name: (lambda sweeps=sweeps: _repeated_raw_loader(sweeps))
                   for name, sweeps in list_and_load_repeated(dataset)}
    else:
        loaders = {name: load_fn for name, load_fn in list_and_load_single(dataset)}

    results = []
    wavs = sorted(glob.glob(os.path.join(wav_dir, "*_truncated.wav")))
    print(f"{dataset}: {len(wavs)} cleaned RIRs to classify", flush=True)
    for i, wav_path in enumerate(wavs, 1):
        name = os.path.basename(wav_path).replace("_truncated.wav", "")
        load_fn = loaders.get(name)
        if load_fn is None:
            print(f"[{i}/{len(wavs)}] SKIP {name}: raw source not found", flush=True)
            continue
        try:
            r = classify_rir(name, load_fn, wav_path)
            results.append(r)
            print(f"[{i}/{len(wavs)}] cat={r['category']} ratio={r['low_high_ratio']} "
                  f"low_red={r['low_reduction']} high_red={r['high_reduction']} {name}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(wavs)}] ERR {name}: {e}", flush=True)
    return results, out_name


def build_qa_folders(dataset, results, out_name):
    # Single parent folder at this stage: Truncation_QA/<Name>/category_N_.../
    # {audio,plots} + classification.json -- all 3 categories, for review.
    # Narrowband_RIR_Dataset (the Zenodo release) is NOT touched here: that
    # is a separate step (promote_to_release), only after manual review
    # approves category_2 in the QA. Reads from the flat <Name>/plots and
    # <Name>/wavs that produce_dataset.py produced, and deletes them when done.
    base = os.path.join(NB, "Assets", "Cleaned_Dataset")
    flat_dir = os.path.join(base, f"{out_name}_{PIPELINE_TAG}")
    plots_dir = os.path.join(flat_dir, "plots")
    wavs_dir = os.path.join(flat_dir, "wavs")

    qa_dir = os.path.join(base, "Truncation_QA", f"{out_name}_{PIPELINE_TAG}")
    cat_dirs = {
        1: os.path.join(qa_dir, "category_1_no_truncation_needed"),
        2: os.path.join(qa_dir, "category_2_correctly_truncated"),
        3: os.path.join(qa_dir, "category_3_incomplete_low_frequency_truncation"),
    }

    import shutil
    manifest = {1: [], 2: [], 3: []}
    for cat, d in cat_dirs.items():
        os.makedirs(os.path.join(d, "plots"), exist_ok=True)
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)

    # Files actually copied into a category, so cleanup below only removes
    # THOSE from the flat dir -- a file whose classification errored (not in
    # `results`) is never deleted, it just stays in the flat dir for retry.
    # (2026-08-19: an unconditional rmtree of the whole flat dir here wiped
    # out an entire produced dataset -- 622 OpenAIR wavs+plots -- when every
    # single file failed classification with an unrelated scipy error.)
    copied = []
    for r in results:
        name, cat = r["name"], r["category"]
        png_src = os.path.join(plots_dir, name + "_2panel.png")
        wav_src = os.path.join(wavs_dir, name + "_truncated.wav")
        if os.path.exists(png_src):
            shutil.copy2(png_src, os.path.join(cat_dirs[cat], "plots", name + "_2panel.png"))
        if os.path.exists(wav_src):
            shutil.copy2(wav_src, os.path.join(cat_dirs[cat], "audio", name + "_truncated.wav"))
        manifest[cat].append(r)
        copied.append((png_src, wav_src))

    for cat, d in cat_dirs.items():
        with open(os.path.join(d, "metrics.txt"), "w") as f:
            f.write(f"Category {cat} -- {len(manifest[cat])} RIRs\n")
            f.write("low_reduction  = 1 - (active duration <200 Hz band, truncated / raw)\n")
            f.write("high_reduction = 1 - (active duration 500-8000 Hz band, truncated / raw)\n")
            f.write("low_high_ratio = low_trunc_s / high_trunc_s (>1.6x -> leftover low-freq bar)\n\n")
            for r in manifest[cat]:
                f.write(f"{r['name']}: ratio={r['low_high_ratio']} low_red={r['low_reduction']} "
                        f"high_red={r['high_reduction']} (low {r['low_raw_s']}s -> {r['low_trunc_s']}s, "
                        f"high {r['high_raw_s']}s -> {r['high_trunc_s']}s)\n")

    json.dump(results, open(os.path.join(qa_dir, "classification.json"), "w"), indent=2)

    # Carry over which code version produced this run (written by
    # produce_dataset.py into flat_dir) before flat_dir gets deleted below --
    # otherwise a QA folder has no way to tell v19 output from v17 output.
    version_src = os.path.join(flat_dir, "PIPELINE_VERSION.json")
    if os.path.exists(version_src):
        shutil.copy2(version_src, os.path.join(qa_dir, "PIPELINE_VERSION.json"))

    # Clean up ONLY the files that made it into a category above. Anything
    # left in wavs_dir/plots_dir (classification error, or a raw file the
    # classifier had no loader for) stays there untouched -- never guess it's
    # safe to delete just because the loop finished.
    for png_src, wav_src in copied:
        for f in (png_src, wav_src):
            try:
                os.remove(f)
            except OSError:
                pass

    remaining = 0
    for d in (plots_dir, wavs_dir):
        if os.path.isdir(d):
            remaining += len(os.listdir(d))
    if remaining:
        print(f"NOTE: {remaining} file(s) left in {flat_dir} (not classified, not deleted) -- "
              f"investigate before re-running.", flush=True)
    else:
        shutil.rmtree(plots_dir, ignore_errors=True)
        shutil.rmtree(wavs_dir, ignore_errors=True)
        try:
            os.rmdir(flat_dir)
        except OSError:
            pass

    return {cat: len(v) for cat, v in manifest.items()}


def promote_to_release(out_name):
    """Copies category_2 from the already-reviewed and approved QA to
    Narrowband_RIR_Dataset (the Zenodo release). Separate manual step, only
    run once the QA has been manually reviewed and approved."""
    import shutil
    base = os.path.join(NB, "Assets", "Cleaned_Dataset")
    qa_dir = os.path.join(base, "Truncation_QA", f"{out_name}_{PIPELINE_TAG}")
    src_audio = os.path.join(qa_dir, "category_2_correctly_truncated", "audio")
    src_plots = os.path.join(qa_dir, "category_2_correctly_truncated", "plots")
    if not os.path.isdir(src_audio):
        print(f"No QA found for {out_name} ({PIPELINE_TAG}). Run first: python3 classify_cleanliness.py <dataset>")
        return
    release_dir = os.path.join(base, "Narrowband_RIR_Dataset", f"{out_name}_Narrowband_Dataset")
    release_audio = os.path.join(release_dir, f"{out_name}_narrowband_truncated_rirs")
    release_plots = os.path.join(release_dir, f"{out_name}_before_after_spectrograms")
    os.makedirs(release_audio, exist_ok=True)
    os.makedirs(release_plots, exist_ok=True)
    n = 0
    for f in os.listdir(src_audio):
        shutil.copy2(os.path.join(src_audio, f), os.path.join(release_audio, f))
        n += 1
    for f in os.listdir(src_plots):
        shutil.copy2(os.path.join(src_plots, f), os.path.join(release_plots, f))
    # Provenance: which code version this promoted batch came from, so the
    # release folder itself says whether it holds v17 or v19 output.
    version_src = os.path.join(qa_dir, "PIPELINE_VERSION.json")
    if os.path.exists(version_src):
        shutil.copy2(version_src, os.path.join(release_dir, "PIPELINE_VERSION.json"))
    print(f"DONE. {n} RIRs promoted to release: Assets/Cleaned_Dataset/Narrowband_RIR_Dataset/{out_name}_Narrowband_Dataset/")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "promote":
        promote_to_release(sys.argv[2])
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] in DATASET_OUT_NAME:
        dataset = sys.argv[1]
        results, out_name = run_dataset(dataset)
        counts = build_qa_folders(dataset, results, out_name)
        print(f"\nDONE. {len(results)} classified: "
              f"cat1={counts[1]} cat2={counts[2]} cat3={counts[3]}")
        print(f"QA: Assets/Cleaned_Dataset/Truncation_QA/{out_name}_{PIPELINE_TAG}/")
        print(f"(not promoted to release; run 'python3 classify_cleanliness.py promote {out_name}' once approved)")
        sys.exit(0)

    # Calibration against the 14 manually validated IKS cases. Reads
    # from the QA category folders (the flat wavs/ used to live at
    # Cleaned_Dataset/IKS/wavs but that gets deleted once classification runs).
    iks_raw = {name: load_fn for name, load_fn in list_and_load_single("iks")}
    iks_qa = os.path.join(NB, "Assets", "Cleaned_Dataset", "Truncation_QA", f"IKS_{PIPELINE_TAG}")
    iks_wav_by_name = {}
    for cat_dir in glob.glob(os.path.join(iks_qa, "category_*", "audio")):
        for w in glob.glob(os.path.join(cat_dir, "*_truncated.wav")):
            iks_wav_by_name[os.path.basename(w).replace("_truncated.wav", "")] = w
    tests = [
        ("air_binaural_aula_carolina_0_1_1_90_3", 1),
        ("air_binaural_stairway_0_1_1_30", 2),
        ("air_binaural_booth_1_0_3", 3),
        ("air_binaural_office_0_0_1", 2),
        ("air_binaural_office_0_1_1", 2),
        ("air_binaural_office_1_0_1", 2),
        ("air_binaural_office_1_1_1", 2),
        ("air_binaural_booth_1_0_2", 3),
        ("air_binaural_stairway_1_1_2_75", 2),
        ("air_binaural_meeting_1_0_5", 2),
        ("air_binaural_booth_0_0_1", 3),
        ("air_binaural_booth_0_1_1", 3),
        ("air_binaural_booth_0_1_2", 3),
        ("air_binaural_booth_0_1_3", 3),
    ]
    for name, expected in tests:
        wav_path = iks_wav_by_name.get(name)
        if wav_path is None:
            print(f"[SKIP] {name}: not found in Truncation_QA/IKS_{PIPELINE_TAG}")
            continue
        result = classify_rir(name, iks_raw[name], wav_path)
        ok = "OK" if result["category"] == expected else "MISMATCH"
        print(f"[{ok}] {name}: expected={expected} got={result['category']} "
              f"ratio={result['low_high_ratio']}")
