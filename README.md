# Narrowband RIR Truncation

Frequency-dependent truncation of room impulse responses (RIRs). Instead of a single broadband truncation point, the truncation time is estimated per frequency band, and narrowband measurement artifacts (ringing concentrated in a few frequency bins) are detected and removed.

Developed in collaboration with Karolina Prawda and Nils Meyer-Kahlen, building on the `RIR-cropping` code base.

## Installation

```
pip install -r requirements.txt
```

## Current version: v19

New versions are added as new files with the next version number. Changes since v16:

- **v17**: onset trimming (silence before the direct sound) added to the single/Lundeby pipeline; the covariance pipeline already had it.
- **v18**: peak normalization on load removed (RIRs are analyzed at their recorded level); multichannel RIRs (stereo, B-format/ambisonic) are truncated per channel instead of keeping only channel 0.
- **v19**: dataset-specific ingestion split out to `dataset_loaders_v19.py` so the truncation core stays format-agnostic.

## Pipelines

### `truncate_repeated_v19.py` — repeated measurements (e.g. Arni)

Covariance method with unimodal regression. Main entry point:

```python
from truncate_repeated_v19 import truncate_covariance_band_expanded, load_rir_raw

rir_a, fs = load_rir_raw("meas_1.wav")
rir_b, _  = load_rir_raw("meas_2.wav")
rir_clean, mask, final_trunc, *_ = truncate_covariance_band_expanded([rir_a, rir_b], fs)
```

### `truncate_single_v19.py` — single RIRs (e.g. OpenAIR, IKS, dEchorate)

Lundeby-based per-bin truncation. The input is resampled to 48 kHz. Runs from the command line:

```
python truncate_single_v19.py input.wav --out results/
```

## Dataset production and quality control

Two scripts run the pipelines over whole datasets and sort the results:

- **`produce_dataset.py <openair|iks|dechorate|arni>`** — runs the right pipeline per dataset and writes truncated wavs plus before/after spectrogram plots. Expects the raw data under `Assets/Raw/<dataset>/` next to the script; adapt the paths in `list_and_load_single` / `list_and_load_repeated` (or `dataset_loaders_v19.py` for the .mat/.h5 formats) to point at your own copies.
- **`classify_cleanliness.py <dataset>`** — classifies each truncated RIR into three categories by comparing raw vs. truncated spectrograms: (1) no truncation needed, (2) correctly truncated, (3) incomplete low-frequency truncation. Category 3 is detected by faint noise "islands" that reappear in the low band (<200 Hz) after the main energy block ends, plus a solid-uncut-bar check; thresholds were calibrated against 65 manually reviewed cases across IKS and dEchorate. `classify_cleanliness.py promote <Dataset>` copies the approved category-2 set to the release folder.

## Method

The truncation curve over frequency is stabilised with a band-expanded outlier filter:

- per-bin truncation times compared against their 1/3-octave neighbourhood median
- artifact detector by ratio (1.25x over the local median)
- strip width of 150 Hz around the artifact band (validated by a parameter sweep over 107 confirmed-ringing RIRs)
- replacement of the artifact bins by the median of the neighbouring healthy bins

To detect which RIRs carry the artifact, the removal step is switched off and the code is run as a detector; the removal is switched back on for the parameter sweep.

## Test

```
python tests/test_smoke.py
```

Runs both pipelines on a synthetic RIR (no data download needed) and checks the outputs.

## Support modules

- `truncation_core_v19.py` — shared functions used by both pipelines: RIR loading (mono and multichannel), DC-offset removal, per-bin Lundeby, healthy-neighbour median replacement, truncation-excess ratio
- `dataset_loaders_v19.py` — per-format ingestion (IKS `.mat`, dEchorate `.h5`)
- `utils.py` — unimodal regression and time alignment

## License

GPL-3.0, inherited from the `RIR-cropping` code base.
