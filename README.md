# Narrowband RIR Truncation

Frequency-dependent truncation of room impulse responses (RIRs). Instead of a single broadband truncation point, the truncation time is estimated per frequency band, and narrowband measurement artifacts (ringing concentrated in a few frequency bins) are detected and removed.

Developed in collaboration with Karolina Prawda and Nils Meyer-Kahlen, building on the `RIR-cropping` code base.

## Installation

```
pip install -r requirements.txt
```

## Pipelines

New versions are added as new files with the next version number.

### `truncate_repeated_v16.py` — repeated measurements (e.g. Arni)

Covariance method with unimodal regression. Main entry point:

```python
from truncate_repeated_v16 import truncate_covariance_band_expanded, load_rir_raw

rir_a, fs = load_rir_raw("meas_1.wav")
rir_b, _  = load_rir_raw("meas_2.wav")
rir_clean, mask, final_trunc, *_ = truncate_covariance_band_expanded([rir_a, rir_b], fs)
```

### `truncate_single_v16.py` — single RIRs (e.g. OpenAIR)

Lundeby-based per-bin truncation. The input is resampled to 48 kHz and
peak-normalised before processing. Runs from the command line:

```
python truncate_single_v16.py input.wav --out results/
```

## Method

The truncation curve over frequency is stabilised with a band-expanded outlier filter:

- per-bin truncation times compared against their 1/3-octave neighbourhood median
- artifact detector by ratio (1.25x over the local median)
- strip width of 100 Hz around the artifact band
- replacement of the artifact bins by the median of the neighbouring healthy bins

To detect which RIRs carry the artifact, the removal step is switched off and the code is run as a detector; the removal is switched back on for the parameter sweep.

## Test

```
python tests/test_smoke.py
```

Runs both pipelines on a synthetic RIR (no data download needed) and checks the outputs.

## Support modules

- `truncation_core_v16.py` — shared functions used by both pipelines: RIR loading, per-bin Lundeby, healthy-neighbour median replacement, truncation-excess ratio
- `utils.py` — unimodal regression and time alignment

## License

GPL-3.0, inherited from the `RIR-cropping` code base.
