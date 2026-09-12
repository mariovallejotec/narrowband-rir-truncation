"""Build the deliberately mis-truncated (half truncation time) version of the
website example RIR (IKS air_binaural_stairway_1_1_1_150), so the site can show
the SAME RIR as raw / correctly truncated / incorrectly truncated side by side
(Karolina's request, 11-Sep-2026).

Uses exactly the test pipeline from build_paired_comparison_test.py:
half_truncate() (same per-bin estimate, mask at half the truncation time),
normalize_pair_shared() (one gain measured on the raw), and the global trim
of the v1.3/v1.4 test (0.126395 = -17.97 dB). The raw output is checked
against the existing trial file interest03_..._A_raw.wav to confirm the
pipeline is reproduced sample-exactly before the half-truncated file is written.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_paired_comparison_test import half_truncate, normalize_pair_shared, load_raw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ROOT / "Assets/ListeningTest_PairedComparison/trials"
BASE = "air_binaural_stairway_1_1_1_150"
GLOBAL_TRIM = 0.126395
OUT_NAME = "iks_stairway_halftruncated.wav"

raw_audio, fs = load_raw("IKS", BASE)
half_audio = half_truncate(raw_audio, fs)
raw_out, half_out = normalize_pair_shared(raw_audio, fs, half_audio, fs)
raw_out, half_out = raw_out * GLOBAL_TRIM, half_out * GLOBAL_TRIM

ref, ref_fs = sf.read(TRIALS / f"interest03_IKS_{BASE}_A_raw.wav")
assert ref_fs == fs and len(ref) == len(raw_out), (ref_fs, fs, len(ref), len(raw_out))
max_diff = float(np.max(np.abs(ref - raw_out)))
print(f"raw reproduced vs trial file: max |diff| = {max_diff:.2e} (24-bit LSB = {2**-23:.2e})")
assert max_diff < 2 ** -22, "raw does not match the trial file; pipeline not reproduced"

peak_db = 20 * np.log10(np.max(np.abs(half_out)))
print(f"half-truncated: peak {peak_db:.2f} dBFS, {len(half_out)/fs:.2f} s")
for target in (ROOT / "Sitio Web/assets/audio", ROOT / "narrowband-rir-truncation/docs/assets/audio"):
    sf.write(target / OUT_NAME, half_out, fs, subtype="PCM_24")
    print("written", target / OUT_NAME)
