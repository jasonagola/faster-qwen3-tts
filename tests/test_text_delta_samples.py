from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples" / "text_delta_streaming"
EXPECTED_WAVS = [
    "custom_voice_200_text_delta.wav",
    "custom_voice_200_fulltext.wav",
    "voice_design_100_text_delta.wav",
    "voice_design_100_fulltext.wav",
    "voice_clone_xvec_100_text_delta.wav",
    "voice_clone_xvec_100_fulltext.wav",
]


def test_text_delta_sample_wavs_are_valid():
    for filename in EXPECTED_WAVS:
        path = SAMPLES_DIR / filename
        assert path.exists(), f"Missing sample WAV: {path}"
        audio, sample_rate = sf.read(path, always_2d=False)
        samples = np.asarray(audio)

        assert sample_rate > 0
        assert samples.size > 0
        assert np.isfinite(samples).all()
        assert samples.shape[0] / sample_rate > 0.25
        assert samples.shape[0] / sample_rate < 180.0
        assert np.max(np.abs(samples)) > 1e-4


def test_text_delta_sample_readme_links_exist():
    readme_text = (SAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    for filename in EXPECTED_WAVS:
        assert filename in readme_text
