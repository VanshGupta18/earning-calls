"""
Tests for src/features/audio_utils.py

These tests use synthetically generated audio (sine waves and silence)
so they work without any real audio files and without a GPU.

Run with: pytest tests/test_audio_utils.py -v
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torchaudio

from src.features.audio_utils import (
    TARGET_SR,
    load_audio,
    normalize_amplitude,
    save_segment,
    split_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sine_wave(
    freq: float = 440.0,
    sr: int = TARGET_SR,
    duration: float = 2.0,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Return a (1, N) mono sine-wave tensor."""
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    return (amplitude * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)


def save_tmp_wav(waveform: torch.Tensor, sr: int, suffix: str = ".wav") -> Path:
    """Save waveform to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    torchaudio.save(tmp.name, waveform, sr, format="wav")
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


class TestLoadAudio:
    def test_loads_and_resamples(self):
        """File saved at 8kHz should be resampled to 16kHz."""
        wave = make_sine_wave(sr=8000, duration=1.0)
        path = save_tmp_wav(wave, sr=8000)
        waveform, sr = load_audio(path, target_sr=TARGET_SR)
        assert sr == TARGET_SR
        assert waveform.shape[0] == 1          # mono
        assert waveform.shape[1] > 0

    def test_mono_output_for_stereo_input(self):
        """Stereo input should be mixed down to mono."""
        stereo = torch.stack([make_sine_wave()[0], make_sine_wave(freq=880.0)[0]])
        path = save_tmp_wav(stereo, sr=TARGET_SR)
        waveform, sr = load_audio(path)
        assert waveform.shape[0] == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")

    def test_target_sr_preserved(self):
        wave = make_sine_wave(sr=TARGET_SR, duration=1.0)
        path = save_tmp_wav(wave, sr=TARGET_SR)
        _, sr = load_audio(path, target_sr=TARGET_SR)
        assert sr == TARGET_SR


# ---------------------------------------------------------------------------
# normalize_amplitude
# ---------------------------------------------------------------------------


class TestNormalizeAmplitude:
    def test_peak_is_one(self):
        wave = make_sine_wave(amplitude=0.3)
        normalized = normalize_amplitude(wave)
        assert normalized.abs().max().item() == pytest.approx(1.0, abs=1e-5)

    def test_silent_wave_unchanged(self):
        silence = torch.zeros(1, 1000)
        out = normalize_amplitude(silence)
        assert torch.all(out == 0)

    def test_already_normalized(self):
        wave = make_sine_wave(amplitude=1.0)
        # Peak might not be exactly 1.0 due to discrete sampling, but should be close
        normalized = normalize_amplitude(wave)
        assert normalized.abs().max().item() == pytest.approx(1.0, abs=1e-3)

    def test_shape_preserved(self):
        wave = torch.randn(1, 8000)
        out = normalize_amplitude(wave)
        assert out.shape == wave.shape


# ---------------------------------------------------------------------------
# split_audio
# ---------------------------------------------------------------------------


class TestSplitAudio:
    def test_basic_split(self):
        wave = make_sine_wave(sr=TARGET_SR, duration=4.0)
        sliced = split_audio(wave, TARGET_SR, start_time=1.0, end_time=3.0)
        expected_samples = int(2.0 * TARGET_SR)
        assert abs(sliced.shape[-1] - expected_samples) <= 1  # ±1 for rounding

    def test_full_duration(self):
        wave = make_sine_wave(sr=TARGET_SR, duration=2.0)
        sliced = split_audio(wave, TARGET_SR, start_time=0.0, end_time=2.0)
        assert sliced.shape[-1] == wave.shape[-1]

    def test_start_equals_end_raises(self):
        wave = make_sine_wave(duration=2.0)
        with pytest.raises(ValueError, match="less than end_time"):
            split_audio(wave, TARGET_SR, start_time=1.0, end_time=1.0)

    def test_start_after_end_raises(self):
        wave = make_sine_wave(duration=2.0)
        with pytest.raises(ValueError, match="less than end_time"):
            split_audio(wave, TARGET_SR, start_time=2.0, end_time=1.0)

    def test_negative_start_raises(self):
        wave = make_sine_wave(duration=2.0)
        with pytest.raises(ValueError, match="negative"):
            split_audio(wave, TARGET_SR, start_time=-0.1, end_time=1.0)

    def test_end_beyond_duration_raises(self):
        wave = make_sine_wave(duration=2.0)
        with pytest.raises(ValueError, match="exceeds audio duration"):
            split_audio(wave, TARGET_SR, start_time=0.0, end_time=3.0)


# ---------------------------------------------------------------------------
# save_segment
# ---------------------------------------------------------------------------


class TestSaveSegment:
    def test_saves_and_reloads(self):
        wave = make_sine_wave(sr=TARGET_SR, duration=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "segment.wav"
            save_segment(wave, TARGET_SR, out_path)
            assert out_path.exists()
            loaded, sr = torchaudio.load(str(out_path))
            assert sr == TARGET_SR
            assert loaded.shape[0] == 1

    def test_creates_parent_dirs(self):
        wave = make_sine_wave(duration=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "nested" / "deep" / "seg.wav"
            save_segment(wave, TARGET_SR, out_path)
            assert out_path.exists()

    def test_resamples_before_saving(self):
        """Input at 8kHz should be saved at 16kHz."""
        wave = make_sine_wave(sr=8000, duration=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "seg.wav"
            save_segment(wave, sr=8000, output_path=out_path)
            _, sr = torchaudio.load(str(out_path))
            assert sr == TARGET_SR

    def test_stereo_saved_as_mono(self):
        stereo = torch.stack([make_sine_wave()[0], make_sine_wave(freq=880.0)[0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "seg.wav"
            save_segment(stereo, TARGET_SR, out_path)
            loaded, _ = torchaudio.load(str(out_path))
            assert loaded.shape[0] == 1
