"""
Audio preprocessing utility functions for the Multimodal Earnings Call Intelligence System.

Functions:
    load_audio(path, target_sr)     — load and resample an audio file
    normalize_amplitude(waveform)   — peak-normalize waveform to [-1, 1]
    split_audio(waveform, sr, t0, t1) — slice a waveform by time range
    save_segment(waveform, sr, path) — save a waveform as 16kHz mono WAV
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

# Target sample rate for all audio in this project
TARGET_SR: int = 16_000


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SR,
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and resample it to `target_sr` Hz, converting to mono.

    Args:
        path: Path to the audio file (WAV, MP3, FLAC, etc.).
        target_sr: Desired sample rate in Hz. Default is 16000 Hz.

    Returns:
        (waveform, sample_rate) where waveform has shape (1, num_samples)
        and sample_rate == target_sr.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If torchaudio cannot load the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))  # shape: (channels, samples)

    # Mix down to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        logger.debug("Resampled %s from %d Hz to %d Hz.", path.name, sr, target_sr)

    return waveform, target_sr


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


def normalize_amplitude(waveform: torch.Tensor) -> torch.Tensor:
    """
    Peak-normalize a waveform tensor to the range [-1, 1].

    If the waveform is silent (all zeros), it is returned unchanged.

    Args:
        waveform: Audio tensor of shape (1, num_samples) or (num_samples,).

    Returns:
        Normalized tensor with the same shape.
    """
    peak = waveform.abs().max()
    if peak == 0:
        logger.warning("Silent waveform encountered — returning zeros unchanged.")
        return waveform
    return waveform / peak


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_audio(
    waveform: torch.Tensor,
    sr: int,
    start_time: float,
    end_time: float,
) -> torch.Tensor:
    """
    Extract a time slice from a waveform.

    Args:
        waveform: Audio tensor of shape (1, num_samples).
        sr: Sample rate in Hz.
        start_time: Start time in seconds (inclusive).
        end_time: End time in seconds (exclusive).

    Returns:
        Sliced waveform tensor of shape (1, num_slice_samples).

    Raises:
        ValueError: If start_time >= end_time or if times are out of range.
    """
    if start_time >= end_time:
        raise ValueError(
            f"start_time ({start_time}s) must be less than end_time ({end_time}s)."
        )

    total_duration = waveform.shape[-1] / sr
    if start_time < 0:
        raise ValueError(f"start_time ({start_time}s) cannot be negative.")
    if end_time > total_duration + 1e-6:
        raise ValueError(
            f"end_time ({end_time}s) exceeds audio duration ({total_duration:.2f}s)."
        )

    start_sample = int(start_time * sr)
    end_sample = min(int(end_time * sr), waveform.shape[-1])
    return waveform[..., start_sample:end_sample]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_segment(
    waveform: torch.Tensor,
    sr: int,
    output_path: Union[str, Path],
) -> Path:
    """
    Save a waveform tensor as a 16kHz mono WAV file.

    If `sr` != 16000, the waveform will be resampled before saving.
    If the waveform is multi-channel, it will be mixed to mono.

    Args:
        waveform: Audio tensor of shape (1, num_samples) or (num_samples,).
        sr: Current sample rate of the waveform.
        output_path: Destination path (will create parent directories).

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 2D: (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        sr = TARGET_SR

    torchaudio.save(str(output_path), waveform, sr, format="wav")
    logger.debug("Saved segment to %s (%d samples @ %d Hz).",
                 output_path, waveform.shape[-1], sr)
    return output_path
