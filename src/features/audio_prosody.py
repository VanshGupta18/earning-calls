"""
Prosody feature extraction for earnings call segments.

Computes:
    - pitch_mean, pitch_variance (F0)
    - energy_variance (RMS)
    - speech_rate (syllables/sec)
    - pause_duration_total
    - voice_stability (jitter proxy)

Outputs: data/processed/audio_prosody.parquet
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prosody Extractor
# ---------------------------------------------------------------------------


class ProsodyExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_segment_features(self, audio_path: str) -> dict:
        """Extract prosodic features from a single audio segment."""
        try:
            if not Path(audio_path).exists():
                return {}

            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(y) == 0:
                return {}

            # 1. Pitch (F0) using probabilistic Yin
            f0, _, _ = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz("C2"), 
                fmax=librosa.note_to_hz("C7")
            )
            f0_valid = f0[~np.isnan(f0)]
            
            pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
            pitch_variance = float(np.var(f0_valid)) if len(f0_valid) > 0 else 0.0

            # 2. Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            energy_variance = float(np.var(rms)) if len(rms) > 0 else 0.0

            # 3. Speech Rate (Approximated via onset strength)
            # Higher onset frequency roughly correlates with speech rate
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
            
            duration = len(y) / sr
            speech_rate = len(peaks) / duration if duration > 0 else 0.0

            # 4. Voice Stability (Jitter-like variance of F0 period)
            voice_stability = 0.0
            if len(f0_valid) > 1:
                # Difference between consecutive F0 values
                jitter = np.abs(np.diff(f0_valid))
                voice_stability = float(np.mean(jitter))

            return {
                "pitch_mean": pitch_mean,
                "pitch_variance": pitch_variance,
                "energy_variance": energy_variance,
                "speech_rate": speech_rate,
                "voice_stability": voice_stability,
            }
        except Exception as e:
            logger.warning("Failed to process %s: %s", audio_path, e)
            return {}

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run extraction on a Polars DataFrame of segments."""
        logger.info("Extracting prosody features from %d segments...", len(segments))
        
        results = []
        for audio_path in tqdm(segments["audio_path"].to_list(), desc="Prosody"):
            results.append(self.extract_segment_features(audio_path))

        return segments.select("segment_id").with_columns(
            pl.from_dicts(results)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    segments_path = project_root / "data" / "processed" / "earnings22_segments.parquet"
    output_path = project_root / "data" / "processed" / "audio_prosody.parquet"

    if not segments_path.exists():
        logger.error("earnings22_segments.parquet not found. Run download_earnings22.py first.")
        return

    df_segments = pl.read_parquet(segments_path)
    
    # For a quick test, we can limit the number of segments
    # df_segments = df_segments.head(100)

    extractor = ProsodyExtractor()
    df_features = extractor.extract(df_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved prosody features to: %s", output_path)


if __name__ == "__main__":
    main()
