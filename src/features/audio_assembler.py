"""
Unified audio feature assembler for the Multimodal Earnings Call Intelligence System.

Merges prosody, openSMILE, and wav2vec2 features into a single 
audio_features.parquet matching Contract C. Filters out segments
flagged as low quality.

Contract C Schema:
    - segment_id (PK)
    - pitch_mean, pitch_variance
    - energy_variance
    - speech_rate
    - voice_stability
    - [88 openSMILE features]
    - [768 wav2vec2 features]
"""

import logging
from pathlib import Path

import polars as pl

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
# Assembler
# ---------------------------------------------------------------------------


def assemble_audio_features(data_dir: Path, output_path: Path):
    """Join all audio feature Parquet files and save the result."""
    
    paths = {
        "prosody": data_dir / "audio_prosody.parquet",
        "opensmile": data_dir / "audio_opensmile.parquet",
        "wav2vec2": data_dir / "audio_wav2vec2.parquet",
        "quality": data_dir / "audio_quality.parquet",
    }

    # Check existence
    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        logger.error("Missing feature tables: %s", missing)
        return

    logger.info("Loading feature tables...")
    df_pros = pl.read_parquet(paths["prosody"])
    df_smile = pl.read_parquet(paths["opensmile"])
    df_wav = pl.read_parquet(paths["wav2vec2"])
    df_qual = pl.read_parquet(paths["quality"])

    # 1. Filter by quality
    usable_ids = df_qual.filter(pl.col("is_usable") == True).select("segment_id")
    logger.info("Usable segments: %d / %d", len(usable_ids), len(df_qual))

    # 2. Sequential Joins
    logger.info("Merging tables on segment_id...")
    df_merged = usable_ids.join(df_pros, on="segment_id", how="inner")
    df_merged = df_merged.join(df_smile, on="segment_id", how="inner")
    df_merged = df_merged.join(df_wav, on="segment_id", how="inner")

    # Handle NaNs
    n_nan = df_merged.null_count().sum().to_series()[0]
    if n_nan > 0:
        logger.warning("Found %d null values in merged table. Filling with 0.0.", n_nan)
        df_merged = df_merged.fill_null(0.0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.write_parquet(output_path)
    logger.info("Saved unified audio features (%d rows) to: %s", len(df_merged), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "processed"
    output_path = data_dir / "audio_features.parquet"
    
    assemble_audio_features(data_dir, output_path)


if __name__ == "__main__":
    main()
