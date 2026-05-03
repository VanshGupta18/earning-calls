"""
Download and prepare the Earnings-22 dataset (Multimodal) using direct URLs.

Source: https://huggingface.co/datasets/anton-l/earnings22_baseline_5_gram/
This script downloads a subset of calls, saves the audio as WAV, 
and produces a compatible segments.parquet.
"""

import logging
import os
import tarfile
from pathlib import Path

import polars as pl
import requests
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
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://huggingface.co/datasets/anton-l/earnings22_baseline_5_gram/resolve/main/"
METADATA_URL = f"{BASE_URL}metadata.csv"

# ---------------------------------------------------------------------------
# Download Logic
# ---------------------------------------------------------------------------


def download_earnings22(output_dir: Path, max_calls: int = 5):
    """Download audio and transcripts from Earnings-22 via direct file access."""
    
    data_dir = output_dir / "raw" / "earnings22"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Metadata
    metadata_path = output_dir / "earnings22_metadata.csv"
    if not metadata_path.exists():
        logger.info("Downloading metadata.csv...")
        r = requests.get(METADATA_URL)
        metadata_path.write_bytes(r.content)
    
    df_meta = pl.read_csv(metadata_path)
    unique_calls = df_meta["source_id"].unique().to_list()[:max_calls]
    
    logger.info("Downloading %d unique calls...", len(unique_calls))
    
    segments_records = []
    
    for source_id in tqdm(unique_calls, desc="Calls"):
        tar_url = f"{BASE_URL}data/chunked/{source_id}.tar.gz"
        tar_path = data_dir / f"{source_id}.tar.gz"
        extract_path = data_dir / source_id
        
        # Download Tarball
        if not tar_path.exists():
            r = requests.get(tar_url, stream=True)
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract Tarball
        if not extract_path.exists():
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
        
        # Process segments for this call from metadata
        call_segments = df_meta.filter(pl.col("source_id") == source_id)
        
        for row in call_segments.iter_rows(named=True):
            # Segment audio is usually in source_id/segment_id.wav or similar
            # The 'file' column has the relative path inside the tarball
            rel_audio_path = row["file"]
            abs_audio_path = data_dir / rel_audio_path
            
            segments_records.append({
                "segment_id": f"{source_id}_{row['segment_id']}",
                "call_id": source_id,
                "speaker_role": "unknown",
                "speaker_name": "unknown", # Metadata doesn't have speaker names, just IDs
                "text": row["sentence"],
                "audio_path": str(abs_audio_path),
                "start_time": float(row["start_ts"]),
                "end_time": float(row["end_ts"]),
                "segment_type": "unknown"
            })

    # Save segments
    df_segments = pl.DataFrame(segments_records)
    output_path = output_dir / "processed" / "earnings22_segments.parquet"
    df_segments.write_parquet(output_path)
    logger.info("Saved %d segments to %s", len(df_segments), output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    download_earnings22(project_root / "data", max_calls=50)
