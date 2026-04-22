#!/usr/bin/env python3
"""
Download and process S&P 500 earnings call transcripts from HuggingFace.

Downloads the Bose345/sp500_earnings_transcripts dataset (or a filtered subset)
and converts it to the project's segments.parquet format (Contract A).

Usage:
    python scripts/download_transcripts.py                          # Full dataset
    python scripts/download_transcripts.py --years 2023 2024        # Filter by year
    python scripts/download_transcripts.py --sectors Technology      # Filter by sector
    python scripts/download_transcripts.py --tickers AAPL MSFT GOOGL # Filter by ticker
    python scripts/download_transcripts.py --max-calls 200          # Limit total calls
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import polars as pl
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "Bose345/sp500_earnings_transcripts"

# S&P 500 Tech sector tickers (as of 2024) — used for default MVP filtering.
SP500_TECH_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN",
    "ADBE", "IBM", "INTC", "INTU", "TXN", "QCOM", "AMAT", "NOW", "PANW",
    "LRCX", "ADI", "SNPS", "KLAC", "CDNS", "MCHP", "FTNT", "ROP", "NXPI",
    "MPWR", "ON", "FSLR", "KEYS", "ANSS", "HPE", "HPQ", "ZBRA", "EPAM",
    "VRSN", "JNPR", "GEN", "TRMB", "SWKS", "PTC", "TYL", "TER", "AKAM",
    "FFIV", "CTSH", "IT", "ENPH", "WDC",
    # Mega-cap tech often classified under Communication Services / Consumer Disc.
    "GOOGL", "GOOG", "META", "AMZN", "TSLA", "NFLX",
]

# Speaker classification heuristics
OPERATOR_KEYWORDS = ["operator", "conference call", "moderator"]
EXECUTIVE_TITLES = [
    "ceo", "chief executive", "president",
    "cfo", "chief financial", "treasurer",
    "coo", "chief operating",
    "cto", "chief technology",
    "chairman", "vice president", "vp", "svp", "evp",
    "director", "head of", "general manager", "controller",
    "ir ", "investor relations",
]
ANALYST_KEYWORDS = ["analyst", "research", "capital", "securities",
                     "partners", "advisors", "bank", "morgan",
                     "goldman", "barclays", "citi", "jpmorgan",
                     "credit suisse", "ubs", "wells fargo",
                     "bofa", "merrill", "deutsche"]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Speaker classification
# ---------------------------------------------------------------------------


def classify_speaker_role(speaker_name: str, segment_index: int, total_segments: int) -> str:
    """Classify a speaker into one of: ceo, cfo, analyst, operator, other."""
    if not speaker_name:
        return "other"

    name_lower = speaker_name.lower().strip()

    # Operator detection
    if any(kw in name_lower for kw in OPERATOR_KEYWORDS):
        return "operator"

    # Executive detection
    for title in EXECUTIVE_TITLES:
        if title in name_lower:
            if "ceo" in name_lower or "chief executive" in name_lower or "president" in name_lower:
                return "ceo"
            if "cfo" in name_lower or "chief financial" in name_lower or "treasurer" in name_lower:
                return "cfo"
            return "executive"

    # Analyst detection
    if any(kw in name_lower for kw in ANALYST_KEYWORDS):
        return "analyst"

    # Position-based heuristic: first few speakers are often executives
    if segment_index < 3:
        return "executive"

    return "other"


def classify_segment_type(
    speaker_role: str,
    segment_index: int,
    total_segments: int,
    qa_started: bool,
) -> tuple[str, bool]:
    """
    Classify a segment as one of:
        prepared_remarks, analyst_question, management_answer, operator_transition.

    Returns (segment_type, qa_started_flag).
    """
    if speaker_role == "operator":
        # Check if this is the Q&A transition
        return "operator_transition", qa_started

    if not qa_started:
        # Before Q&A starts, everything is prepared remarks
        if speaker_role == "analyst":
            # First analyst question triggers Q&A
            return "analyst_question", True
        return "prepared_remarks", False

    # Inside Q&A
    if speaker_role == "analyst":
        return "analyst_question", True
    if speaker_role in ("ceo", "cfo", "executive"):
        return "management_answer", True

    return "management_answer", True  # Default to management answer in Q&A


# ---------------------------------------------------------------------------
# Transcript processing
# ---------------------------------------------------------------------------


def process_transcript(record: dict) -> list[dict]:
    """Convert one HuggingFace dataset record to a list of segment dicts matching Contract A."""
    symbol = record.get("symbol", "UNK")
    year = record.get("year", 0)
    quarter = record.get("quarter", 0)
    call_id = f"{symbol}_{year}Q{quarter}"

    structured = record.get("structured_content")
    if not structured:
        return []

    segments = []
    qa_started = False
    total = len(structured)

    for i, turn in enumerate(structured):
        speaker = turn.get("speaker", "") or ""
        text = turn.get("text", "") or ""

        # Skip empty segments
        if not text.strip():
            continue

        speaker_role = classify_speaker_role(speaker, i, total)
        segment_type, qa_started = classify_segment_type(speaker_role, i, total, qa_started)

        segments.append({
            "call_id": call_id,
            "segment_id": f"{call_id}_seg_{i:04d}",
            "speaker_role": speaker_role,
            "speaker_name": speaker.strip(),
            "segment_type": segment_type,
            "text": text.strip(),
            "start_time": None,  # No audio timestamps in text-only MVP
            "end_time": None,
            "audio_path": None,
        })

    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download S&P 500 earnings call transcripts and convert to segments.parquet",
    )
    parser.add_argument(
        "--years", nargs="*", type=int, default=None,
        help="Filter to specific years (e.g., --years 2023 2024). Default: all years.",
    )
    parser.add_argument(
        "--tickers", nargs="*", type=str, default=None,
        help="Filter to specific tickers (e.g., --tickers AAPL MSFT). Default: all.",
    )
    parser.add_argument(
        "--tech-only", action="store_true",
        help="Filter to S&P 500 Tech sector tickers only.",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Maximum number of calls to process. Default: no limit.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for segments.parquet. Default: data/processed",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Directory to cache the raw HuggingFace dataset. Default: data/raw",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    raw_dir = project_root / args.raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Download dataset
    # -----------------------------------------------------------------------
    logger.info("Downloading dataset: %s", DATASET_ID)
    logger.info("This may take a few minutes on first run (~1.8 GB)...")

    ds = load_dataset(DATASET_ID, cache_dir=str(raw_dir / "hf_cache"))
    train = ds["train"]
    logger.info("Dataset loaded: %d total records", len(train))

    # -----------------------------------------------------------------------
    # Step 2: Apply filters
    # -----------------------------------------------------------------------
    tickers_filter = None
    if args.tech_only:
        tickers_filter = set(SP500_TECH_TICKERS)
        logger.info("Filtering to %d Tech sector tickers", len(tickers_filter))
    elif args.tickers:
        tickers_filter = set(t.upper() for t in args.tickers)
        logger.info("Filtering to %d specified tickers: %s", len(tickers_filter), tickers_filter)

    years_filter = set(args.years) if args.years else None
    if years_filter:
        logger.info("Filtering to years: %s", years_filter)

    # Filter the dataset
    def should_include(record):
        if tickers_filter and record.get("symbol") not in tickers_filter:
            return False
        if years_filter and record.get("year") not in years_filter:
            return False
        return True

    filtered = train.filter(should_include, desc="Filtering transcripts")
    logger.info("After filtering: %d records", len(filtered))

    if args.max_calls and len(filtered) > args.max_calls:
        filtered = filtered.select(range(args.max_calls))
        logger.info("Capped at %d calls", args.max_calls)

    # -----------------------------------------------------------------------
    # Step 3: Process transcripts into segments
    # -----------------------------------------------------------------------
    all_segments = []
    skipped = 0

    for record in tqdm(filtered, desc="Processing transcripts"):
        segments = process_transcript(record)
        if segments:
            all_segments.extend(segments)
        else:
            skipped += 1

    logger.info(
        "Processed %d segments from %d calls (%d skipped due to missing content)",
        len(all_segments), len(filtered) - skipped, skipped,
    )

    if not all_segments:
        logger.error("No segments produced. Check filters and dataset.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 4: Save as Parquet
    # -----------------------------------------------------------------------
    df = pl.DataFrame(all_segments)

    # Validate schema matches Contract A
    expected_columns = [
        "call_id", "segment_id", "speaker_role", "speaker_name",
        "segment_type", "text", "start_time", "end_time", "audio_path",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

    output_path = output_dir / "segments.parquet"
    df.write_parquet(output_path)
    logger.info("Saved segments to: %s", output_path)

    # -----------------------------------------------------------------------
    # Step 5: Summary statistics
    # -----------------------------------------------------------------------
    n_calls = df["call_id"].n_unique()
    n_segments = len(df)
    n_tickers = df["call_id"].str.extract(r"^([A-Z]+)_", 1).n_unique()

    segment_type_counts = df.group_by("segment_type").len().sort("len", descending=True)
    speaker_role_counts = df.group_by("speaker_role").len().sort("len", descending=True)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total calls:    %d", n_calls)
    logger.info("Total segments: %d", n_segments)
    logger.info("Unique tickers: %d", n_tickers)
    logger.info("")
    logger.info("Segment types:")
    for row in segment_type_counts.iter_rows():
        logger.info("  %-25s %d", row[0], row[1])
    logger.info("")
    logger.info("Speaker roles:")
    for row in speaker_role_counts.iter_rows():
        logger.info("  %-25s %d", row[0], row[1])

    # Save a manifest CSV for downstream use
    manifest = (
        df.group_by("call_id")
        .agg([
            pl.col("segment_id").count().alias("n_segments"),
            pl.col("text").str.len_chars().sum().alias("total_chars"),
        ])
        .sort("call_id")
    )
    manifest_path = output_dir / "call_manifest.csv"
    manifest.write_csv(manifest_path)
    logger.info("Saved call manifest to: %s", manifest_path)


if __name__ == "__main__":
    main()
