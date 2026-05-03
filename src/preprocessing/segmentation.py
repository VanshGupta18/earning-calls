"""
Structural segmentation for earnings call transcripts.

Labels each segment as one of:
    - prepared_remarks
    - analyst_question
    - management_answer
    - operator_transition

Combines speaker role classification with positional heuristics
to produce the final labelled segments.parquet.

Phase 2 - Task 2.D.5
"""

import logging
import re
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
# Speaker classification heuristics
# ---------------------------------------------------------------------------

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
ANALYST_KEYWORDS = [
    "analyst", "research", "capital", "securities",
    "partners", "advisors", "bank", "morgan",
    "goldman", "barclays", "citi", "jpmorgan",
    "credit suisse", "ubs", "wells fargo",
    "bofa", "merrill", "deutsche",
]

# Q&A transition markers
QA_MARKERS = [
    "question-and-answer",
    "question and answer",
    "q&a session",
    "q & a",
    "open the line",
    "open it up for questions",
    "take your questions",
    "first question",
]


# ---------------------------------------------------------------------------
# Speaker role classification
# ---------------------------------------------------------------------------


def classify_speaker_role(speaker_name: str, segment_index: int) -> str:
    """Classify a speaker into: ceo, cfo, executive, analyst, operator, other."""
    if not speaker_name:
        return "other"

    name_lower = speaker_name.lower().strip()

    # Operator detection
    if any(kw in name_lower for kw in OPERATOR_KEYWORDS):
        return "operator"

    # Executive detection (check specific titles first)
    for title in EXECUTIVE_TITLES:
        if title in name_lower:
            if "ceo" in name_lower or "chief executive" in name_lower or "president" in name_lower:
                return "ceo"
            if "cfo" in name_lower or "chief financial" in name_lower:
                return "cfo"
            return "executive"

    # Analyst detection
    if any(kw in name_lower for kw in ANALYST_KEYWORDS):
        return "analyst"

    # Position-based heuristic: first few speakers are typically executives
    if segment_index < 3:
        return "executive"

    return "other"


# ---------------------------------------------------------------------------
# Segment type classification
# ---------------------------------------------------------------------------


def detect_qa_transition(text: str) -> bool:
    """Check if a text segment contains Q&A transition markers."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in QA_MARKERS)


def classify_segment_type(
    speaker_role: str,
    text: str,
    qa_started: bool,
) -> tuple[str, bool]:
    """
    Classify a segment as one of:
        prepared_remarks, analyst_question, management_answer, operator_transition.

    Returns (segment_type, qa_started_flag).
    """
    # Check for Q&A transition
    if not qa_started and detect_qa_transition(text):
        qa_started = True

    if speaker_role == "operator":
        return "operator_transition", qa_started

    if not qa_started:
        # Before Q&A, everything is prepared remarks
        if speaker_role == "analyst":
            # First analyst appearance triggers Q&A
            return "analyst_question", True
        return "prepared_remarks", False

    # Inside Q&A
    if speaker_role == "analyst":
        return "analyst_question", True

    if speaker_role in ("ceo", "cfo", "executive", "other"):
        return "management_answer", True

    return "management_answer", True


# ---------------------------------------------------------------------------
# Question detection (text-based)
# ---------------------------------------------------------------------------


def is_question(text: str) -> bool:
    """Heuristic to detect if text contains a question."""
    # Direct question mark
    if "?" in text:
        return True
    # Question-like starts
    question_starters = [
        "can you", "could you", "would you", "will you",
        "how ", "what ", "when ", "where ", "why ", "who ",
        "is there", "are there", "do you", "does ",
        "have you", "has ", "should ",
    ]
    text_lower = text.lower().strip()
    return any(text_lower.startswith(qs) for qs in question_starters)


# ---------------------------------------------------------------------------
# Full segmentation pipeline
# ---------------------------------------------------------------------------


def apply_segmentation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply speaker role and segment type classification to a segments DataFrame.

    Expects columns: call_id, segment_id, speaker_name (or speaker_role), text.
    Adds/updates: speaker_role, segment_type.
    """
    # Process call-by-call to maintain Q&A state
    all_rows = []
    call_ids = df["call_id"].unique().sort().to_list()

    for call_id in call_ids:
        call_df = df.filter(pl.col("call_id") == call_id).sort("segment_id")
        qa_started = False

        for i, row in enumerate(call_df.iter_rows(named=True)):
            speaker_name = row.get("speaker_name", "")
            text = row.get("text", "")

            # Classify speaker
            speaker_role = classify_speaker_role(speaker_name, i)

            # Classify segment type
            segment_type, qa_started = classify_segment_type(
                speaker_role, text, qa_started
            )

            # Build output row
            new_row = dict(row)
            new_row["speaker_role"] = speaker_role
            new_row["segment_type"] = segment_type
            new_row["is_question"] = is_question(text)
            all_rows.append(new_row)

    result = pl.DataFrame(all_rows)

    # Log summary
    type_counts = result.group_by("segment_type").len().sort("len", descending=True)
    logger.info("Segment type distribution:")
    for row in type_counts.iter_rows():
        logger.info("  %-25s %d", row[0], row[1])

    role_counts = result.group_by("speaker_role").len().sort("len", descending=True)
    logger.info("Speaker role distribution:")
    for row in role_counts.iter_rows():
        logger.info("  %-25s %d", row[0], row[1])

    qa_calls = result.filter(pl.col("segment_type") == "analyst_question")["call_id"].n_unique()
    logger.info("Calls with detected Q&A: %d / %d", qa_calls, len(call_ids))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    segments_path = project_root / "data" / "processed" / "segments.parquet"
    output_path = project_root / "data" / "processed" / "segments_labelled.parquet"

    if not segments_path.exists():
        logger.error("segments.parquet not found at %s", segments_path)
        return

    df = pl.read_parquet(segments_path)
    logger.info("Loaded %d segments from %s", len(df), segments_path)

    result = apply_segmentation(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    logger.info("Saved labelled segments (%d rows) to: %s", len(result), output_path)


if __name__ == "__main__":
    main()
