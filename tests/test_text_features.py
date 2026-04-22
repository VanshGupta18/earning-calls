"""
Tests for text feature extraction modules.

Run with: pytest tests/test_text_features.py -v
"""

import pytest
import polars as pl
from pathlib import Path
from src.features.text_uncertainty import UncertaintyDetector
from src.features.text_specificity import SpecificityScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_path():
    return Path("configs/text_config.yaml")

@pytest.fixture
def sample_segments():
    return pl.DataFrame({
        "segment_id": ["seg1", "seg2"],
        "text": [
            "Revenue was $4.2 billion, up 15% from Q3. We are confident in our growth.",
            "Things are going well overall. We might possibly see some headwinds."
        ]
    })

# ---------------------------------------------------------------------------
# Uncertainty Tests
# ---------------------------------------------------------------------------

def test_uncertainty_scores(config_path):
    detector = UncertaintyDetector(config_path)
    
    # Text with explicit uncertainty terms from our lexicon
    text = "We might possibly see some uncertain headwinds."
    scores = detector.process_segment(text)
    
    assert scores["uncertainty_score"] > 0
    assert scores["hedging_frequency"] >= 0

def test_hedging_detection(config_path):
    detector = UncertaintyDetector(config_path)
    
    text = "We believe that to some extent things are likely to change."
    scores = detector.process_segment(text)
    
    # "We believe", "to some extent", "likely" are in our lexicons
    assert scores["hedging_frequency"] > 0

# ---------------------------------------------------------------------------
# Specificity Tests
# ---------------------------------------------------------------------------

def test_specificity_calculation(config_path):
    scorer = SpecificityScorer(config_path)
    
    high_spec = "Revenue was $4.2 billion, up 15% from Q3 in New York."
    low_spec = "Things are going well overall."
    
    res_high = scorer.process_segment(high_spec)
    res_low = scorer.process_segment(low_spec)
    
    assert res_high["specificity_score"] > res_low["specificity_score"]

def test_forward_looking_detection(config_path):
    scorer = SpecificityScorer(config_path)
    
    text = "We expect to grow next quarter and looking forward to 2025."
    res = scorer.process_segment(text)
    
    assert res["forward_looking_score"] > 0

def test_flesch_kincaid(config_path):
    scorer = SpecificityScorer(config_path)
    
    simple = "The cat sat on the mat."
    complex_text = "The multidimensional nature of global macroeconomic volatility necessitates a rigorous quantitative assessment."
    
    res_simple = scorer.process_segment(simple)
    res_complex = scorer.process_segment(complex_text)
    
    assert res_complex["linguistic_complexity"] > res_simple["linguistic_complexity"]
