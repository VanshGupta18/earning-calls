# Multimodal Earnings Call Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ready-00A67E)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A structured multimodal pipeline that analyzes earnings call audio, transcripts, and conversational structure to extract short-horizon market signals such as returns, volatility, earnings surprises, and risk proxies.

## Overview

## Current Status

> [!NOTE]
> **Professional Status:** The project is currently in the planning and data-foundation stage, with architecture, methodology, and tooling defined, and implementation of the ingestion and preprocessing pipeline as the next milestone.

- **Completed:** Problem statement defined, research direction finalized, system architecture drafted, README and project roadmap prepared, tech stack reviewed.
- **In Progress:** Dataset sourcing, environment setup, repository structuring, and preprocessing module planning.
- **Not Started:** Large-scale call collection, transcript alignment, feature extraction, model training, and dashboard implementation.


Earnings calls are not just disclosure events. They are interactive signaling systems where managers reveal information through:

- what they say,
- how they say it,
- and how they respond under analyst pressure.

This project models earnings calls as **pressure-driven information environments**. The key idea is that markets may underreact to inconsistencies between text, tone, and conversational behavior, especially in the Q&A section where managers are forced to respond in real time.

## Why this project matters

Most earnings-call analysis projects focus on one of three things:

- transcript sentiment,
- audio tone,
- or basic multimodal fusion.

This project goes further by explicitly modeling **interaction effects**:

- text–audio divergence,
- response latency,
- specificity under pressure,
- Q&A vs prepared remarks,
- speaker-role behavior.

The goal is not just to predict price movement, but to build a more realistic representation of how information is revealed during earnings calls.

## Novelty

The novelty of this project is **not** that earnings-call prediction is a new topic. Prior research has already shown that transcripts, audio cues, and Q&A structure can all contain useful signal.

The novelty is in how this system treats the earnings call as a **pressure-sensitive interaction system**. Instead of flattening the entire call into one document, it models the moments where managerial wording, vocal delivery, and response behavior diverge under questioning.

That means the project focuses on:

- contradictions between sentiment and tone,
- changes in specificity under analyst pressure,
- differences between prepared remarks and Q&A,
- interaction chains across speaker roles.

## Contributions

This project includes the following contributions:

1. A structured multimodal pipeline for earnings calls.
2. Segmentation of calls into prepared remarks, analyst questions, and management answers.
3. Text, audio, structural, and interaction-level feature extraction.
4. Explicit divergence features such as tone mismatch and specificity under pressure.
5. Hierarchical aggregation from segment level to call level.
6. Leakage-aware evaluation using strict time-based splits.
7. Decision-level outputs for returns, volatility, and risk use cases.

## Guaranteed Deliverables

The team commits to delivering the following concrete artifacts by the end of the project:

1. **Modular Codebase**: A fully documented GitHub repository for preprocessing, feature engineering, modeling, evaluation, and inference.
2. **Processed Dataset**: Earnings calls with aligned transcripts, speaker labels, and structural tags (Prepared Remarks, Questions, Answers).
3. **Multi-level Feature Table**: Segment and call-level features (text, audio, structural, interaction).
4. **Model Suite**: At least one trained baseline and one interaction-aware model with reproducible training scripts.
5. **Inference Artifacts**: Saved model weights and inference scripts for the best-performing configuration.
6. **Comparative Report**: Comprehensive analysis covering baselines, ablations, leakage controls, and error analysis.
7. **Research Dashboard**: A lightweight interface for viewing predictions, feature summaries, and call diagnostics.
8. **Final Presentation**: Summary of methodology, experiments, results, and future work.

> [!IMPORTANT]
> **Minimum Guaranteed Scope:** In case of limited time, the output will prioritize the documented codebase, processed dataset, baseline results, comparison report, and a simple demo interface.

## Problem Statement

Given an earnings call and related market data, predict short-horizon outcomes such as:

- next-day return,
- 1–5 day return,
- realized volatility,
- earnings surprise,
- downside risk proxy.

The system should learn whether unscripted call behavior contains predictive signal beyond what is already known from prices and text alone.

## Core Hypothesis

Markets may not fully price:

- vocal stress and hesitation,
- text–audio inconsistencies,
- low specificity under pressure,
- Q&A interaction dynamics.

The strongest signals are expected to appear when a manager’s narrative breaks down under questioning.

## System Architecture

```text
Raw Audio + Transcript
    ↓
Speaker Diarization + Transcript Alignment
    ↓
Structural Segmentation
    ↓
Feature Extraction
    ↓
Interaction Feature Engineering
    ↓
Hierarchical Aggregation
    ↓
Prediction Model
    ↓
Return / Volatility / Risk Signal
```

## Data Requirements

### Inputs

- Earnings call audio files.
- Transcript text, preferably timestamped.
- Speaker metadata such as CEO, CFO, analyst, operator.
- Market data such as prices, returns, volume, volatility, and fundamentals.

### Recommended storage layout

- Raw data: local disk or free-tier object storage if available.
- Processed segments: Parquet files.
- Feature tables: DuckDB locally, optional SQL database later.

This project is designed to work without paid infrastructure in the MVP stage.

## Preprocessing Pipeline

### 1. Audio preprocessing

- Resample audio to 16 kHz mono with torchaudio or librosa.
- Normalize amplitude.
- Run speaker diarization with pyannote.audio.
- Split audio into segment-level chunks.

### 2. Transcript alignment

- If timestamps exist, align directly.
- If not, use WhisperX for transcription and word-level alignment.
- Attach speaker and time boundaries to each segment.

### 3. Structural segmentation

Label each segment as one of:

- prepared remarks,
- analyst question,
- management answer,
- operator transition.

This step is critical because Q&A often carries more predictive information than scripted remarks.

## Feature Engineering

### Text features

- sentiment score,
- uncertainty language,
- forward-looking language,
- hedging frequency,
- linguistic complexity,
- specificity score.

### Audio features

- pitch variance,
- speech rate,
- pause duration,
- energy variance,
- voice stability,
- openSMILE acoustic descriptors,
- wav2vec2 embeddings.

### Structural features

- Q&A duration ratio,
- average answer length,
- response latency,
- speaker-specific behavior,
- turn-taking frequency.

### Interaction features

These are the most important features in the system:

- text–audio divergence,
- positive language with stressed delivery,
- low specificity under pressure,
- rising hesitation after difficult questions,
- analyst tone × management response dynamics.

## Formal Feature Definitions

- `specificity = (numeric_tokens + named_entities) / total_tokens`
- `tone_divergence = abs(text_sentiment - audio_valence)`
- `response_latency = answer_start_time - question_end_time`
- `analyst_pressure = negative_sentiment(question) × question_length`
- `qa_intensity = qa_duration / total_call_duration`

## Modeling Approach

### Baseline model

Start with a strong tabular model:

- LightGBM as the primary baseline,
- XGBoost as a benchmark.

Use engineered features first. This gives a clean baseline and helps prove whether the signal exists.

### Advanced model

If the baseline works, move to a hierarchical multimodal architecture:

- text encoder,
- audio encoder,
- cross-modal attention,
- hierarchical pooling,
- segment → section → call aggregation.

This version is more powerful but harder to train and validate.

## Aggregation Strategy

The system should aggregate features at three levels:

### Segment level

- raw extracted features.

### Section level

- prepared remarks,
- Q&A,
- analyst question,
- management answer.

Compute:
- mean,
- standard deviation,
- max,
- min.

### Call level

Use weighted aggregation so that Q&A has more influence than prepared remarks, since it is more likely to contain unscripted information.

Example:

```text
call_feature = 3 × Q&A features + 1 × prepared remarks features
```

## Targets

### Return

```text
return(t+1) = close(t+1) / close(t) - 1
```

```text
return(t+5) = close(t+5) / close(t) - 1
```

### Volatility

Use realized variance over the post-call window.

### Earnings surprise

```text
surprise = (reported - expected) / expected
```

## Leakage Control

This project must be leakage-aware from the start.

Rules:

- Use only information available at call time.
- Do not use post-call price movement as features.
- Use strict time-based train/test splits.
- Avoid overlapping windows across splits.
- Validate that transcript timestamps do not include future information.
- Check that labels are aligned only after the call ends.

This section is essential for credibility.

## Evaluation Plan

### Metrics

- Volatility: RMSE.
- Returns: directional accuracy.
- Risk ranking: information coefficient.
- Classification: AUROC, F1.
- Trading relevance: Sharpe ratio, turnover, drawdown.

### Benchmark setup

Compare against:

1. Price-only baseline.
2. Text-only model.
3. Audio-only model.
4. Late-fusion model.
5. Full interaction model.

### Ablation tests

Run ablations for:

- Q&A vs prepared remarks,
- divergence features on/off,
- structure features on/off,
- speaker metadata on/off.

## Inference Pipeline

```text
New earnings call
    ↓
Audio + transcript ingestion
    ↓
Segmentation
    ↓
Feature extraction
    ↓
Aggregation
    ↓
Prediction
    ↓
Signal output
```

### Outputs

- return score,
- volatility score,
- confidence score,
- optional trading direction.

## Trading Integration

This is optional, but if you want to use the model in a market setting:

- high predicted volatility can support options strategies,
- positive signals can support long positions,
- negative signals can support short positions.

For portfolio construction:

- rank stocks by signal,
- go long top decile,
- short bottom decile,
- rebalance around earnings events.

## Risk Controls

Because this is financial modeling, the system should include:

- signal decay monitoring,
- feature drift detection,
- regime shift detection,
- liquidity filters,
- confidence thresholds before trade execution.

## Preliminary Experiment Plan

### Step 1: The "High-Confidence" Baseline
The first experiment is designed to be simple and robust, focusing on one sector to reduce heterogeneity.

- **Dataset Scope:** 100–200 earnings calls from S&P 500 Technology or large-cap firms (2023–2024).
- **Primary Target:** Predict next-day realized volatility (a cleaner benchmark than directional returns).
- **Baseline Model:** Text-only model using FinBERT sentiment/uncertainty features + LightGBM regressor.
- **Success Criteria:** 
  1. End-to-end pipeline execution.
  2. Verified leakage-safe labeling.
  3. Text-only baseline outperforms price-only baseline.
  4. Q&A-aware features show measurable incremental value.

### Step 2: Multimodal Extension
Once the baseline is established, we will integrate:
- Audio prosody and openSMILE descriptors.
- Wav2vec2 embeddings.
- Text–audio divergence (interaction) features.

## Fallback Plan

If the full multimodal interaction pipeline proves too noisy or technically unstable, we will fall back in the following order:

### Fallback A: Text-only Structure-aware Model
Drop audio but preserve the Q&A vs. Prepared Remarks split and speaker-level features (specificity, uncertainty, sentiment).

### Fallback B: Q&A Structural Model
Focus purely on interaction dynamics such as response latency, answer length, turn-taking, and hedging frequency.

### Fallback C: Text-only Benchmark System
If diarization/alignment fails, build a strong section-level transcript-only system using FinBERT and aggregated call metadata.

## Project Roadmap

### Phase 1: Data foundation

- collect earnings calls,
- align audio and transcript with WhisperX,
- diarize speakers with pyannote.audio,
- split into segments,
- store structured datasets in Parquet and DuckDB.

### Phase 2: Baseline model

- build text-only and audio-only baselines,
- train LightGBM on engineered features,
- evaluate on time-based splits.

### Phase 3: Interaction layer

- add divergence features,
- build pressure-aware Q&A features,
- test whether they improve performance.

### Phase 4: Advanced modeling

- train hierarchical multimodal architecture,
- compare against baselines,
- run ablations.

### Phase 5: Deployment

- build inference pipeline,
- generate signals,
- add monitoring and retraining.

## Suggested Tech Stack

- Python
- Polars for pipelines, Pandas for compatibility
- PyTorch
- Hugging Face Transformers
- LightGBM as the main baseline, XGBoost as benchmark
- torchaudio + librosa
- WhisperX for transcription and alignment
- pyannote.audio for diarization
- openSMILE + wav2vec2 embeddings
- Parquet + DuckDB for local analytics
- Optional SQL database for team-scale workflows
- No paid APIs or paid managed services in the MVP

## Limitations

- audio quality varies across firms,
- transcript alignment may be noisy,
- event frequency is low,
- models may decay over time,
- leakage risk is high if evaluation is careless.

These limitations are normal for financial NLP projects and should be explicitly documented.

## Free-Tier Technology Policy

The MVP should use only free and open-source technologies:

- WhisperX for transcription and alignment.
- pyannote.audio for speaker diarization.
- FinBERT and Hugging Face models for text features.
- openSMILE and wav2vec2 for audio features.
- Polars, Pandas, DuckDB, and Parquet for data engineering.
- LightGBM and XGBoost for baseline modeling.
- PyTorch for advanced multimodal models.

No paid APIs are required. The only unavoidable cost is compute, which can be local, Colab-style, or low-cost cloud GPU time.

## Repository Structure

```text
repo/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── modeling/
│   ├── evaluation/
│   └── inference/
├── configs/
├── scripts/
├── outputs/
├── README.md
└── requirements.txt
```

## Team & Responsibilities

- **Devasya (Data Foundation & Preprocessing):** Dataset collection (WhisperX, pyannote), transcript cleaning, structural labeling, and storage layer (Parquet/DuckDB).
- **Aadi (Audio & Multimodal Modeling):** Audio pipeline (openSMILE, wav2vec2), audio quality checks, feature fusion, and multimodal interaction experiments.
- **Vansh (Text NLP & Evaluation):** Sentiment feature engineering (FinBERT), baseline modeling (LightGBM/XGBoost), leakage control, backtesting, and reporting.

## Summary


This project builds a pressure-aware multimodal system for earnings calls by modeling:

- text,
- audio,
- structure,
- interaction dynamics.

The key insight is simple:

**The strongest signals appear when a manager’s narrative breaks under pressure.**
