-- ============================================================
-- Multimodal Earnings Call Intelligence System
-- DuckDB Schema — All Four Data Contracts
-- Run with: duckdb earnings.db < scripts/init_db.sql
-- ============================================================

-- Contract A: segments
-- One row per speaker segment within a call.
CREATE TABLE IF NOT EXISTS segments (
    call_id        VARCHAR   NOT NULL,
    segment_id     VARCHAR   NOT NULL PRIMARY KEY,
    speaker_role   VARCHAR   NOT NULL,   -- ceo | cfo | analyst | operator | executive | other
    speaker_name   VARCHAR,
    segment_type   VARCHAR   NOT NULL,   -- prepared_remarks | analyst_question | management_answer | operator_transition
    text           VARCHAR   NOT NULL,
    start_time     DOUBLE,               -- seconds; NULL in text-only MVP
    end_time       DOUBLE,               -- seconds; NULL in text-only MVP
    audio_path     VARCHAR               -- relative path to WAV; NULL in text-only MVP
);

CREATE INDEX IF NOT EXISTS idx_segments_call_id  ON segments (call_id);
CREATE INDEX IF NOT EXISTS idx_segments_type     ON segments (segment_type);
CREATE INDEX IF NOT EXISTS idx_segments_role     ON segments (speaker_role);

-- Contract B: text_features
-- One row per segment; FK to segments.segment_id.
CREATE TABLE IF NOT EXISTS text_features (
    segment_id              VARCHAR  NOT NULL PRIMARY KEY,
    sentiment_score         DOUBLE,  -- FinBERT: positive_prob - negative_prob, range [-1, 1]
    uncertainty_score       DOUBLE,  -- ratio of uncertainty language tokens [0, 1]
    forward_looking_score   DOUBLE,  -- ratio of forward-looking phrases [0, 1]
    hedging_frequency       DOUBLE,  -- ratio of hedging tokens [0, 1]
    specificity_score       DOUBLE,  -- (numeric_tokens + named_entities) / total_tokens [0, 1]
    linguistic_complexity   DOUBLE   -- Flesch-Kincaid grade level
);

CREATE INDEX IF NOT EXISTS idx_text_features_segment ON text_features (segment_id);

-- Contract C: audio_features
-- One row per segment; FK to segments.segment_id.
CREATE TABLE IF NOT EXISTS audio_features (
    segment_id            VARCHAR    NOT NULL PRIMARY KEY,
    pitch_mean            DOUBLE,    -- mean F0 in Hz
    pitch_variance        DOUBLE,    -- variance of F0
    speech_rate           DOUBLE,    -- syllables per second
    pause_duration_total  DOUBLE,    -- total silence in seconds
    energy_variance       DOUBLE,    -- variance of RMS energy
    voice_stability       DOUBLE,    -- jitter/shimmer-based stability score
    opensmile_vector      DOUBLE[],  -- eGeMAPS 88-dim feature vector
    wav2vec2_embedding    DOUBLE[]   -- wav2vec2 768-dim mean-pooled embedding
);

CREATE INDEX IF NOT EXISTS idx_audio_features_segment ON audio_features (segment_id);

-- Contract D: market_data
-- One row per earnings call; FK to segments.call_id.
CREATE TABLE IF NOT EXISTS market_data (
    call_id           VARCHAR NOT NULL PRIMARY KEY,
    ticker            VARCHAR NOT NULL,
    call_date         DATE    NOT NULL,
    close_t0          DOUBLE,          -- closing price on call date
    close_t1          DOUBLE,          -- closing price next trading day
    close_t5          DOUBLE,          -- closing price 5 trading days later
    return_1d         DOUBLE,          -- (close_t1 / close_t0) - 1
    return_5d         DOUBLE,          -- (close_t5 / close_t0) - 1
    realized_vol_1d   DOUBLE,          -- next-day realized volatility
    realized_vol_5d   DOUBLE,          -- 5-day realized volatility
    earnings_surprise DOUBLE           -- (reported - expected) / expected; NULL if unavailable
);

CREATE INDEX IF NOT EXISTS idx_market_data_ticker    ON market_data (ticker);
CREATE INDEX IF NOT EXISTS idx_market_data_date      ON market_data (call_date);
