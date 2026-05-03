#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Multimodal Earnings Call Intelligence System - Setup"
echo "=========================================================="

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "1. Checking System Dependencies..."

# FFmpeg (required for audio processing)
if ! command -v ffmpeg &> /dev/null; then
    echo "  FFmpeg not found."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Installing via Homebrew..."
        brew install ffmpeg
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  Installing via apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "  Please install FFmpeg manually and re-run."
        exit 1
    fi
else
    echo "  ✓ FFmpeg is installed."
fi

# libomp (required for LightGBM on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list libomp &> /dev/null 2>&1; then
        echo "  Installing libomp (required for LightGBM)..."
        brew install libomp
    else
        echo "  ✓ libomp is installed."
    fi
fi

# ---------------------------------------------------------------------------
# 2. Python environment via uv (preferred) or pip
# ---------------------------------------------------------------------------
echo "2. Setting up Python environment..."

if command -v uv &> /dev/null; then
    echo "  Using uv for dependency management..."
    uv sync
    echo "  ✓ Dependencies installed via uv."
else
    echo "  uv not found, falling back to pip..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo "  Virtual environment created at .venv/"
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "  ✓ Dependencies installed via pip."
fi

# ---------------------------------------------------------------------------
# 3. spaCy model
# ---------------------------------------------------------------------------
echo "3. Downloading spaCy English model..."
if command -v uv &> /dev/null; then
    .venv/bin/python -m ensurepip 2>/dev/null || true
    .venv/bin/python -m pip install \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
        2>/dev/null || echo "  spaCy model may already be installed."
else
    python -m spacy download en_core_web_sm
fi
echo "  ✓ spaCy model ready."

# ---------------------------------------------------------------------------
# 4. Database initialization
# ---------------------------------------------------------------------------
echo "4. Initializing DuckDB database..."
if command -v uv &> /dev/null; then
    uv run python -c "from src.preprocessing.db import init_database; init_database('data/processed/earning_calls.db')" 2>/dev/null || true
else
    PYTHONPATH=. python -c "from src.preprocessing.db import init_database; init_database('data/processed/earning_calls.db')" 2>/dev/null || true
fi
echo "  ✓ Database initialized."

# ---------------------------------------------------------------------------
# 5. Environment variable hints
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo " Setup Complete!"
echo "=========================================================="
echo ""
echo "To get started:"
echo "  source .venv/bin/activate"
echo ""

# macOS-specific: export libomp path for LightGBM
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  # Required for LightGBM on macOS:"
    echo "  export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib"
    echo ""
fi

echo "Then pull the required datasets:"
echo "  python scripts/download_transcripts.py --tech-only --max-calls 200"
echo "  python scripts/download_market_data.py"
echo "  python scripts/download_earnings22.py"
echo "=========================================================="
