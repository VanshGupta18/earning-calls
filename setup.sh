#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Multimodal Earnings Call Intelligence System - Setup"
echo "=========================================================="

echo "1. Checking System Dependencies (FFmpeg)..."
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg could not be found."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Attempting to install via Homebrew..."
        brew install ffmpeg
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Attempting to install via apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "Please install FFmpeg manually and re-run."
        exit 1
    fi
else
    echo "FFmpeg is installed."
fi

echo "2. Setting up Python Virtual Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created at .venv/"
fi

# Activate venv
source .venv/bin/activate

echo "3. Installing Python Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "4. Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "5. Initializing Database..."
python src/preprocessing/db.py

echo "6. Environment Check Complete!"
echo ""
echo "To get started, activate the environment:"
echo "source .venv/bin/activate"
echo ""
echo "Then, pull the required datasets by running:"
echo "python scripts/download_transcripts.py"
echo "python scripts/download_market_data.py"
echo "python scripts/download_earnings22.py"
echo "=========================================================="
