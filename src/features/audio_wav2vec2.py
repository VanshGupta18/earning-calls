"""
Vocal embedding extraction using wav2vec 2.0 for earnings call segments.

Extracts deep acoustic representations from segments using a pre-trained
wav2vec2 model. Embeddings are mean-pooled across time to produce a 
768-dimensional vector per segment.

Outputs: data/processed/audio_wav2vec2.parquet
"""

import logging
from pathlib import Path

import polars as pl
import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

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
# wav2vec2 Extractor
# ---------------------------------------------------------------------------


class Wav2Vec2Extractor:
    def __init__(self, config_path: Path):
        self.model_name = "facebook/wav2vec2-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for Apple Silicon MPS
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        logger.info("Loading wav2vec2 model: %s on %s", self.model_name, self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def extract_embedding(self, audio_path: str) -> list[float] | None:
        """Extract mean-pooled embedding for a single audio file."""
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if not 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Preprocess
            inputs = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling across time (dimension 1)
            # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return embeddings.cpu().tolist()
        except Exception as e:
            logger.warning("wav2vec2 failed for %s: %s", audio_path, e)
            return None

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run extraction on a Polars DataFrame of segments."""
        logger.info("Extracting wav2vec2 embeddings from %d segments...", len(segments))
        
        results = []
        segment_ids = []
        
        for row in tqdm(segments.iter_rows(named=True), desc="wav2vec2"):
            emb = self.extract_embedding(row["audio_path"])
            if emb:
                results.append(emb)
                segment_ids.append(row["segment_id"])

        if not results:
            return pl.DataFrame({"segment_id": []})

        # Convert list of embeddings to columns (emb_0, emb_1, ...)
        # Contract C says 'vocal_embeddings' (JSON or multi-col)
        # We'll use 768 float columns to match typical Parquet patterns
        n_dim = len(results[0])
        col_names = [f"wav2vec2_{i}" for i in range(n_dim)]
        
        df_embs = pl.DataFrame(results, schema=col_names)
        
        return pl.DataFrame({"segment_id": segment_ids}).with_columns(df_embs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    # config_path = project_root / "configs" / "audio_config.yaml"
    segments_path = project_root / "data" / "processed" / "earnings22_segments.parquet"
    output_path = project_root / "data" / "processed" / "audio_wav2vec2.parquet"

    if not segments_path.exists():
        logger.error("Segments file not found.")
        return

    df_segments = pl.read_parquet(segments_path)
    
    # Subset for testing
    # df_segments = df_segments.head(20)

    extractor = Wav2Vec2Extractor(None) # Passing None for config now
    df_features = extractor.extract(df_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved wav2vec2 embeddings (%d rows) to: %s", len(df_features), output_path)


if __name__ == "__main__":
    main()
