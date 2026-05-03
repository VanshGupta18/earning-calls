"""
Phase 4: Advanced Modeling — LightGBM + PCA Baseline

This script provides a robust baseline for the 20-call dataset.
It uses PCA to reduce the feature space and LightGBM for prediction.

Architecture:
    1. Load multimodal dataset
    2. Split features (Audio, Text, Interaction)
    3. PCA on high-dim Audio features (3000+ -> 16)
    4. Train LightGBM Regressor (Volatility)
    5. Train LightGBM Classifier (Direction)
    6. Evaluate and save results
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_lightgbm():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed = project_root / "data" / "processed"
    outputs = project_root / "outputs" / "models"
    outputs.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df_path = processed / "multimodal_dataset.parquet"
    if not df_path.exists():
        logger.error("multimodal_dataset.parquet not found. Run multimodal_join.py first.")
        return

    df = pl.read_parquet(df_path)
    
    # 2. Identify feature groups
    audio_cols = [c for c in df.columns if "_audio" in c or "wav2vec2" in c or "prosody" in c or "opensmile" in c]
    text_cols = [c for c in df.columns if any(k in c for k in ["sentiment", "uncertainty", "forward_looking", "hedging", "specificity", "linguistic"])]
    interaction_cols = [c for c in df.columns if any(k in c for k in ["divergence", "pressure", "qa_", "response_length"])]
    
    # Interaction features might overlap with text keywords, so let's refine
    text_cols = [c for c in text_cols if c not in interaction_cols]
    
    logger.info("Raw features: %d audio, %d text, %d interaction", 
                len(audio_cols), len(text_cols), len(interaction_cols))

    # 3. PCA on Audio Features (High dimension -> 16)
    audio_data = df.select(audio_cols).to_numpy()
    # Fill NaNs in audio
    audio_data = np.nan_to_num(audio_data)
    
    n_components = min(16, audio_data.shape[0], audio_data.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    audio_pca = pca.fit_transform(audio_data)
    
    # Create PCA column names
    pca_cols = [f"audio_pca_{i}" for i in range(n_components)]
    audio_pca_df = pl.DataFrame(audio_pca, schema=pca_cols)
    
    # 4. Prepare Final Feature Set
    X_df = pl.concat([
        audio_pca_df,
        df.select(text_cols),
        df.select(interaction_cols)
    ], how="horizontal")
    
    # Fill any remaining NaNs
    X = X_df.fill_null(0.0).to_numpy()
    feature_names = X_df.columns
    
    # Targets
    y_vol = df["realized_vol_5d"].to_numpy()
    y_ret = df["return_1d"].to_numpy()
    y_dir = (y_ret > 0).astype(int)
    
    # 5. Chronological Split (Train 60%, Val 20%, Test 20%)
    # Data is already sorted by call_date in the parquet usually, but let's be sure
    df = df.with_columns(pl.col("call_date").cast(pl.Date))
    indices = np.argsort(df["call_date"].to_numpy())
    
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_vol_train, y_dir_train = X[train_idx], y_vol[train_idx], y_dir[train_idx]
    X_val, y_vol_val, y_dir_val = X[val_idx], y_vol[val_idx], y_dir[val_idx]
    X_test, y_vol_test, y_dir_test = X[test_idx], y_vol[test_idx], y_dir[test_idx]

    logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    # 6. Train Volatility Regressor
    logger.info("Training Volatility Regressor (LightGBM)...")
    reg = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        random_state=42,
        verbosity=-1
    )
    reg.fit(X_train, y_vol_train, eval_set=[(X_val, y_vol_val)], 
            eval_metric='rmse', callbacks=[lgb.early_stopping(10)])

    # 7. Train Direction Classifier
    logger.info("Training Direction Classifier (LightGBM)...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        random_state=42,
        verbosity=-1
    )
    clf.fit(X_train, y_dir_train, eval_set=[(X_val, y_dir_val)], 
            eval_metric='binary_logloss', callbacks=[lgb.early_stopping(10)])

    # 8. Evaluate on Test Set
    vol_pred = reg.predict(X_test)
    dir_pred = clf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_vol_test, vol_pred))
    acc = accuracy_score(y_dir_test, dir_pred)
    
    # Information Coefficient (IC) - Correlation for volatility
    ic, _ = pearsonr(vol_pred, y_vol_test) if len(y_vol_test) > 1 else (0.0, 0.0)

    results = {
        "mode": "lightgbm_pca",
        "features": len(feature_names),
        "rmse": float(rmse),
        "ic": float(ic),
        "accuracy": float(acc),
        "params": {
            "n_audio_pca": n_components,
            "n_text": len(text_cols),
            "n_interaction": len(interaction_cols)
        }
    }

    logger.info("=" * 40)
    logger.info("LIGHTGBM TEST RESULTS:")
    logger.info("RMSE: %.6f", rmse)
    logger.info("IC:   %.4f", ic)
    logger.info("ACC:  %.4f", acc)
    logger.info("=" * 40)

    # Save results
    with open(outputs / "results_lightgbm.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save feature importance
    importance = pl.DataFrame({
        "feature": feature_names,
        "importance": reg.feature_importances_
    }).sort("importance", descending=True)
    
    logger.info("Top Features:\n%s", importance.head(10))

    return results


if __name__ == "__main__":
    train_lightgbm()
