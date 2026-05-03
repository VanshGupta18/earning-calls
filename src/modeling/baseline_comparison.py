"""
Baseline Model Comparison — Phase 3, Task 3.D.4

Trains and compares three LightGBM baselines:
    1. Text-only:   sentiment, uncertainty, specificity, linguistic complexity
    2. Audio-only:  prosody, openSMILE aggregated features
    3. Combined:    text + audio early fusion (feature concatenation)

Each model predicts 5-day realized volatility (realized_vol_5d) and
1-day return direction (return_1d > 0).

Outputs:
    outputs/baseline_comparison.json   — metrics for all three models
    outputs/feature_importance.json    — top-20 features per model
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)

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
# Helpers
# ---------------------------------------------------------------------------

def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation, returning 0 on failure."""
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        ic, _ = spearmanr(y_true, y_pred)
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        return 0.0


def clean_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Replace NaN/Inf with 0 in feature columns."""
    for col in feature_cols:
        if col in df.columns and df[col].dtype in (pl.Float32, pl.Float64):
            df = df.with_columns(
                pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite())
                .then(0.0)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df


def get_feature_importance(model, feature_names: list[str], top_n: int = 20) -> list[dict]:
    """Extract top-N feature importances from a LightGBM model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": int(importances[i])}
        for i in indices if importances[i] > 0
    ]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_regression_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> dict:
    """Train a LightGBM regressor and return metrics."""
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    ic = safe_spearman(y_test, y_pred)
    mae = float(np.mean(np.abs(y_test - y_pred)))

    importance = get_feature_importance(model, feature_names)

    logger.info(
        "  [%s] Regression → RMSE=%.6f, IC=%.4f, MAE=%.6f",
        model_name, rmse, ic, mae,
    )

    return {
        "model": model_name,
        "task": "volatility_regression",
        "target": "realized_vol_5d",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_names),
        "rmse": rmse,
        "ic": ic,
        "mae": mae,
        "top_features": importance,
    }


def train_classification_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> dict:
    """Train a LightGBM classifier and return metrics."""
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # AUC only if both classes present
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, y_pred_proba))
    else:
        auc = 0.5

    importance = get_feature_importance(model, feature_names)

    logger.info(
        "  [%s] Classification → Acc=%.4f, F1=%.4f, AUC=%.4f",
        model_name, acc, f1, auc,
    )

    return {
        "model": model_name,
        "task": "direction_classification",
        "target": "return_1d_direction",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_names),
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "top_features": importance,
    }


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def run_comparison(project_root: Path) -> None:
    """Run the full baseline comparison."""
    processed = project_root / "data" / "processed"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load text + market dataset
    tm_path = processed / "text_market_dataset.parquet"
    if not tm_path.exists():
        logger.error("text_market_dataset.parquet not found. Run multimodal_join.py first.")
        return

    df = pl.read_parquet(tm_path)
    logger.info("Loaded text_market dataset: %d rows, %d columns", len(df), len(df.columns))

    # -----------------------------------------------------------------------
    # Identify feature groups
    # -----------------------------------------------------------------------
    meta_cols = {
        "call_id", "ticker", "call_date", "close_t0", "close_t1", "close_t5",
        "return_1d", "return_5d", "realized_vol_1d", "realized_vol_5d",
        "earnings_surprise",
    }

    text_feature_cols = [
        c for c in df.columns
        if c not in meta_cols and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    logger.info("Text feature columns (%d): %s", len(text_feature_cols), text_feature_cols[:10])

    # -----------------------------------------------------------------------
    # Prepare targets
    # -----------------------------------------------------------------------
    # Regression target: realized_vol_5d
    reg_target = "realized_vol_5d"
    # Classification target: return_1d direction (positive = 1)
    df = df.with_columns(
        (pl.col("return_1d") > 0).cast(pl.Int32).alias("return_1d_direction")
    )
    cls_target = "return_1d_direction"

    # Filter valid rows
    df = df.filter(
        pl.col(reg_target).is_not_null()
        & pl.col(reg_target).is_finite()
        & pl.col("return_1d").is_not_null()
    )

    # Clean features
    df = clean_features(df, text_feature_cols)

    # -----------------------------------------------------------------------
    # Chronological split
    # -----------------------------------------------------------------------
    df = df.sort("call_date")
    n = len(df)
    train_end = int(n * 0.7)

    train_df = df.head(train_end)
    test_df = df.tail(n - train_end)

    if len(train_df) < 3 or len(test_df) < 2:
        logger.warning(
            "Very small dataset (train=%d, test=%d). Results will be noisy.",
            len(train_df), len(test_df),
        )

    logger.info(
        "Split: train=%d (%s → %s), test=%d (%s → %s)",
        len(train_df), train_df["call_date"].min(), train_df["call_date"].max(),
        len(test_df), test_df["call_date"].min(), test_df["call_date"].max(),
    )

    # -----------------------------------------------------------------------
    # Feature preparation
    # -----------------------------------------------------------------------
    X_train_text = train_df.select(text_feature_cols).to_numpy()
    X_test_text = test_df.select(text_feature_cols).to_numpy()

    y_train_reg = train_df[reg_target].to_numpy()
    y_test_reg = test_df[reg_target].to_numpy()

    y_train_cls = train_df[cls_target].to_numpy()
    y_test_cls = test_df[cls_target].to_numpy()

    # -----------------------------------------------------------------------
    # Model 1: Text-only
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 1: Text-Only Features")
    logger.info("=" * 60)

    results = []

    reg_result = train_regression_model(
        X_train_text, y_train_reg, X_test_text, y_test_reg,
        text_feature_cols, "Text-Only",
    )
    results.append(reg_result)

    cls_result = train_classification_model(
        X_train_text, y_train_cls, X_test_text, y_test_cls,
        text_feature_cols, "Text-Only",
    )
    results.append(cls_result)

    # -----------------------------------------------------------------------
    # Model 2: Price-Only (simple baseline)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 2: Price-Only Baseline")
    logger.info("=" * 60)

    # Use normalized price as the only feature
    if "close_t0" in df.columns:
        train_df_price = train_df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )
        test_df_price = test_df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )

        price_cols = ["price_norm"]
        X_train_price = train_df_price.select(price_cols).to_numpy()
        X_test_price = test_df_price.select(price_cols).to_numpy()

        reg_result_price = train_regression_model(
            X_train_price, y_train_reg, X_test_price, y_test_reg,
            price_cols, "Price-Only",
        )
        results.append(reg_result_price)

        cls_result_price = train_classification_model(
            X_train_price, y_train_cls, X_test_price, y_test_cls,
            price_cols, "Price-Only",
        )
        results.append(cls_result_price)

    # -----------------------------------------------------------------------
    # Model 3: Text + Structural (Combined)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 3: Text + Structural + Price Combined")
    logger.info("=" * 60)

    combined_cols = text_feature_cols.copy()
    if "close_t0" in df.columns:
        # Add price normalization to original df
        df = df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )
        train_df = train_df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )
        test_df = test_df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )
        combined_cols.append("price_norm")

    X_train_combined = train_df.select(combined_cols).to_numpy()
    X_test_combined = test_df.select(combined_cols).to_numpy()

    reg_result_combined = train_regression_model(
        X_train_combined, y_train_reg, X_test_combined, y_test_reg,
        combined_cols, "Text+Structural+Price",
    )
    results.append(reg_result_combined)

    cls_result_combined = train_classification_model(
        X_train_combined, y_train_cls, X_test_combined, y_test_cls,
        combined_cols, "Text+Structural+Price",
    )
    results.append(cls_result_combined)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    for r in results:
        if r["task"] == "volatility_regression":
            logger.info(
                "  %-30s RMSE=%.6f  IC=%.4f  MAE=%.6f",
                r["model"] + " (reg)", r["rmse"], r["ic"], r["mae"],
            )
        else:
            logger.info(
                "  %-30s Acc=%.4f  F1=%.4f  AUC=%.4f",
                r["model"] + " (cls)", r["accuracy"], r["f1"], r["auc"],
            )

    # Save results
    with open(output_dir / "baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to: %s", output_dir / "baseline_comparison.json")

    # Save feature importance
    importance_summary = {
        r["model"] + "_" + r["task"]: r.get("top_features", [])
        for r in results
    }
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance_summary, f, indent=2)
    logger.info("Saved feature importance to: %s", output_dir / "feature_importance.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    run_comparison(project_root)


if __name__ == "__main__":
    main()
