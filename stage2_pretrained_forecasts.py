"""Run Stage 2 frozen-TimesFM covariate and text fusion outside Jupyter.

This is the script equivalent of ``stage2_pretrained_forecasts.ipynb``.  It
loads the prepared Polars artifacts, reuses or extracts frozen TimesFM hidden
states, jointly adapts raw text-embedding families inside each fold, performs
Optuna model selection, refits on all labeled training rows, evaluates 2022
and 2023 separately, and writes one 52,000-row submission CSV per fitted model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Sequence

import polars as pl
import torch

from latent_fusion import (
    generate_timesfm_price_latents,
    parquet_embedding_dim,
    run_walk_forward_fusion,
)


DATA_DIR = Path("data")
ARTIFACT_DIR = DATA_DIR / "model_artifacts" / "pca_embeddings"
BASELINE_DIR = ARTIFACT_DIR / "baselines"
OUTPUT_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_covariates_unified_raw_text_attention2"
)
PRICE_CACHE_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_simple_concatenation"
    / "price_latents"
)
PREPARED_TRAIN_PATH = ARTIFACT_DIR / "train_features_without_embeddings.parquet"
PREPARED_TEST_PATH = ARTIFACT_DIR / "test_features_without_embeddings.parquet"
TRAIN_TARGET_PATH = ARTIFACT_DIR / "train_target.parquet"
TEST_TARGET_PATH = ARTIFACT_DIR / "test_target.parquet"
TRAIN_LINK_PATH = ARTIFACT_DIR / "train_text_links.parquet"
TEST_LINK_PATH = ARTIFACT_DIR / "test_text_links.parquet"
FOLD_PATH = ARTIFACT_DIR / "walk_forward_assignments.parquet"

PRICE_ENCODER_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
HORIZON = 20
LOOKBACK = 512
MIN_CONTEXT = 64
TIMESFM_INPUT_COLUMN = "ret_20"
DEFAULT_TEXT_FAMILIES = ("linq", "qwen") #"bert"
RAW_TEXT_DIM = 384
TEXT_ATTENTION_HEADS = 4
TEXT_ATTENTION_LAYERS = 1
FUSION_HIDDEN_DIM = 256
MARKET_DEPTH = 2
FUSION_DEPTH = 2
RESIDUAL_EXPANSION = 2
FUSION_DROPOUT = 0.10
FUSION_EPOCHS = 100
PRICE_BATCH_SIZE = 16
FUSION_BATCH_SIZE = 128
OPTUNA_TRIALS = 12
SUBMISSION_YEARS = (2022, 2023)
EXPECTED_SUBMISSION_ROWS = 52_000
RANDOM_STATE = 42

ID_COLUMNS = ("row_id", "date", "ticker")
TEXT_AVAILABILITY_PREFIXES = (
    "macro_",
    "sector_category_",
    "target_company_",
    "related_company_",
    "filing_",
)
KNOWN_FUTURE_PREFIXES = (
    "calendar_",
    "trading_day_",
    "trading_days_",
    "days_since_start",
    "is_month_",
    "is_first_",
    "is_last_",
    "is_quarter_end",
    "trading_week_fourier_",
    "month_of_year_fourier_",
    "trading_month_fourier_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_TEXT_FAMILIES),
        help="Original text-embedding families to use.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=OPTUNA_TRIALS,
        help="Total persistent Optuna trial budget.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Use the default residual-network hyperparameters.",
    )
    parser.add_argument(
        "--no-price-extraction",
        action="store_true",
        help="Require existing TimesFM latent caches.",
    )
    parser.add_argument(
        "--force-price-refresh",
        action="store_true",
        help="Regenerate the frozen TimesFM latent caches.",
    )
    return parser.parse_args()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_paths(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required Stage 2 artifacts are absent:\n{formatted}")


def classify_covariates(
    train_features: pl.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive feature groups from the prepared train schema."""
    text_availability = tuple(
        column
        for column in train_features.columns
        if column.startswith(TEXT_AVAILABILITY_PREFIXES)
        or column.startswith("has_")
        or column in ("text_count_total", "unique_text_id_count")
    )
    known_future = tuple(
        column
        for column in train_features.columns
        if column.startswith(KNOWN_FUTURE_PREFIXES)
    )
    past_market = tuple(
        column
        for column, dtype in train_features.schema.items()
        if dtype.is_numeric()
        and column not in ID_COLUMNS
        and column not in known_future
        and column not in text_availability
    )
    if TIMESFM_INPUT_COLUMN not in past_market:
        raise ValueError(
            f"{TIMESFM_INPUT_COLUMN} is absent from historical market features"
        )
    model_covariates = (*past_market, *known_future, *text_availability)
    if len(model_covariates) != len(set(model_covariates)):
        raise ValueError("Engineered covariate groups overlap")
    return past_market, known_future, text_availability, model_covariates


def main() -> None:
    args = parse_args()
    if args.no_price_extraction and args.force_price_refresh:
        raise ValueError(
            "--no-price-extraction and --force-price-refresh cannot be combined"
        )
    args.families = list(dict.fromkeys(args.families))
    for family in args.families:
        path = DATA_DIR / f"{family}_textemb.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Requested embedding family is absent: {path}")
        parquet_embedding_dim(path)
    if not args.no_tune and importlib.util.find_spec("optuna") is None:
        raise ImportError(
            "Optuna tuning is enabled. Install it in this environment with "
            "`python -m pip install optuna`, or pass --no-tune."
        )
    require_paths((
        PREPARED_TRAIN_PATH,
        PREPARED_TEST_PATH,
        TRAIN_TARGET_PATH,
        TEST_TARGET_PATH,
        TRAIN_LINK_PATH,
        TEST_LINK_PATH,
        FOLD_PATH,
        DATA_DIR / "test.parquet",
    ))

    device = select_device()
    print(f"Device: {device}")
    train_features = pl.read_parquet(PREPARED_TRAIN_PATH).sort(["ticker", "date"])
    test_features = pl.read_parquet(PREPARED_TEST_PATH).sort(["ticker", "date"])
    if train_features.columns != test_features.columns:
        raise ValueError("Train and test engineered-feature schemas differ")
    (
        past_market_covariates,
        known_future_covariates,
        text_availability_covariates,
        model_covariates,
    ) = classify_covariates(train_features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_groups.json").write_text(json.dumps({
        "timesfm_hidden_input": TIMESFM_INPUT_COLUMN,
        "past_market_covariates": list(past_market_covariates),
        "known_future_covariates": list(known_future_covariates),
        "past_text_availability_covariates": list(text_availability_covariates),
        "model_covariate_count": len(model_covariates),
        "timesfm_hidden_covariate_support": "univariate_target_only",
        "text_input": "raw_embedding_families",
        "raw_text_shared_dim": RAW_TEXT_DIM,
        "text_attention_heads": TEXT_ATTENTION_HEADS,
        "text_attention_layers": TEXT_ATTENTION_LAYERS,
    }, indent=2))

    train_targets = pl.read_parquet(TRAIN_TARGET_PATH).select([
        "row_id", "date", "ticker", "fwd_log_return_20", "target_up",
    ])
    test_targets = pl.read_parquet(TEST_TARGET_PATH).select([
        "row_id", "date", "ticker", "fwd_log_return_20", "target_up",
    ])
    train_links = pl.read_parquet(TRAIN_LINK_PATH)
    test_links = pl.read_parquet(TEST_LINK_PATH)
    fold_assignments = pl.read_parquet(FOLD_PATH)
    train_origins = train_targets.filter(
        pl.col("target_up").is_not_null()
    ).select(ID_COLUMNS)
    test_origins = test_features.select(ID_COLUMNS)
    print(
        f"Prepared features: train={train_features.shape}, test={test_features.shape}; "
        f"covariates={len(model_covariates)}"
    )

    train_price_latents = generate_timesfm_price_latents(
        split="train",
        prepared_features=train_features,
        origins=train_origins,
        cache_path=PRICE_CACHE_DIR / "train_timesfm_pooled_hidden.parquet",
        model_id=PRICE_ENCODER_MODEL_ID,
        device=device,
        horizon=HORIZON,
        lookback=LOOKBACK,
        min_context=MIN_CONTEXT,
        batch_size=PRICE_BATCH_SIZE,
        run_extraction=not args.no_price_extraction,
        force_refresh=args.force_price_refresh,
    )
    test_price_latents = generate_timesfm_price_latents(
        split="test",
        prepared_features=test_features,
        origins=test_origins,
        cache_path=PRICE_CACHE_DIR / "test_timesfm_pooled_hidden.parquet",
        model_id=PRICE_ENCODER_MODEL_ID,
        device=device,
        horizon=HORIZON,
        lookback=LOOKBACK,
        min_context=MIN_CONTEXT,
        batch_size=PRICE_BATCH_SIZE,
        run_extraction=not args.no_price_extraction,
        force_refresh=args.force_price_refresh,
    )
    print(
        f"Price latent caches: train={train_price_latents.shape}, "
        f"test={test_price_latents.shape}"
    )

    results = run_walk_forward_fusion(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        baseline_dir=BASELINE_DIR,
        train_price_latents=train_price_latents,
        test_price_latents=test_price_latents,
        train_features=train_features,
        test_features=test_features,
        train_links=train_links,
        test_links=test_links,
        train_targets=train_targets,
        test_targets=test_targets,
        fold_assignments=fold_assignments,
        requested_families=tuple(args.families),
        covariate_columns=model_covariates,
        device=device,
        fusion_epochs=FUSION_EPOCHS,
        fusion_batch_size=FUSION_BATCH_SIZE,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        market_depth=MARKET_DEPTH,
        fusion_depth=FUSION_DEPTH,
        residual_expansion=RESIDUAL_EXPANSION,
        fusion_dropout=FUSION_DROPOUT,
        tuning_trials=args.optuna_trials,
        tune_hyperparameters=not args.no_tune,
        forecast_horizon_weekdays=HORIZON,
        submission_years=SUBMISSION_YEARS,
        expected_submission_rows=EXPECTED_SUBMISSION_ROWS,
        raw_test_path=DATA_DIR / "test.parquet",
        seed=RANDOM_STATE,
        raw_text_dim=RAW_TEXT_DIM,
        text_attention_heads=TEXT_ATTENTION_HEADS,
        text_attention_layers=TEXT_ATTENTION_LAYERS,
    )

    print("\nFold metrics")
    print(results["fold_metrics"])
    print("\nAggregate walk-forward metrics")
    print(results["aggregate"])
    print("\nComparison with baselines")
    print(results["comparison_aggregate"])
    print("\nTest metrics by prediction year")
    print(results["final_metrics"])
    print("\nSubmission files")
    print(results["submission_manifest"])


if __name__ == "__main__":
    main()
