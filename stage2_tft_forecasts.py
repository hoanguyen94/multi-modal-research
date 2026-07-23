"""Run Stage 2 with a Temporal Fusion Transformer market encoder.

This variant preserves the frozen TimesFM price representation and replaces
the row-level residual market MLP with an encoder-only TFT over historical
market feature windows. Raw embedding families are adapted jointly inside the
model, combined with field/family identities, contextualized by text
self-attention, and queried by the TFT state through cross-attention.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import polars as pl

from latent_fusion import (
    generate_timesfm_price_latents,
    parquet_embedding_dim,
    run_walk_forward_fusion,
)
from stage2_pretrained_forecasts import (
    ARTIFACT_DIR,
    BASELINE_DIR,
    DATA_DIR,
    EXPECTED_SUBMISSION_ROWS,
    FOLD_PATH,
    FUSION_DEPTH,
    FUSION_DROPOUT,
    FUSION_EPOCHS,
    FUSION_HIDDEN_DIM,
    HORIZON,
    ID_COLUMNS,
    LOOKBACK,
    MARKET_DEPTH,
    MIN_CONTEXT,
    OPTUNA_TRIALS,
    PREPARED_TEST_PATH,
    PREPARED_TRAIN_PATH,
    PRICE_BATCH_SIZE,
    PRICE_CACHE_DIR,
    PRICE_ENCODER_MODEL_ID,
    RANDOM_STATE,
    RESIDUAL_EXPANSION,
    SUBMISSION_YEARS,
    TEST_LINK_PATH,
    TEST_TARGET_PATH,
    TIMESFM_INPUT_COLUMN,
    TRAIN_LINK_PATH,
    TRAIN_TARGET_PATH,
    classify_covariates,
    require_paths,
    select_device,
)


OUTPUT_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_temporal_tft_unified_raw_text_attention"
)

# Keep the temporal tensor focused on genuinely historical price/market inputs.
# The full current-origin covariate vector still enters the TFT as static context.
TFT_TEMPORAL_COLUMNS = (
    "ret_1",
    "log_hl_range",
    "log_close_open",
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_std_5",
    "ret_std_20",
    "ret_std_60",
    "drawdown_20",
    "momentum_accel_5",
    "log_volume",
    "volume_change_1",
    "volume_z_20",
    "ret_1_market_relative",
    "ret_20_market_relative",
)
TFT_LOOKBACK = 32
TFT_LOOKBACK_CANDIDATES = (16, 20, 32)
TFT_ATTENTION_HEADS = 4
RAW_TEXT_DIM = 384
TEXT_ATTENTION_HEADS = 4
TEXT_ATTENTION_LAYERS = 1
RAW_FUSION_BATCH_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", default=["bert"],
        help="Original text-embedding families to use.",
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=OPTUNA_TRIALS,
        help=(
            "Persistent Optuna trial budget for each enabled inner-selection "
            "study."
        ),
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Use the default TFT and fusion hyperparameters.",
    )
    parser.add_argument(
        "--training-mode",
        choices=("nested-folds", "full-only"),
        default="nested-folds",
        help=(
            "Run five nested walk-forward folds before the final refit, or "
            "skip them and tune on one purged inner split of all training data."
        ),
    )
    parser.add_argument(
        "--no-price-extraction", action="store_true",
        help="Require existing TimesFM latent caches.",
    )
    parser.add_argument(
        "--force-price-refresh", action="store_true",
        help="Regenerate the frozen TimesFM latent caches.",
    )
    return parser.parse_args()


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
            "Optuna tuning is enabled. Install it with "
            "`python -m pip install optuna`, or pass --no-tune."
        )
    required_paths = [
        PREPARED_TRAIN_PATH,
        PREPARED_TEST_PATH,
        TRAIN_TARGET_PATH,
        TEST_TARGET_PATH,
        TRAIN_LINK_PATH,
        TEST_LINK_PATH,
        DATA_DIR / "test.parquet",
    ]
    if args.training_mode == "nested-folds":
        required_paths.append(FOLD_PATH)
    require_paths(required_paths)

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
    missing_temporal = sorted(set(TFT_TEMPORAL_COLUMNS) - set(past_market_covariates))
    if missing_temporal:
        raise ValueError(f"TFT temporal features are unavailable: {missing_temporal}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_groups.json").write_text(json.dumps({
        "frozen_price_encoder": PRICE_ENCODER_MODEL_ID,
        "timesfm_hidden_input": TIMESFM_INPUT_COLUMN,
        "market_encoder": "temporal_fusion_transformer",
        "tft_temporal_covariates": list(TFT_TEMPORAL_COLUMNS),
        "tft_lookback": TFT_LOOKBACK,
        "tft_lookback_candidates": list(TFT_LOOKBACK_CANDIDATES),
        "tft_attention_heads": TFT_ATTENTION_HEADS,
        "raw_text_shared_dim": RAW_TEXT_DIM,
        "text_attention_heads": TEXT_ATTENTION_HEADS,
        "text_attention_layers": TEXT_ATTENTION_LAYERS,
        "training_mode": args.training_mode,
        "current_past_market_covariates": list(past_market_covariates),
        "current_known_future_covariates": list(known_future_covariates),
        "current_text_availability_covariates": list(text_availability_covariates),
        "current_covariate_count": len(model_covariates),
    }, indent=2))

    train_targets = pl.read_parquet(TRAIN_TARGET_PATH).select([
        "row_id", "date", "ticker", "fwd_log_return_20", "target_up",
    ])
    test_targets = pl.read_parquet(TEST_TARGET_PATH).select([
        "row_id", "date", "ticker", "fwd_log_return_20", "target_up",
    ])
    train_links = pl.read_parquet(TRAIN_LINK_PATH)
    test_links = pl.read_parquet(TEST_LINK_PATH)
    fold_assignments = (
        pl.read_parquet(FOLD_PATH)
        if args.training_mode == "nested-folds" else None
    )
    train_origins = train_targets.filter(
        pl.col("target_up").is_not_null()
    ).select(ID_COLUMNS)
    test_origins = test_features.select(ID_COLUMNS)

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
        fusion_batch_size=RAW_FUSION_BATCH_SIZE,
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
        market_encoder="tft",
        temporal_covariate_columns=TFT_TEMPORAL_COLUMNS,
        temporal_lookback=TFT_LOOKBACK,
        temporal_lookback_candidates=TFT_LOOKBACK_CANDIDATES,
        tft_attention_heads=TFT_ATTENTION_HEADS,
        raw_text_dim=RAW_TEXT_DIM,
        text_attention_heads=TEXT_ATTENTION_HEADS,
        text_attention_layers=TEXT_ATTENTION_LAYERS,
        run_outer_folds=args.training_mode == "nested-folds",
    )

    reports = []
    if args.training_mode == "nested-folds":
        reports.extend([
            ("Fold metrics", "fold_metrics"),
            ("Aggregate walk-forward metrics", "aggregate"),
            ("Comparison with baselines", "comparison_aggregate"),
        ])
    reports.extend([
        ("Test metrics by prediction year", "final_metrics"),
        ("Submission files", "submission_manifest"),
    ])
    for title, key in reports:
        print(f"\n{title}")
        print(results[key])


if __name__ == "__main__":
    main()
