"""Run Stage 2 with a Temporal Fusion Transformer market encoder.

By default, this variant preserves a frozen TimesFM or multivariate Chronos-2
price representation and replaces the row-level residual market MLP with an
encoder-only TFT over historical market feature windows. The pretrained price
representation can be disabled for a TFT-plus-covariates ablation. Raw
embedding families are adapted jointly inside the model, combined with
field/family identities, contextualized by text self-attention, and queried by
the TFT state through cross-attention.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import polars as pl

from latent_fusion import (
    generate_frozen_price_latents,
    parquet_embedding_dim,
    run_walk_forward_fusion,
)
from model_config import (
    ADAPTER_LEARNING_RATE_MULTIPLIER,
    BASELINE_DIR,
    CALIBRATE_DECISION_THRESHOLD,
    DATA_DIR,
    DEFAULT_PRICE_ENCODER,
    DEFAULT_TEXT_FAMILIES,
    DEFAULT_TRAINING_MODE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MIN_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EXPECTED_SUBMISSION_ROWS,
    FIXED_DECISION_THRESHOLD,
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
    PRICE_ENCODERS,
    RANDOM_STATE,
    RAW_FUSION_BATCH_SIZE,
    RAW_TEST_PATH,
    RAW_TEXT_DIM,
    RESIDUAL_EXPANSION,
    SUBMISSION_YEARS,
    TEST_LINK_PATH,
    TEST_TARGET_PATH,
    TEXT_ATTENTION_HEADS,
    TEXT_ATTENTION_LAYERS,
    TFT_ATTENTION_HEADS,
    TFT_LOOKBACK,
    TFT_LOOKBACK_CANDIDATES,
    TFT_OUTPUT_DIRS,
    TFT_TEMPORAL_COLUMNS,
    TRAINING_MODES,
    TRAIN_LINK_PATH,
    TRAIN_TARGET_PATH,
    CROSS_STOCK_ATTENTION_HEADS,
    USE_PRETRAINED_PRICE_MODEL,
)
from stage2_pretrained_forecasts import (
    classify_covariates,
    price_encoder_settings,
    require_paths,
    select_device,
)


def parse_args(description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--families", nargs="+", default=list(DEFAULT_TEXT_FAMILIES),
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
        choices=TRAINING_MODES,
        default=DEFAULT_TRAINING_MODE,
        help=(
            "Run five nested walk-forward folds before the final refit, or "
            "skip them and tune on one purged split of all training data."
        ),
    )
    parser.add_argument(
        "--price-encoder",
        choices=PRICE_ENCODERS,
        default=DEFAULT_PRICE_ENCODER,
        help=(
            "Frozen backbone: TimesFM over ret_20 or Chronos-2 over jointly "
            "attended market variates."
        ),
    )
    pretrained_model_group = parser.add_mutually_exclusive_group()
    pretrained_model_group.add_argument(
        "--no-pretrained-model",
        dest="no_pretrained_model",
        action="store_true",
        help=(
            "Do not load or use a pretrained TimesFM/Chronos price model. "
            "The TFT still uses engineered static and temporal covariates, "
            "and the configured text embeddings remain enabled."
        ),
    )
    pretrained_model_group.add_argument(
        "--use-pretrained-model",
        dest="no_pretrained_model",
        action="store_false",
        help=(
            "Use the selected pretrained TimesFM/Chronos price model, "
            "overriding USE_PRETRAINED_PRICE_MODEL=False."
        ),
    )
    parser.set_defaults(
        no_pretrained_model=not USE_PRETRAINED_PRICE_MODEL
    )
    parser.add_argument(
        "--no-price-extraction", action="store_true",
        help="Require existing frozen price-latent caches.",
    )
    parser.add_argument(
        "--force-price-refresh", action="store_true",
        help="Regenerate the frozen price-latent caches.",
    )
    return parser.parse_args()


def _index_only_price_inputs(origins: pl.DataFrame) -> pl.DataFrame:
    """Return aligned row keys without any pretrained price-latent columns."""
    missing = sorted(set(ID_COLUMNS) - set(origins.columns))
    if missing:
        raise ValueError(
            f"Price-input origins are missing identifier columns: {missing}"
        )
    index = origins.select(ID_COLUMNS)
    if index.select(
        pl.any_horizontal(
            pl.col(column).is_null() for column in ID_COLUMNS
        ).any()
    ).item():
        raise ValueError("Price-input identifiers cannot contain null values")
    if index["row_id"].n_unique() != index.height:
        raise ValueError("Price-input row_id values must be unique")
    return index.sort(["ticker", "date"])


def run_tft_pipeline(
    args: argparse.Namespace,
    *,
    output_dir=None,
    cross_stock_attention: bool = False,
    cross_stock_attention_heads: int = CROSS_STOCK_ATTENTION_HEADS,
) -> dict[str, pl.DataFrame]:
    no_pretrained_model = bool(
        getattr(
            args,
            "no_pretrained_model",
            not USE_PRETRAINED_PRICE_MODEL,
        )
    )
    if args.no_price_extraction and args.force_price_refresh:
        raise ValueError(
            "--no-price-extraction and --force-price-refresh cannot be combined"
        )
    if no_pretrained_model and (
        args.no_price_extraction or args.force_price_refresh
    ):
        raise ValueError(
            "--no-price-extraction/--force-price-refresh cannot be used with "
            "--no-pretrained-model"
        )
    if args.optuna_trials < 0:
        raise ValueError("--optuna-trials cannot be negative")
    args.families = list(dict.fromkeys(args.families))
    usable_family_count = 0
    for family in args.families:
        path = DATA_DIR / f"{family}_textemb.parquet"
        try:
            parquet_embedding_dim(path)
            usable_family_count += 1
        except Exception as error:
            print(
                f"Warning: skipping unusable embedding family "
                f"{family!r}: {error}"
            )
    if not usable_family_count:
        raise RuntimeError("No requested text-embedding family is usable")
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
        RAW_TEST_PATH,
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
    if no_pretrained_model:
        price_model_id = None
        price_input_columns = ()
        price_cache_dir = None
    else:
        (
            price_model_id,
            price_input_columns,
            price_cache_dir,
            _,
        ) = price_encoder_settings(args.price_encoder)
    if output_dir is None:
        if no_pretrained_model:
            output_dir = (
                TFT_OUTPUT_DIRS[args.price_encoder].parent
                / "tft_no_pretrained_price_unified_raw_text_attention"
            )
        else:
            output_dir = TFT_OUTPUT_DIRS[args.price_encoder]
    missing_price_inputs = sorted(
        set(price_input_columns) - set(past_market_covariates)
    )
    if missing_price_inputs:
        raise ValueError(
            f"{args.price_encoder} historical inputs are unavailable: "
            f"{missing_price_inputs}"
        )
    missing_temporal = sorted(set(TFT_TEMPORAL_COLUMNS) - set(past_market_covariates))
    if missing_temporal:
        raise ValueError(f"TFT temporal features are unavailable: {missing_temporal}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_groups.json").write_text(json.dumps({
        "pretrained_price_model_enabled": not no_pretrained_model,
        "frozen_price_encoder": (
            None if no_pretrained_model else args.price_encoder
        ),
        "price_encoder_model_id": price_model_id,
        "price_encoder_input_columns": list(price_input_columns),
        "price_hidden_covariate_support": (
            "none"
            if no_pretrained_model else
            "univariate_target_only"
            if args.price_encoder == "timesfm"
            else "joint_multivariate_targets"
        ),
        "price_hidden_pooling": (
            "none"
            if no_pretrained_model else
            "mean_context_patches"
            if args.price_encoder == "chronos2"
            else "mean_hidden_sequence"
        ),
        "market_encoder": "temporal_fusion_transformer",
        "tft_temporal_covariates": list(TFT_TEMPORAL_COLUMNS),
        "tft_lookback": TFT_LOOKBACK,
        "tft_lookback_candidates": list(TFT_LOOKBACK_CANDIDATES),
        "tft_attention_heads": TFT_ATTENTION_HEADS,
        "raw_text_shared_dim": RAW_TEXT_DIM,
        "text_attention_heads": TEXT_ATTENTION_HEADS,
        "text_attention_layers": TEXT_ATTENTION_LAYERS,
        "adapter_learning_rate_multiplier": (
            ADAPTER_LEARNING_RATE_MULTIPLIER
        ),
        "inner_validation_splits": 1,
        "training_loss": "binary_cross_entropy",
        "checkpoint_metric": "validation_bce",
        "optuna_objective": "negative_validation_bce",
        "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "threshold_calibrated": CALIBRATE_DECISION_THRESHOLD,
        "fixed_decision_threshold": FIXED_DECISION_THRESHOLD,
        "epoch_selection": (
            "post_optuna_inner_validation"
            if not args.no_tune and args.optuna_trials > 0 else
            "fixed_configuration_inner_validation"
        ),
        "threshold_selection": (
            "exact_inner_validation_balanced_accuracy"
            if CALIBRATE_DECISION_THRESHOLD else "fixed"
        ),
        "cross_stock_attention": cross_stock_attention,
        "cross_stock_attention_heads": (
            cross_stock_attention_heads
            if cross_stock_attention else None
        ),
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

    if no_pretrained_model:
        train_price_latents = _index_only_price_inputs(train_origins)
        test_price_latents = _index_only_price_inputs(test_origins)
    else:
        train_price_latents = generate_frozen_price_latents(
            encoder=args.price_encoder,
            input_columns=price_input_columns,
            split="train",
            prepared_features=train_features,
            origins=train_origins,
            cache_path=price_cache_dir / (
                f"train_{args.price_encoder}_pooled_hidden.parquet"
            ),
            model_id=price_model_id,
            device=device,
            horizon=HORIZON,
            lookback=LOOKBACK,
            min_context=MIN_CONTEXT,
            batch_size=PRICE_BATCH_SIZE,
            run_extraction=not args.no_price_extraction,
            force_refresh=args.force_price_refresh,
        )
        test_price_latents = generate_frozen_price_latents(
            encoder=args.price_encoder,
            input_columns=price_input_columns,
            split="test",
            prepared_features=test_features,
            origins=test_origins,
            cache_path=price_cache_dir / (
                f"test_{args.price_encoder}_pooled_hidden.parquet"
            ),
            model_id=price_model_id,
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
        output_dir=output_dir,
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
        raw_test_path=RAW_TEST_PATH,
        seed=RANDOM_STATE,
        market_encoder="tft",
        temporal_covariate_columns=TFT_TEMPORAL_COLUMNS,
        temporal_lookback=TFT_LOOKBACK,
        temporal_lookback_candidates=TFT_LOOKBACK_CANDIDATES,
        tft_attention_heads=TFT_ATTENTION_HEADS,
        raw_text_dim=RAW_TEXT_DIM,
        text_attention_heads=TEXT_ATTENTION_HEADS,
        text_attention_layers=TEXT_ATTENTION_LAYERS,
        cross_stock_attention=cross_stock_attention,
        cross_stock_attention_heads=cross_stock_attention_heads,
        adapter_learning_rate_multiplier=(
            ADAPTER_LEARNING_RATE_MULTIPLIER
        ),
        calibrate_decision_threshold=CALIBRATE_DECISION_THRESHOLD,
        fixed_decision_threshold=FIXED_DECISION_THRESHOLD,
        early_stopping_min_epochs=EARLY_STOPPING_MIN_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
        run_outer_folds=args.training_mode == "nested-folds",
        price_encoder=(
            "none" if no_pretrained_model else args.price_encoder
        ),
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
    return results


def main() -> None:
    run_tft_pipeline(parse_args())


if __name__ == "__main__":
    main()
