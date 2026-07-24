"""Fine-tune Chronos-2 with LoRA or TopLoRA and raw-text decision fusion.

This runner performs a purged inner selection fit, calibrates the directional
threshold on validation data, refits a fresh adapter model on all labeled
training rows for the selected epoch count, evaluates the held-out test labels,
and writes one organizer-format submission.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from chronos2_lora_fusion import (
    ChronosContextStore,
    assemble_raw_text_indices,
    fit_lora_fusion,
    load_chronos2_lora_fusion,
    predict_lora_fusion,
    save_lora_fusion,
)
from latent_fusion import (
    _apply_covariate_scaler,
    _best_validation_threshold,
    _covariate_matrix,
    _fit_covariate_scaler,
    _purged_inner_positions,
    parquet_embedding_dim,
    prepare_raw_embedding_store,
)
from model_config import (
    ADAPTER_LEARNING_RATE_MULTIPLIER,
    CHRONOS2_ADAPTER_OUTPUT_DIRS,
    CHRONOS2_INPUT_COLUMNS,
    CHRONOS2_LORA_ALPHA,
    CHRONOS2_LORA_BATCH_SIZE,
    CHRONOS2_LORA_DROPOUT,
    CHRONOS2_LORA_EPOCHS,
    CHRONOS2_LORA_LEARNING_RATE,
    CHRONOS2_LORA_RANK,
    CHRONOS2_MODEL_ID,
    DATA_DIR,
    DEFAULT_TEXT_FAMILIES,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_MIN_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EXPECTED_SUBMISSION_ROWS,
    FUSION_DEPTH,
    FUSION_DROPOUT,
    FUSION_HIDDEN_DIM,
    HORIZON,
    LOOKBACK,
    MIN_CONTEXT,
    PREPARED_TEST_PATH,
    PREPARED_TRAIN_PATH,
    RANDOM_STATE,
    RAW_TEST_PATH,
    RAW_TEXT_DIM,
    RESIDUAL_EXPANSION,
    SUBMISSION_YEARS,
    TEST_LINK_PATH,
    TEST_TARGET_PATH,
    TEXT_ATTENTION_HEADS,
    TEXT_ATTENTION_LAYERS,
    TRAIN_LINK_PATH,
    TRAIN_TARGET_PATH,
)
from stage2_pretrained_forecasts import (
    classify_covariates,
    require_paths,
    select_device,
)
from utils import directional_classification_metrics, probability_to_price


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_TEXT_FAMILIES),
        help="Raw text-embedding families jointly adapted with Chronos-2.",
    )
    parser.add_argument(
        "--adapter-type",
        choices=("lora", "toplora"),
        default="lora",
        help="Standard LoRA or token-wise projected TopLoRA.",
    )
    parser.add_argument("--epochs", type=int, default=CHRONOS2_LORA_EPOCHS)
    parser.add_argument(
        "--batch-size", type=int, default=CHRONOS2_LORA_BATCH_SIZE
    )
    parser.add_argument("--lora-rank", type=int, default=CHRONOS2_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=CHRONOS2_LORA_ALPHA)
    parser.add_argument(
        "--lora-dropout", type=float, default=CHRONOS2_LORA_DROPOUT
    )
    parser.add_argument(
        "--lora-learning-rate",
        type=float,
        default=CHRONOS2_LORA_LEARNING_RATE,
        help="Learning rate for either LoRA or TopLoRA adapter parameters.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the text/fusion/decision layers.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the adapter-specific artifact directory.",
    )
    return parser.parse_args()


def _slice_indices(
    values: dict[str, np.ndarray], positions: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        family: np.ascontiguousarray(indices[positions])
        for family, indices in values.items()
    }


def _save_scaler(
    path: Path,
    columns: tuple[str, ...],
    scaler: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "covariate": list(columns),
        "imputation_median": scaler["median"],
        "standardization_mean": scaler["mean"],
        "standardization_scale": scaler["scale"],
    }).write_csv(path)


def _model_factory(
    args: argparse.Namespace,
    *,
    device: str,
    covariate_dim: int,
    family_dims: dict[str, int],
    field_count: int,
):
    return load_chronos2_lora_fusion(
        model_id=CHRONOS2_MODEL_ID,
        device=device,
        n_variates=len(CHRONOS2_INPUT_COLUMNS),
        covariate_dim=covariate_dim,
        family_dims=family_dims,
        field_count=field_count,
        hidden_dim=FUSION_HIDDEN_DIM,
        text_dim=RAW_TEXT_DIM,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fusion_depth=FUSION_DEPTH,
        expansion=RESIDUAL_EXPANSION,
        text_attention_heads=TEXT_ATTENTION_HEADS,
        text_attention_layers=TEXT_ATTENTION_LAYERS,
        dropout=FUSION_DROPOUT,
        adapter_type=args.adapter_type,
    )


def _release_model(model, device: str) -> None:
    model.cpu()
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0 or args.lora_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    args.families = list(dict.fromkeys(args.families))
    required_packages = (
        ("chronos", "peft")
        if args.adapter_type == "lora" else
        ("chronos",)
    )
    missing_packages = [
        package for package in required_packages
        if importlib.util.find_spec(package) is None
    ]
    if missing_packages:
        raise ImportError(
            "Chronos-2 adapter fusion is missing required packages: "
            f"{missing_packages}. Install `chronos-forecasting>=2.1.0`"
            + (" and `peft>=0.18.1`." if args.adapter_type == "lora" else ".")
        )
    require_paths([
        PREPARED_TRAIN_PATH,
        PREPARED_TEST_PATH,
        TRAIN_TARGET_PATH,
        TEST_TARGET_PATH,
        TRAIN_LINK_PATH,
        TEST_LINK_PATH,
        RAW_TEST_PATH,
    ])
    usable_families = []
    for family in args.families:
        try:
            parquet_embedding_dim(DATA_DIR / f"{family}_textemb.parquet")
            usable_families.append(family)
        except Exception as error:
            print(f"Warning: skipping unusable family {family!r}: {error}")
    if not usable_families:
        raise RuntimeError("No requested text-embedding family is usable")
    args.families = usable_families

    device = select_device()
    print(f"Device: {device}")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else CHRONOS2_ADAPTER_OUTPUT_DIRS[args.adapter_type]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_features = pl.read_parquet(PREPARED_TRAIN_PATH).sort(
        ["ticker", "date"]
    )
    test_features = pl.read_parquet(PREPARED_TEST_PATH).sort(
        ["ticker", "date"]
    )
    if train_features.columns != test_features.columns:
        raise ValueError("Train and test feature schemas differ")
    (
        past_market,
        _,
        _,
        model_covariates,
    ) = classify_covariates(train_features)
    missing_chronos = sorted(
        set(CHRONOS2_INPUT_COLUMNS) - set(past_market)
    )
    if missing_chronos:
        raise ValueError(f"Chronos-2 inputs are unavailable: {missing_chronos}")
    covariate_columns = tuple(model_covariates)

    train_targets = pl.read_parquet(TRAIN_TARGET_PATH).select([
        "row_id", "date", "ticker", "fwd_log_return_20", "target_up",
    ])
    test_targets = pl.read_parquet(TEST_TARGET_PATH).select([
        "row_id", "date", "ticker", "fwd_log_return_20", "target_up",
    ])
    train_links = pl.read_parquet(TRAIN_LINK_PATH)
    test_links = pl.read_parquet(TEST_LINK_PATH)
    text_fields = sorted(set(
        train_links["text_field"].drop_nulls().to_list()
    ) | set(
        test_links["text_field"].drop_nulls().to_list()
    ))
    raw_cache_dir = output_dir.parent / "raw_embedding_memmaps"
    stores = {
        family: prepare_raw_embedding_store(
            family=family,
            embedding_path=DATA_DIR / f"{family}_textemb.parquet",
            cache_dir=raw_cache_dir,
        )
        for family in args.families
    }
    family_dims = {
        family: store.input_dim for family, store in stores.items()
    }

    train_contexts = ChronosContextStore.from_features(
        train_features,
        CHRONOS2_INPUT_COLUMNS,
        horizon=HORIZON,
        lookback=LOOKBACK,
        min_context=MIN_CONTEXT,
    )
    test_contexts = ChronosContextStore.from_features(
        test_features,
        CHRONOS2_INPUT_COLUMNS,
        horizon=HORIZON,
        lookback=LOOKBACK,
        min_context=MIN_CONTEXT,
    )
    train_index = train_contexts.valid_index(
        train_targets.filter(pl.col("target_up").is_not_null())
    ).sort(["date", "ticker"])
    test_index = test_contexts.valid_index(test_targets).sort(["date", "ticker"])
    train_target = train_index["target_up"].to_numpy().astype(np.float32)
    train_raw_covariates = _covariate_matrix(
        train_index, train_features, covariate_columns
    )
    test_raw_covariates = _covariate_matrix(
        test_index, test_features, covariate_columns
    )
    train_text_indices = assemble_raw_text_indices(
        train_index, train_links, stores, text_fields
    )
    test_text_indices = assemble_raw_text_indices(
        test_index, test_links, stores, text_fields
    )
    no_train_text = np.logical_not(np.logical_or.reduce([
        indices >= 0 for indices in train_text_indices.values()
    ])).all(axis=1)
    no_test_text = np.logical_not(np.logical_or.reduce([
        indices >= 0 for indices in test_text_indices.values()
    ])).all(axis=1)
    if no_train_text.any() or no_test_text.any():
        raise ValueError(
            "Every modeled row must have at least one available text embedding; "
            f"missing train={int(no_train_text.sum())}, "
            f"test={int(no_test_text.sum())}"
        )

    inner_train_pos, validation_pos, split_manifest = _purged_inner_positions(
        train_index, HORIZON
    )
    inner_scaler = _fit_covariate_scaler(
        train_raw_covariates[inner_train_pos]
    )
    inner_train_covariates = _apply_covariate_scaler(
        train_raw_covariates[inner_train_pos], inner_scaler
    )
    validation_covariates = _apply_covariate_scaler(
        train_raw_covariates[validation_pos], inner_scaler
    )
    torch.manual_seed(RANDOM_STATE)
    selection_model = _model_factory(
        # Reproduce both Chronos-adapter and multimodal-head initialization.
        args,
        device=device,
        covariate_dim=len(covariate_columns),
        family_dims=family_dims,
        field_count=len(text_fields),
    )
    selection_model, history = fit_lora_fusion(
        selection_model,
        train_index=train_index[inner_train_pos],
        contexts=train_contexts,
        train_covariates=inner_train_covariates,
        train_text_indices=_slice_indices(
            train_text_indices, inner_train_pos
        ),
        stores=stores,
        train_target=train_target[inner_train_pos],
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_learning_rate=args.lora_learning_rate,
        adapter_learning_rate_multiplier=ADAPTER_LEARNING_RATE_MULTIPLIER,
        weight_decay=args.weight_decay,
        seed=RANDOM_STATE,
        validation_index=train_index[validation_pos],
        validation_covariates=validation_covariates,
        validation_text_indices=_slice_indices(
            train_text_indices, validation_pos
        ),
        validation_target=train_target[validation_pos],
        early_stopping_min_epochs=EARLY_STOPPING_MIN_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    validation_score = predict_lora_fusion(
        selection_model,
        index=train_index[validation_pos],
        contexts=train_contexts,
        covariates=validation_covariates,
        text_indices=_slice_indices(train_text_indices, validation_pos),
        stores=stores,
        device=device,
        batch_size=args.batch_size,
    )
    threshold = _best_validation_threshold(
        train_target[validation_pos].astype(np.int8), validation_score
    )
    best_epoch = int(history[0]["best_epoch"])
    pl.DataFrame(history).write_csv(output_dir / "selection_history.csv")
    _save_scaler(
        output_dir / "selection_covariate_scaler.csv",
        covariate_columns,
        inner_scaler,
    )
    selection_metadata = {
        "scope": "purged_inner_selection",
        "model_id": CHRONOS2_MODEL_ID,
        "adapter_type": args.adapter_type,
        "chronos_base_frozen": True,
        "chronos_trainable": (
            f"{args.adapter_type}_attention_adapters_only"
        ),
        "input_columns": list(CHRONOS2_INPUT_COLUMNS),
        "n_variates": len(CHRONOS2_INPUT_COLUMNS),
        "covariate_columns": list(covariate_columns),
        "covariate_dim": len(covariate_columns),
        "text_families": args.families,
        "text_fields": text_fields,
        "family_dims": family_dims,
        "field_count": len(text_fields),
        "hidden_dim": FUSION_HIDDEN_DIM,
        "text_dim": RAW_TEXT_DIM,
        "fusion_depth": FUSION_DEPTH,
        "expansion": RESIDUAL_EXPANSION,
        "fusion_dropout": FUSION_DROPOUT,
        "text_attention_heads": TEXT_ATTENTION_HEADS,
        "text_attention_layers": TEXT_ATTENTION_LAYERS,
        "best_epoch": best_epoch,
        "decision_threshold": threshold,
        "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_learning_rate": args.lora_learning_rate,
        "head_learning_rate": args.learning_rate,
        "split": split_manifest,
    }
    save_lora_fusion(
        selection_model, output_dir / "selection_model", selection_metadata
    )
    _release_model(selection_model, device)
    del selection_model
    gc.collect()

    # Refit a fresh adapter on all labeled data for the selected duration.
    full_scaler = _fit_covariate_scaler(train_raw_covariates)
    full_train_covariates = _apply_covariate_scaler(
        train_raw_covariates, full_scaler
    )
    scaled_test_covariates = _apply_covariate_scaler(
        test_raw_covariates, full_scaler
    )
    torch.manual_seed(RANDOM_STATE)
    final_model = _model_factory(
        args,
        device=device,
        covariate_dim=len(covariate_columns),
        family_dims=family_dims,
        field_count=len(text_fields),
    )
    final_model, final_history = fit_lora_fusion(
        final_model,
        train_index=train_index,
        contexts=train_contexts,
        train_covariates=full_train_covariates,
        train_text_indices=train_text_indices,
        stores=stores,
        train_target=train_target,
        device=device,
        epochs=best_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_learning_rate=args.lora_learning_rate,
        adapter_learning_rate_multiplier=ADAPTER_LEARNING_RATE_MULTIPLIER,
        weight_decay=args.weight_decay,
        seed=RANDOM_STATE,
    )
    test_score = predict_lora_fusion(
        final_model,
        index=test_index,
        contexts=test_contexts,
        covariates=scaled_test_covariates,
        text_indices=test_text_indices,
        stores=stores,
        device=device,
        batch_size=args.batch_size,
    )
    pl.DataFrame(final_history).write_csv(output_dir / "final_history.csv")
    _save_scaler(
        output_dir / "final_covariate_scaler.csv",
        covariate_columns,
        full_scaler,
    )
    final_metadata = {
        **selection_metadata,
        "scope": "all_labeled_training_refit",
        "epochs": best_epoch,
        "training_rows": train_index.height,
        "test_rows": test_index.height,
    }
    save_lora_fusion(final_model, output_dir / "final_model", final_metadata)

    predictions = (
        test_index.select([
            "row_id", "date", "ticker", "target_up",
        ])
        .with_columns(
            pl.Series("y_score", test_score),
            pl.Series(
                "y_pred", (test_score >= threshold).astype(np.int8)
            ),
            pl.lit(threshold).alias("decision_threshold"),
            pl.col("date").dt.add_business_days(HORIZON)
            .alias("prediction_date"),
        )
        .with_columns(
            pl.col("prediction_date").dt.year().alias("test_year")
        )
    )
    predictions.write_parquet(
        output_dir / "final_test_predictions.parquet", compression="zstd"
    )
    metric_rows = []
    for year in SUBMISSION_YEARS:
        scored = predictions.filter(
            (pl.col("test_year") == year)
            & pl.col("target_up").is_not_null()
        )
        metrics = {
            "test_year": int(year),
            "scored_rows": scored.height,
            "decision_threshold": threshold,
        }
        if scored.height:
            metrics.update(directional_classification_metrics(
                scored["target_up"].to_numpy(),
                scored["y_score"].to_numpy(),
                threshold,
            ))
        metric_rows.append(metrics)
    pl.DataFrame(metric_rows, infer_schema_length=None).write_csv(
        output_dir / "final_test_metrics_by_year.csv"
    )

    return_scale = train_targets.select(
        pl.col("fwd_log_return_20").drop_nulls().abs().median()
    ).item()
    if (
        return_scale is None
        or not np.isfinite(return_scale)
        or return_scale <= 0
    ):
        raise ValueError("Could not derive a positive submission return scale")
    source = predictions.filter(
        pl.col("test_year").is_in(SUBMISSION_YEARS)
    ).join(
        pl.read_parquet(RAW_TEST_PATH).select([
            "date", "ticker",
            pl.col("close").cast(pl.Float64).alias("origin_close"),
        ]).unique(["date", "ticker"]),
        on=["date", "ticker"],
        how="left",
        validate="m:1",
    )
    if source["origin_close"].null_count():
        raise ValueError("Submission rows are missing origin closing prices")
    close = probability_to_price(
        source["origin_close"].to_numpy(),
        source["y_score"].to_numpy(),
        float(return_scale),
        source["decision_threshold"].to_numpy(),
    )
    submission = (
        source.select(
            pl.concat_str([
                pl.col("ticker"),
                pl.lit("_"),
                pl.col("prediction_date").dt.strftime("%Y-%m-%d"),
            ]).alias("ID")
        )
        .with_columns(pl.Series("Close", close))
        .sort("ID")
    )
    if (
        submission.height != EXPECTED_SUBMISSION_ROWS
        or submission["ID"].n_unique() != EXPECTED_SUBMISSION_ROWS
    ):
        raise ValueError(
            f"Submission has {submission.height:,} rows and "
            f"{submission['ID'].n_unique():,} unique IDs; expected "
            f"{EXPECTED_SUBMISSION_ROWS:,}"
        )
    expected_per_year = EXPECTED_SUBMISSION_ROWS // len(SUBMISSION_YEARS)
    year_counts = dict(source.group_by("test_year").len().iter_rows())
    if any(
        year_counts.get(int(year), 0) != expected_per_year
        for year in SUBMISSION_YEARS
    ):
        raise ValueError(
            f"Submission prediction-year counts are invalid: {year_counts}"
        )
    if not np.isfinite(submission["Close"].to_numpy()).all() or (
        submission["Close"] <= 0
    ).any():
        raise ValueError("Submission contains an invalid closing price")
    submission.write_csv(
        output_dir / f"chronos2_{args.adapter_type}_text_submission.csv"
    )
    (output_dir / "run_config.json").write_text(json.dumps({
        **vars(args),
        "output_dir": str(output_dir),
        "device": device,
        "model_id": CHRONOS2_MODEL_ID,
    }, indent=2, default=str))
    _release_model(final_model, device)
    print(
        f"Saved Chronos-2 {args.adapter_type} fusion artifacts to {output_dir}"
    )


if __name__ == "__main__":
    main()
