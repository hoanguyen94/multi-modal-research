#!/usr/bin/env python3
"""Train and evaluate the walk-forward baseline models.

This is the executable Python equivalent of baseline_models.ipynb. It keeps
the notebook's model configurations, fold-local feature handling, threshold
selection, full-data refit, test evaluation, and submission generation.

Optional model families are used when their packages are installed.

Run from the repository root with:
    python baseline_models.py
"""


def main() -> None:
    # Core dependencies: numpy, polars, pyarrow, scikit-learn, and joblib.
    # Optional baselines: lightgbm, catboost, xgboost, and tabpfn.
    
    
    from pathlib import Path
    from importlib.util import find_spec
    import gc
    import json
    import warnings
    
    import numpy as np
    import polars as pl
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from utils import directional_classification_metrics, predicted_direction, probability_to_direction_score, probability_to_price
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    pl.Config.set_tbl_cols(100)
    
    DATA_DIR = Path("data")
    RAW_TEST_PATH = DATA_DIR / "test.parquet"
    EMBEDDING_EXPORT = "pca_embeddings"  # no_embeddings | pca_embeddings | original_embeddings
    FEATURE_REPRESENTATION = "article_mean_embeddings"  # without_embeddings | article_mean_embeddings
    ARTIFACT_DIR = DATA_DIR / "model_artifacts" / EMBEDDING_EXPORT
    RANDOM_STATE = 42
    TABPFN_MAX_TRAIN_ROWS = None  # None uses all fold/full-training rows; set an integer only for a capped experiment
    TABPFN_PREDICT_BATCH_SIZE = 2_048
    FORECAST_HORIZON_WEEKDAYS = 20
    SUBMISSION_YEARS = (2022, 2023)
    EXPECTED_SUBMISSION_ROWS = 52_000
    DECISION_THRESHOLDS = tuple(np.round(np.arange(0.35, 0.651, 0.05), 2))
    SELECTION_METRIC = "tuned_mean_balanced_accuracy"
    SELECTION_STD_METRIC = "tuned_std_balanced_accuracy"
    REFIT_ALL_MODEL_FAMILIES = True  # False refits only the overall CV winner
    SAVE_RESULTS = True
    
    if FEATURE_REPRESENTATION == "without_embeddings":
        FEATURE_TAG = "without_embeddings"
        preferred_feature_path = ARTIFACT_DIR / "train_features_without_embeddings.parquet"
        preferred_test_feature_path = ARTIFACT_DIR / "test_features_without_embeddings.parquet"
        legacy_feature_path = ARTIFACT_DIR / "train_features.parquet"
        FEATURE_PATH = preferred_feature_path if preferred_feature_path.exists() else legacy_feature_path
        TEST_FEATURE_PATH = preferred_test_feature_path
    elif FEATURE_REPRESENTATION == "article_mean_embeddings":
        if EMBEDDING_EXPORT == "no_embeddings":
            raise ValueError("Select pca_embeddings or original_embeddings for article-mean features.")
        FEATURE_TAG = f"article_mean_{EMBEDDING_EXPORT}"
        FEATURE_PATH = ARTIFACT_DIR / f"train_features_with_article_mean_{EMBEDDING_EXPORT}.parquet"
        TEST_FEATURE_PATH = ARTIFACT_DIR / f"test_features_with_article_mean_{EMBEDDING_EXPORT}.parquet"
    else:
        raise ValueError("Unknown FEATURE_REPRESENTATION.")
    
    TARGET_PATH = ARTIFACT_DIR / "train_target.parquet"
    TEST_TARGET_PATH = ARTIFACT_DIR / "test_target.parquet"
    FOLD_PATH = ARTIFACT_DIR / "walk_forward_assignments.parquet"
    OUTPUT_DIR = ARTIFACT_DIR / "baselines" / FEATURE_TAG
    print(f"Training feature input: {FEATURE_PATH}")
    print(f"Test feature input: {TEST_FEATURE_PATH}")
    
    
    required_paths = [FEATURE_PATH, TARGET_PATH, TEST_TARGET_PATH, FOLD_PATH, RAW_TEST_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing modeling artifacts. Run data_analysis.ipynb with the matching EMBEDDING_EXPORT first: "
            + ", ".join(missing_paths)
        )
    
    feature_frame = pl.read_parquet(FEATURE_PATH)
    target_frame = pl.read_parquet(TARGET_PATH)
    fold_assignments = pl.read_parquet(FOLD_PATH)
    
    if "row_id" not in feature_frame.columns:
        feature_frame = feature_frame.with_row_index("row_id")
    modeling_frame = feature_frame.join(
        target_frame.select(["row_id", "target_up"]), on="row_id", how="inner"
    ).filter(pl.col("target_up").is_not_null())
    NON_FEATURE_COLUMNS = {"row_id", "date", "ticker", "target_up", "fwd_log_return_20"}
    MODEL_FEATURE_COLUMNS = [
        column for column, dtype in modeling_frame.schema.items()
        if dtype.is_numeric() and column not in NON_FEATURE_COLUMNS
    ]
    FOLD_NUMBERS = fold_assignments["fold"].unique().sort().to_list()
    FOLD_FEATURE_PATHS = {}
    if FEATURE_REPRESENTATION == "article_mean_embeddings":
        for fold_number in FOLD_NUMBERS:
            fold_dir = ARTIFACT_DIR / "embeddings" / f"fold_{fold_number}"
            FOLD_FEATURE_PATHS[fold_number] = {
                "train": fold_dir / f"train_features_with_article_mean_{EMBEDDING_EXPORT}.parquet",
                "validation": fold_dir / f"validation_features_with_article_mean_{EMBEDDING_EXPORT}.parquet",
            }
        missing_fold_paths = [
            str(path) for paths in FOLD_FEATURE_PATHS.values() for path in paths.values() if not path.exists()
        ]
        if missing_fold_paths:
            raise FileNotFoundError(
                "Fold-specific embedding features are required to prevent PCA leakage: " + ", ".join(missing_fold_paths)
            )
    print(
        f"Loaded {modeling_frame.height:,} labeled rows, {len(MODEL_FEATURE_COLUMNS):,} features, "
        f"and {len(FOLD_NUMBERS)} walk-forward folds."
    )
    
    
    OPTIONAL_PACKAGES = {"lightgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost", "tabpfn": "tabpfn"}
    package_available = {model: find_spec(package) is not None for model, package in OPTIONAL_PACKAGES.items()}
    print(pl.DataFrame({"model": list(package_available), "installed": list(package_available.values())}))
    
    # Compact, curated configurations focus on capacity, shrinkage, and regularization.
    # Tree count and learning rate are coupled instead of sampled independently.
    RULE_CONFIGS = {"always_up": [{}], "momentum_20": [{}], "reversal_20": [{}]}
    LOGISTIC_CONFIGS = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    TREE_CONFIGS = {
        "lightgbm": [
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
            {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
            {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 50, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 5.0},
            {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0},
        ],
        "catboost": [
            {"iterations": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 1.0},
            {"iterations": 600, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5.0, "random_strength": 1.0, "bagging_temperature": 1.0},
            {"iterations": 400, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5.0, "random_strength": 1.0, "bagging_temperature": 1.0},
            {"iterations": 500, "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 10.0, "random_strength": 1.0, "bagging_temperature": 1.0},
        ],
        "xgboost": [
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5.0, "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
            {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 10.0, "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
            {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 10.0, "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0},
            {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 20.0, "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 10.0},
        ],
    }
    LEARNED_CONFIGS = {
        "logistic_l2": LOGISTIC_CONFIGS,
        **TREE_CONFIGS,
        # One compact setting limits TabPFN's comparatively expensive repeated CV inference.
        "tabpfn": [{"n_estimators": 4}],
    }
    
    def make_learned_baseline(name, params, random_state):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        if name == "logistic_l2":
            classifier = LogisticRegression(
                penalty="l2", C=params["C"], solver="lbfgs", max_iter=2_000, random_state=random_state
            )
            return Pipeline([("impute", imputer), ("scale", StandardScaler()), ("classifier", classifier)])
        if name == "lightgbm":
            from lightgbm import LGBMClassifier
            classifier = LGBMClassifier(**{
                "random_state": random_state, "n_jobs": -1, "verbosity": -1, "subsample_freq": 1, **params
            })
        elif name == "catboost":
            from catboost import CatBoostClassifier
            classifier = CatBoostClassifier(
                **params, loss_function="Logloss", random_seed=random_state, verbose=False, allow_writing_files=False
            )
        elif name == "xgboost":
            from xgboost import XGBClassifier
            classifier = XGBClassifier(**{
                "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                "random_state": random_state, "n_jobs": -1, **params
            })
        elif name == "tabpfn":
            from tabpfn import TabPFNClassifier
            classifier = TabPFNClassifier(device="auto", **params)
        else:
            raise ValueError(f"Unknown learned baseline: {name}")
        return Pipeline([("impute", imputer), ("classifier", classifier)])
    
    def config_id(model_name, position):
        return f"{model_name}__{position:02d}"
    
    def rows_for_fit(model_name, matrix, target):
        if model_name == "tabpfn" and TABPFN_MAX_TRAIN_ROWS and len(matrix) > TABPFN_MAX_TRAIN_ROWS:
            return matrix[-TABPFN_MAX_TRAIN_ROWS:], target[-TABPFN_MAX_TRAIN_ROWS:]
        return matrix, target
    
    def positive_probability(model, matrix, batch_size=None):
        batches = [matrix] if batch_size is None else [matrix[start:start + batch_size] for start in range(0, len(matrix), batch_size)]
        positive_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
        return np.concatenate([np.asarray(model.predict_proba(batch))[:, positive_index] for batch in batches]).astype(np.float32)
    
    def score_candidate(model_name, candidate_id, params_json, fold_number, y_true, y_score, train_rows_used, decision_threshold=0.5):
        y_score = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)
        y_pred = predicted_direction(y_score, decision_threshold)
        shared_metrics = directional_classification_metrics(y_true, y_score, decision_threshold)
        metrics = {
            "feature_set": FEATURE_TAG, "model": model_name, "config_id": candidate_id,
            "params_json": params_json, "decision_threshold": float(decision_threshold), "fold": fold_number,
            "train_rows_used": int(train_rows_used), "validation_rows": len(y_true),
            **shared_metrics,
        }
        return metrics, y_pred
    
    
    metric_rows, prediction_parts = [], []
    status_rows = [
        {"model": model, "config_id": None, "fold": None, "status": "available" if available else "skipped",
         "detail": "installed" if available else f"install {OPTIONAL_PACKAGES[model]}"}
        for model, available in package_available.items()
    ]
    candidate_parameters = {}
    failed_candidates = set()
    
    enabled_learned_configs = {"logistic_l2": LEARNED_CONFIGS["logistic_l2"]}
    enabled_learned_configs.update({
        name: configs for name, configs in LEARNED_CONFIGS.items()
        if name != "logistic_l2" and package_available.get(name, False)
    })
    for model_name, configs in {**RULE_CONFIGS, **enabled_learned_configs}.items():
        for position, params in enumerate(configs):
            candidate_parameters[(model_name, config_id(model_name, position))] = params
    
    for fold_number in FOLD_NUMBERS:
        train_ids = fold_assignments.filter((pl.col("fold") == fold_number) & (pl.col("split") == "train"))["row_id"]
        validation_ids = fold_assignments.filter((pl.col("fold") == fold_number) & (pl.col("split") == "validation"))["row_id"]
        if FEATURE_REPRESENTATION == "article_mean_embeddings":
            train_fold = pl.read_parquet(FOLD_FEATURE_PATHS[fold_number]["train"]).join(
                target_frame.select(["row_id", "target_up"]), on="row_id", how="inner"
            ).filter(pl.col("target_up").is_not_null()).sort(["date", "ticker"])
            validation_fold = pl.read_parquet(FOLD_FEATURE_PATHS[fold_number]["validation"]).join(
                target_frame.select(["row_id", "target_up"]), on="row_id", how="inner"
            ).filter(pl.col("target_up").is_not_null()).sort(["date", "ticker"])
            missing_fold_columns = [
                column for column in MODEL_FEATURE_COLUMNS
                if column not in train_fold.columns or column not in validation_fold.columns
            ]
            if missing_fold_columns:
                raise ValueError(f"Fold {fold_number} is missing {len(missing_fold_columns)} model features.")
        else:
            train_fold = modeling_frame.join(train_ids.to_frame(), on="row_id", how="semi").sort(["date", "ticker"])
            validation_fold = modeling_frame.join(validation_ids.to_frame(), on="row_id", how="semi").sort(["date", "ticker"])
        X_train = train_fold.select(MODEL_FEATURE_COLUMNS).to_numpy().astype(np.float32, copy=False)
        X_validation = validation_fold.select(MODEL_FEATURE_COLUMNS).to_numpy().astype(np.float32, copy=False)
        y_train = train_fold["target_up"].to_numpy().astype(np.int8)
        y_validation = validation_fold["target_up"].to_numpy().astype(np.int8)
        identity = validation_fold.select(["row_id", "date", "ticker"])
    
        return_20 = validation_fold["ret_20"].fill_null(0.0).to_numpy()
        rule_scores = {
            "always_up": np.ones(len(validation_fold), dtype=np.float32),
            "momentum_20": (return_20 > 0).astype(np.float32),
            "reversal_20": (return_20 < 0).astype(np.float32),
        }
        for model_name, y_score in rule_scores.items():
            candidate_id = config_id(model_name, 0)
            params_json = "{}"
            metrics, y_pred = score_candidate(
                model_name, candidate_id, params_json, fold_number, y_validation, y_score, len(train_fold)
            )
            metric_rows.append(metrics)
            prediction_parts.append(identity.with_columns(
                pl.lit(FEATURE_TAG).alias("feature_set"), pl.lit(fold_number).cast(pl.Int8).alias("fold"),
                pl.lit(model_name).alias("model"), pl.lit(candidate_id).alias("config_id"),
                pl.Series("y_true", y_validation), pl.Series("y_score", y_score), pl.Series("y_pred", y_pred),
            ))
    
        for model_name, configs in enabled_learned_configs.items():
            for position, params in enumerate(configs):
                candidate_id = config_id(model_name, position)
                candidate_key = (model_name, candidate_id)
                if candidate_key in failed_candidates:
                    continue
                params_json = json.dumps(params, sort_keys=True)
                try:
                    X_fit, y_fit = rows_for_fit(model_name, X_train, y_train)
                    fitted_model = make_learned_baseline(model_name, params, RANDOM_STATE + fold_number)
                    fitted_model.fit(X_fit, y_fit)
                    batch_size = TABPFN_PREDICT_BATCH_SIZE if model_name == "tabpfn" else None
                    y_score = positive_probability(fitted_model, X_validation, batch_size)
                    metrics, y_pred = score_candidate(
                        model_name, candidate_id, params_json, fold_number, y_validation, y_score, len(X_fit)
                    )
                    metric_rows.append(metrics)
                    prediction_parts.append(identity.with_columns(
                        pl.lit(FEATURE_TAG).alias("feature_set"), pl.lit(fold_number).cast(pl.Int8).alias("fold"),
                        pl.lit(model_name).alias("model"), pl.lit(candidate_id).alias("config_id"),
                        pl.Series("y_true", y_validation), pl.Series("y_score", y_score), pl.Series("y_pred", y_pred),
                    ))
                except Exception as error:
                    failed_candidates.add(candidate_key)
                    status_rows.append({
                        "model": model_name, "config_id": candidate_id, "fold": fold_number, "status": "failed",
                        "detail": f"{type(error).__name__}: {error}"
                    })
                    print(f"Skipping failed candidate {candidate_id}: {error}")
                finally:
                    if "fitted_model" in locals():
                        del fitted_model
                    gc.collect()
    
    tuning_fold_metrics = pl.DataFrame(metric_rows).sort(["model", "config_id", "fold"])
    fold_metric_summaries = {
        fold_number: tuning_fold_metrics.filter(pl.col("fold") == fold_number)
        for fold_number in FOLD_NUMBERS
    }
    aggregate_fold_metrics = pl.concat(
        [fold_metric_summaries[fold_number] for fold_number in FOLD_NUMBERS],
        how="vertical_relaxed",
    ).sort(["model", "config_id", "fold"])
    if aggregate_fold_metrics.height != tuning_fold_metrics.height:
        raise RuntimeError("Fold summaries do not cover every tuning metric row.")
    tuning_predictions = pl.concat(prediction_parts).select(
        ["feature_set", "model", "config_id", "fold", "row_id", "date", "ticker", "y_true", "y_score", "y_pred"]
    ).sort(["model", "config_id", "fold", "date", "ticker"])
    tuning_status = pl.DataFrame(status_rows, infer_schema_length=None)
    
    # Tune the classification threshold from OOF predictions only; the test set is never consulted.
    threshold_rows = []
    for candidate in tuning_predictions.select(["model", "config_id"]).unique().iter_rows(named=True):
        model_name, candidate_id = candidate["model"], candidate["config_id"]
        candidate_predictions = tuning_predictions.filter(
            (pl.col("model") == model_name) & (pl.col("config_id") == candidate_id)
        )
        thresholds = (0.5,) if model_name in RULE_CONFIGS else DECISION_THRESHOLDS
        for threshold in thresholds:
            fold_accuracies, fold_balanced_accuracies = [], []
            for fold_number in candidate_predictions["fold"].unique().sort().to_list():
                fold_predictions = candidate_predictions.filter(pl.col("fold") == fold_number)
                y_true_fold = fold_predictions["y_true"].to_numpy().astype(np.int8)
                y_pred_fold = (fold_predictions["y_score"].to_numpy() >= threshold).astype(np.int8)
                fold_accuracies.append(accuracy_score(y_true_fold, y_pred_fold))
                fold_balanced_accuracies.append(balanced_accuracy_score(y_true_fold, y_pred_fold))
            threshold_rows.append({
                "feature_set": FEATURE_TAG, "model": model_name, "config_id": candidate_id,
                "decision_threshold": float(threshold),
                "threshold_distance_from_half": abs(float(threshold) - 0.5),
                "tuned_mean_accuracy": float(np.mean(fold_accuracies)),
                "tuned_std_accuracy": float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0,
                "tuned_mean_balanced_accuracy": float(np.mean(fold_balanced_accuracies)),
                "tuned_std_balanced_accuracy": float(np.std(fold_balanced_accuracies, ddof=1)) if len(fold_balanced_accuracies) > 1 else 0.0,
            })
    threshold_summary = pl.DataFrame(threshold_rows).sort(
        ["model", "config_id", SELECTION_METRIC, SELECTION_STD_METRIC, "threshold_distance_from_half"],
        descending=[False, False, True, False, False],
    )
    best_threshold_by_config = threshold_summary.group_by(
        ["feature_set", "model", "config_id"], maintain_order=True
    ).first()
    
    tuning_summary = aggregate_fold_metrics.group_by(["feature_set", "model", "config_id", "params_json"]).agg(
        pl.len().alias("completed_folds"),
        pl.col("accuracy").mean().alias("mean_accuracy"),
        pl.col("accuracy").std().alias("std_accuracy"),
        pl.col("balanced_accuracy").mean().alias("mean_balanced_accuracy"),
        pl.col("balanced_accuracy").std().alias("std_balanced_accuracy"),
        pl.col("roc_auc").mean().alias("mean_roc_auc"),
        pl.col("roc_auc").std().alias("std_roc_auc"),
        pl.col("brier_score").mean().alias("mean_brier_score"),
    ).join(
        best_threshold_by_config.drop("threshold_distance_from_half"),
        on=["feature_set", "model", "config_id"], how="left",
    ).sort(["model", SELECTION_METRIC], descending=[False, True])
    
    # This is the cross-fold table used for model selection; incomplete candidates remain visible
    # in the saved aggregate but cannot be selected.
    aggregate_model_selection_summary = tuning_summary
    complete_candidates = aggregate_model_selection_summary.filter(
        pl.col("completed_folds") == len(FOLD_NUMBERS)
    )
    if complete_candidates.is_empty():
        raise RuntimeError("No candidate completed every walk-forward fold. Inspect tuning_status.")
    best_by_model = (
        complete_candidates.sort(
            ["model", SELECTION_METRIC, SELECTION_STD_METRIC, "config_id"],
            descending=[False, True, False, False],
        ).group_by("model", maintain_order=True).first()
    )
    selected_overall = best_by_model.sort(
        [SELECTION_METRIC, SELECTION_STD_METRIC, "model"], descending=[True, False, False]
    ).head(1).with_columns(pl.lit(True).alias("selected_overall"))
    best_by_model = best_by_model.join(
        selected_overall.select(["model", "config_id", "selected_overall"]),
        on=["model", "config_id"], how="left"
    ).with_columns(pl.col("selected_overall").fill_null(False))
    selected_config_predictions = tuning_predictions.join(
        best_by_model.select(["model", "config_id", "decision_threshold"]),
        on=["model", "config_id"], how="inner"
    ).with_columns(
        (pl.col("y_score") >= pl.col("decision_threshold")).cast(pl.Int8).alias("y_pred")
    )
    selected_fold_metric_rows = []
    for selected in best_by_model.iter_rows(named=True):
        model_name, candidate_id = selected["model"], selected["config_id"]
        threshold = float(selected["decision_threshold"])
        for fold_number in FOLD_NUMBERS:
            fold_predictions = selected_config_predictions.filter(
                (pl.col("model") == model_name) & (pl.col("config_id") == candidate_id)
                & (pl.col("fold") == fold_number)
            )
            if fold_predictions.is_empty():
                continue
            original_metric = tuning_fold_metrics.filter(
                (pl.col("model") == model_name) & (pl.col("config_id") == candidate_id)
                & (pl.col("fold") == fold_number)
            ).row(0, named=True)
            metrics, _ = score_candidate(
                model_name, candidate_id, selected["params_json"], fold_number,
                fold_predictions["y_true"].to_numpy(), fold_predictions["y_score"].to_numpy(),
                original_metric["train_rows_used"], decision_threshold=threshold,
            )
            selected_fold_metric_rows.append(metrics)
    selected_config_fold_metrics = pl.DataFrame(selected_fold_metric_rows).sort(["model", "fold"])
    
    print(tuning_summary)
    print(selected_config_fold_metrics.sort(["model", "fold"]))
    print(best_by_model.sort(SELECTION_METRIC, descending=True))
    print(tuning_status)
    
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FITTED_MODEL_DIR = OUTPUT_DIR / "fitted_models"
    if SAVE_RESULTS:
        FITTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    full_training = modeling_frame.sort(["date", "ticker"])
    X_full = full_training.select(MODEL_FEATURE_COLUMNS).to_numpy().astype(np.float32, copy=False)
    y_full = full_training["target_up"].to_numpy().astype(np.int8)
    
    test_feature_frame = pl.read_parquet(TEST_FEATURE_PATH) if TEST_FEATURE_PATH.exists() else None
    if test_feature_frame is not None and "row_id" not in test_feature_frame.columns:
        test_feature_frame = test_feature_frame.with_row_index("row_id")
    if test_feature_frame is not None:
        missing_test_columns = [column for column in MODEL_FEATURE_COLUMNS if column not in test_feature_frame.columns]
        if missing_test_columns:
            raise ValueError(f"Test feature file is missing {len(missing_test_columns)} training features.")
        X_test = test_feature_frame.select(MODEL_FEATURE_COLUMNS).to_numpy().astype(np.float32, copy=False)
        test_identity = test_feature_frame.select(["row_id", "date", "ticker"])
    else:
        X_test, test_identity = None, None
        print(f"No test feature file found at {TEST_FEATURE_PATH}; final models will still be saved.")
    
    rows_to_refit = best_by_model if REFIT_ALL_MODEL_FAMILIES else selected_overall
    final_refit_rows, final_prediction_parts = [], []
    for selected in rows_to_refit.iter_rows(named=True):
        model_name = selected["model"]
        candidate_id = selected["config_id"]
        params = candidate_parameters[(model_name, candidate_id)]
        decision_threshold = float(selected["decision_threshold"])
        is_rule = model_name in RULE_CONFIGS
        model_path = None
        train_rows_used = len(X_full)
    
        if is_rule:
            if test_feature_frame is not None:
                test_return_20 = test_feature_frame["ret_20"].fill_null(0.0).to_numpy()
                if model_name == "always_up":
                    test_score = np.ones(len(test_feature_frame), dtype=np.float32)
                elif model_name == "momentum_20":
                    test_score = (test_return_20 > 0).astype(np.float32)
                else:
                    test_score = (test_return_20 < 0).astype(np.float32)
        else:
            X_fit, y_fit = rows_for_fit(model_name, X_full, y_full)
            train_rows_used = len(X_fit)
            final_model = make_learned_baseline(model_name, params, RANDOM_STATE)
            final_model.fit(X_fit, y_fit)
            if SAVE_RESULTS:
                model_path = FITTED_MODEL_DIR / f"{model_name}_{candidate_id}_{FEATURE_TAG}.joblib"
                joblib.dump({
                    "pipeline": final_model, "feature_columns": MODEL_FEATURE_COLUMNS,
                    "model": model_name, "config_id": candidate_id, "params": params,
                    "decision_threshold": decision_threshold,
                    "feature_set": FEATURE_TAG, "training_rows": train_rows_used,
                }, model_path)
            if X_test is not None:
                batch_size = TABPFN_PREDICT_BATCH_SIZE if model_name == "tabpfn" else None
                test_score = positive_probability(final_model, X_test, batch_size)
    
        final_refit_rows.append({
            "feature_set": FEATURE_TAG, "model": model_name, "config_id": candidate_id,
            "params_json": json.dumps(params, sort_keys=True), "decision_threshold": decision_threshold,
            "is_rule": is_rule,
            "selected_overall": bool(selected.get("selected_overall", False)),
            "full_labeled_rows": len(X_full), "train_rows_used": train_rows_used,
            "model_path": str(model_path) if model_path else None,
        })
        if test_identity is not None:
            final_prediction_parts.append(test_identity.with_columns(
                pl.lit(FEATURE_TAG).alias("feature_set"), pl.lit(model_name).alias("model"),
                pl.lit(candidate_id).alias("config_id"),
                pl.lit(decision_threshold).alias("decision_threshold"), pl.Series("y_score", test_score),
                pl.Series("y_pred", (test_score >= decision_threshold).astype(np.int8)),
            ))
        if not is_rule:
            del final_model
        gc.collect()
    
    final_refit_summary = pl.DataFrame(final_refit_rows, infer_schema_length=None)
    final_test_predictions = pl.concat(final_prediction_parts) if final_prediction_parts else pl.DataFrame()
    test_metric_rows = []
    if not final_test_predictions.is_empty():
        test_targets = pl.read_parquet(TEST_TARGET_PATH).select(
            ["row_id", pl.col("target_up").cast(pl.Int8).alias("y_true")]
        )
        final_test_predictions = final_test_predictions.join(test_targets, on="row_id", how="left").with_columns(
            pl.col("y_true").is_not_null().alias("has_test_label"),
            pl.col("date").dt.add_business_days(FORECAST_HORIZON_WEEKDAYS).alias("prediction_date"),
        ).with_columns(
            pl.col("prediction_date").dt.year().alias("test_year")
        )
        for selected in rows_to_refit.iter_rows(named=True):
            model_name = selected["model"]
            candidate_id = selected["config_id"]
            model_predictions = final_test_predictions.filter(
                (pl.col("model") == model_name)
                & (pl.col("config_id") == candidate_id)
            )
            for test_year in SUBMISSION_YEARS:
                year_predictions = model_predictions.filter(pl.col("test_year") == test_year)
                scored = year_predictions.filter(pl.col("has_test_label"))
                if scored.is_empty():
                    continue
                y_test_true = scored["y_true"].to_numpy().astype(np.int8)
                y_test_score = scored["y_score"].to_numpy().astype(np.float64)
                shared_test_metrics = directional_classification_metrics(
                    y_test_true, y_test_score, float(selected["decision_threshold"])
                )
                test_metric_rows.append({
                    "feature_set": FEATURE_TAG, "model": model_name, "config_id": candidate_id,
                    "test_year": int(test_year),
                    "decision_threshold": float(selected["decision_threshold"]),
                    "selected_overall": bool(selected.get("selected_overall", False)),
                    "test_rows_total": year_predictions.height, "test_rows_scored": scored.height,
                    "test_label_coverage": scored.height / year_predictions.height,
                    "test_hit_rate": shared_test_metrics["hit_rate"],
                    "test_accuracy": shared_test_metrics["accuracy"],
                    "test_balanced_accuracy": shared_test_metrics["balanced_accuracy"],
                    "test_roc_auc": shared_test_metrics["roc_auc"],
                    "test_brier_score": shared_test_metrics["brier_score"],
                })
    final_test_metrics = pl.DataFrame(test_metric_rows, infer_schema_length=None) if test_metric_rows else pl.DataFrame()
    
    # Kaggle submissions require target-date closing prices for every weekday in 2022-2023.
    return_column = f"fwd_log_return_{FORECAST_HORIZON_WEEKDAYS}"
    submission_return_scale = target_frame.select(pl.col(return_column).abs().median()).item()
    if submission_return_scale is None or not np.isfinite(submission_return_scale) or submission_return_scale <= 0:
        raise ValueError("Could not estimate a positive training-only return scale for submission prices.")
    raw_test_close = pl.read_parquet(RAW_TEST_PATH, columns=["date", "ticker", "close"]).with_columns(
        pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64).alias("origin_close")
    ).drop("close")
    submission_source = final_test_predictions.join(
        raw_test_close, on=["date", "ticker"], how="left", validate="m:1"
    ).with_columns(
        pl.col("date").dt.add_business_days(FORECAST_HORIZON_WEEKDAYS).alias("prediction_date")
    ).filter(
        pl.col("prediction_date").dt.year().is_in(SUBMISSION_YEARS)
    )
    submission_direction_score = probability_to_direction_score(
        submission_source["y_score"].to_numpy(), submission_source["decision_threshold"].to_numpy()
    )
    submission_close = probability_to_price(
        submission_source["origin_close"].to_numpy(), submission_source["y_score"].to_numpy(),
        float(submission_return_scale), submission_source["decision_threshold"].to_numpy(),
    )
    submission_source = submission_source.with_columns(
        pl.Series("submission_direction_score", submission_direction_score),
        pl.concat_str([pl.col("ticker"), pl.lit("_"), pl.col("prediction_date").dt.strftime("%Y-%m-%d")]).alias("ID"),
        pl.Series("Close", submission_close),
    )
    submission_tables, submission_manifest_rows = {}, []
    expected_rows_per_submission_year = EXPECTED_SUBMISSION_ROWS // len(SUBMISSION_YEARS)
    for selected in rows_to_refit.iter_rows(named=True):
        model_name, candidate_id = selected["model"], selected["config_id"]
        model_submission_source = submission_source.filter(
            (pl.col("model") == model_name) & (pl.col("config_id") == candidate_id)
        )
        submission_rows_by_year = {
            int(year): model_submission_source.filter(
                pl.col("prediction_date").dt.year() == year
            ).height
            for year in SUBMISSION_YEARS
        }
        if any(rows != expected_rows_per_submission_year for rows in submission_rows_by_year.values()):
            raise ValueError(
                f"{model_name} submission rows by year are {submission_rows_by_year}; "
                f"expected {expected_rows_per_submission_year:,} for each year."
            )
        submission = model_submission_source.select(["ID", "Close"]).sort("ID")
        if submission.height != EXPECTED_SUBMISSION_ROWS:
            raise ValueError(f"{model_name} submission has {submission.height:,} rows; expected {EXPECTED_SUBMISSION_ROWS:,}.")
        if submission["ID"].n_unique() != EXPECTED_SUBMISSION_ROWS:
            raise ValueError(f"{model_name} submission contains duplicate IDs.")
        invalid_close = submission.select(
            (pl.col("Close").is_null() | pl.col("Close").is_nan() | pl.col("Close").is_infinite() | (pl.col("Close") <= 0)).any()
        ).item()
        if invalid_close:
            raise ValueError(f"{model_name} submission contains a missing, non-finite, or non-positive Close.")
        submission_key = f"{model_name}_{candidate_id}_{FEATURE_TAG}"
        submission_tables[submission_key] = submission
        submission_manifest_rows.append({
            "model": model_name, "config_id": candidate_id,
            "selected_overall": bool(selected.get("selected_overall", False)),
            "rows": submission.height, "unique_ids": submission["ID"].n_unique(),
            **{f"rows_{year}": rows for year, rows in submission_rows_by_year.items()},
            "first_id": submission["ID"].min(), "last_id": submission["ID"].max(),
            "return_scale": float(submission_return_scale),
            "filename": f"submission_{submission_key}.csv",
        })
    submission_manifest = pl.DataFrame(submission_manifest_rows, infer_schema_length=None)
    print(final_refit_summary)
    if not final_test_metrics.is_empty():
        print(final_test_metrics.sort(
            ["test_year", "test_balanced_accuracy"], descending=[False, True]
        ))
    else:
        print("Test predictions were produced, but no non-null test labels were available for scoring.")
    print(submission_manifest)
    
    
    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_tables = {
            "tuning_fold_metrics": tuning_fold_metrics,
            "tuning_summary": tuning_summary,
            "aggregate_model_selection_summary": aggregate_model_selection_summary,
            "threshold_tuning_summary": threshold_summary,
            "selected_hyperparameters": best_by_model,
            "selected_config_fold_metrics": selected_config_fold_metrics,
            "tuning_status": tuning_status,
            "final_refit_summary": final_refit_summary,
            "submission_manifest": submission_manifest,
        }
        if not final_test_metrics.is_empty():
            summary_tables["final_test_metrics"] = final_test_metrics
        for stem, table in summary_tables.items():
            # table.write_parquet(OUTPUT_DIR / f"{stem}_{FEATURE_TAG}.parquet", compression="zstd")
            table.write_csv(OUTPUT_DIR / f"{stem}_{FEATURE_TAG}.csv")
    
        # Keep an auditable report inside every walk-forward fold directory.
        FOLD_RESULT_DIR = OUTPUT_DIR / "fold_results"
        fold_manifest_rows = []
        saved_fold_metric_rows = 0
        for fold_number in FOLD_NUMBERS:
            fold_dir = FOLD_RESULT_DIR / f"fold_{fold_number}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_candidate_metrics = fold_metric_summaries[fold_number]
            fold_selected_metrics = selected_config_fold_metrics.filter(pl.col("fold") == fold_number)
            fold_candidate_metrics.write_csv(
                fold_dir / f"candidate_metrics_{FEATURE_TAG}.csv"
            )
            fold_selected_metrics.write_csv(
                fold_dir / f"selected_model_metrics_{FEATURE_TAG}.csv"
            )
            saved_fold_metric_rows += fold_candidate_metrics.height
            fold_manifest_rows.append({
                "fold": int(fold_number),
                "candidate_metric_rows": fold_candidate_metrics.height,
                "selected_model_rows": fold_selected_metrics.height,
                "validation_rows": int(fold_candidate_metrics["validation_rows"].max()),
                "directory": str(fold_dir),
            })
        if saved_fold_metric_rows != tuning_fold_metrics.height:
            raise RuntimeError("Per-fold metric files do not cover every aggregate fold-metric row.")
        fold_result_manifest = pl.DataFrame(fold_manifest_rows).sort("fold")
        fold_result_manifest.write_csv(OUTPUT_DIR / f"fold_result_manifest_{FEATURE_TAG}.csv")
        selected_config_predictions.write_parquet(OUTPUT_DIR / f"selected_config_oof_predictions_{FEATURE_TAG}.parquet", compression="zstd")
        if not final_test_predictions.is_empty():
            final_test_predictions.write_parquet(OUTPUT_DIR / f"final_test_predictions_{FEATURE_TAG}.parquet", compression="zstd")
        SUBMISSION_DIR = OUTPUT_DIR / "submissions"
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        for submission_key, submission in submission_tables.items():
            submission.write_csv(SUBMISSION_DIR / f"submission_{submission_key}.csv")
        run_config = {
            "embedding_export": EMBEDDING_EXPORT, "feature_representation": FEATURE_REPRESENTATION,
            "feature_path": str(FEATURE_PATH), "test_feature_path": str(TEST_FEATURE_PATH),
            "test_target_path": str(TEST_TARGET_PATH),
            "raw_test_path": str(RAW_TEST_PATH),
            "feature_count": len(MODEL_FEATURE_COLUMNS), "feature_columns": MODEL_FEATURE_COLUMNS,
            "folds": FOLD_NUMBERS, "fold_type": "expanding_purged_walk_forward",
            "selection_metric": SELECTION_METRIC, "selection_std_metric": SELECTION_STD_METRIC,
            "decision_thresholds": list(DECISION_THRESHOLDS),
            "forecast_horizon_weekdays": FORECAST_HORIZON_WEEKDAYS,
            "submission_years": list(SUBMISSION_YEARS),
            "expected_submission_rows": EXPECTED_SUBMISSION_ROWS,
            "submission_price_conversion": "origin_close * exp(training_median_abs_horizon_log_return * threshold_centered_probability_score)",
            "hyperparameter_search": "compact_curated_configs",
            "config_counts": {model: len(configs) for model, configs in LEARNED_CONFIGS.items()},
            "random_state": RANDOM_STATE,
            "refit_all_model_families": REFIT_ALL_MODEL_FAMILIES,
            "tabpfn_max_train_rows": TABPFN_MAX_TRAIN_ROWS,
            "rule_configs": RULE_CONFIGS, "learned_configs": LEARNED_CONFIGS,
        }
        (OUTPUT_DIR / f"baseline_configuration_{FEATURE_TAG}.json").write_text(json.dumps(run_config, indent=2))
        print(f"Saved tuning results, fold reports, final refits, and predictions under {OUTPUT_DIR}.")
    
    
    


if __name__ == "__main__":
    main()
