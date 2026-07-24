"""Central paths and default parameters shared by forecasting model scripts."""

from pathlib import Path


# Data and artifact locations shared by baseline and Stage 2 models.
DATA_DIR = Path("data")
EMBEDDING_EXPORT = "pca_embeddings"
ARTIFACT_DIR = DATA_DIR / "model_artifacts" / EMBEDDING_EXPORT
BASELINE_DIR = ARTIFACT_DIR / "baselines"
RAW_TEST_PATH = DATA_DIR / "test.parquet"
PREPARED_TRAIN_PATH = ARTIFACT_DIR / "train_features_without_embeddings.parquet"
PREPARED_TEST_PATH = ARTIFACT_DIR / "test_features_without_embeddings.parquet"
TRAIN_TARGET_PATH = ARTIFACT_DIR / "train_target.parquet"
TEST_TARGET_PATH = ARTIFACT_DIR / "test_target.parquet"
TRAIN_LINK_PATH = ARTIFACT_DIR / "train_text_links.parquet"
TEST_LINK_PATH = ARTIFACT_DIR / "test_text_links.parquet"
FOLD_PATH = ARTIFACT_DIR / "walk_forward_assignments.parquet"
PRICE_CACHE_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_simple_concatenation"
    / "price_latents"
)
PRETRAINED_OUTPUT_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_covariates_unified_raw_text_attention3"
)
TFT_OUTPUT_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_temporal_tft_unified_raw_text_attention"
)
CROSS_STOCK_TFT_OUTPUT_DIR = (
    ARTIFACT_DIR
    / "stage2_pretrained_forecasts"
    / "timesfm_temporal_tft_cross_stock_unified_raw_text_attention"
)

# Shared experiment protocol.
RANDOM_STATE = 42
HORIZON = 20
FORECAST_HORIZON_WEEKDAYS = HORIZON
SUBMISSION_YEARS = (2022, 2023)
EXPECTED_SUBMISSION_ROWS = 52_000
DEFAULT_TEXT_FAMILIES = ("linq", "qwen")
OPTUNA_TRIALS = 12
SELECTION_PROTOCOL_VERSION = 8
TRAINING_MODES = ("nested-folds", "full-only")
DEFAULT_TRAINING_MODE = "nested-folds"

# Feature classification shared by Stage 2 models.
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

# Frozen TimesFM and shared fusion defaults.
PRICE_ENCODER_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
LOOKBACK = 512
MIN_CONTEXT = 64
TIMESFM_INPUT_COLUMN = "ret_20"
RAW_TEXT_DIM = 384
TEXT_ATTENTION_HEADS = 4
TEXT_ATTENTION_LAYERS = 1
FUSION_HIDDEN_DIM = 256
MARKET_DEPTH = 2
FUSION_DEPTH = 2
RESIDUAL_EXPANSION = 2
FUSION_DROPOUT = 0.20
FUSION_EPOCHS = 100
PRICE_BATCH_SIZE = 16
FUSION_BATCH_SIZE = 128
ADAPTER_LEARNING_RATE_MULTIPLIER = 0.1
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5

# Optuna search space shared by the residual and TFT fusion models.
OPTUNA_HIDDEN_DIM_CANDIDATES = (64, 128, 256)
OPTUNA_FUSION_DEPTH_CANDIDATES = (1, 2, 3, 4, 5)
OPTUNA_EXPANSION_CANDIDATES = (1, 2, 3, 4, 5)
OPTUNA_DROPOUT_CANDIDATES = (0.10, 0.15, 0.20, 0.25)
OPTUNA_LEARNING_RATE_MIN = 1e-5
OPTUNA_LEARNING_RATE_MAX = 3e-4
OPTUNA_WEIGHT_DECAY_MIN = 1e-4
OPTUNA_WEIGHT_DECAY_MAX = 1e-2
OPTUNA_MARKET_DEPTH_CANDIDATES = (1, 2, 3, 4, 5)

# TFT-specific defaults shared by per-stock and cross-stock TFT entry points.
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
RAW_FUSION_BATCH_SIZE = 128
CROSS_STOCK_ATTENTION_HEADS = 4
