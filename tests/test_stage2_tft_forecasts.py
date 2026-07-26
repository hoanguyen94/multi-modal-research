import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import torch

import stage2_tft_forecasts as stage2_tft
from latent_fusion import TFTMarketEncoder
from stage2_tft_forecasts import (
    _index_only_price_inputs,
    parse_args,
)


class NoPretrainedPriceModelTests(unittest.TestCase):
    def test_cli_toggle_is_available(self):
        with patch.object(
            sys,
            "argv",
            ["stage2_tft_forecasts.py", "--no-pretrained-model"],
        ):
            args = parse_args()

        self.assertTrue(args.no_pretrained_model)

    def test_index_only_inputs_have_no_price_latent_columns(self):
        origins = pl.DataFrame({
            "row_id": [2, 1],
            "date": [date(2020, 1, 2), date(2020, 1, 1)],
            "ticker": ["B", "A"],
        })

        result = _index_only_price_inputs(origins)

        self.assertEqual(result.columns, ["row_id", "date", "ticker"])
        self.assertFalse(any(
            column.startswith("price_latent_")
            for column in result.columns
        ))
        self.assertEqual(result["row_id"].to_list(), [1, 2])

    def test_tft_accepts_zero_width_price_input(self):
        model = TFTMarketEncoder(
            price_dim=0,
            covariate_dim=3,
            temporal_dim=2,
            hidden_dim=8,
            attention_heads=2,
            dropout=0.0,
        )

        output = model(
            torch.empty(4, 0),
            torch.randn(4, 3),
            torch.randn(4, 5, 2),
            torch.zeros(4, 5, dtype=torch.bool),
        )

        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_pipeline_toggle_skips_pretrained_model_and_latents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {
                "PREPARED_TRAIN_PATH": root / "train_features.parquet",
                "PREPARED_TEST_PATH": root / "test_features.parquet",
                "TRAIN_TARGET_PATH": root / "train_target.parquet",
                "TEST_TARGET_PATH": root / "test_target.parquet",
                "TRAIN_LINK_PATH": root / "train_links.parquet",
                "TEST_LINK_PATH": root / "test_links.parquet",
                "RAW_TEST_PATH": root / "test.parquet",
                "FOLD_PATH": root / "folds.parquet",
            }
            train_features = pl.DataFrame({
                "row_id": [1, 2],
                "date": [date(2020, 1, 1), date(2020, 1, 2)],
                "ticker": ["A", "A"],
                "ret_1": [0.01, -0.01],
            })
            test_features = pl.DataFrame({
                "row_id": [3],
                "date": [date(2022, 1, 3)],
                "ticker": ["A"],
                "ret_1": [0.02],
            })
            train_features.write_parquet(paths["PREPARED_TRAIN_PATH"])
            test_features.write_parquet(paths["PREPARED_TEST_PATH"])
            pl.DataFrame({
                "row_id": [1, 2],
                "date": [date(2020, 1, 1), date(2020, 1, 2)],
                "ticker": ["A", "A"],
                "fwd_log_return_20": [0.1, -0.1],
                "target_up": [1.0, 0.0],
            }).write_parquet(paths["TRAIN_TARGET_PATH"])
            pl.DataFrame({
                "row_id": [3],
                "date": [date(2022, 1, 3)],
                "ticker": ["A"],
                "fwd_log_return_20": [None],
                "target_up": [None],
            }).write_parquet(paths["TEST_TARGET_PATH"])
            links = pl.DataFrame({
                "row_id": [1],
                "text_field": ["macro_1"],
                "text_id": ["text-1"],
            })
            links.write_parquet(paths["TRAIN_LINK_PATH"])
            links.with_columns(
                pl.lit(3).cast(pl.Int64).alias("row_id")
            ).write_parquet(paths["TEST_LINK_PATH"])
            paths["RAW_TEST_PATH"].touch()

            args = SimpleNamespace(
                families=["qwen"],
                optuna_trials=0,
                no_tune=True,
                training_mode="full-only",
                price_encoder="timesfm",
                no_pretrained_model=True,
                no_price_extraction=False,
                force_price_refresh=False,
            )
            reports = {
                "final_metrics": pl.DataFrame(),
                "submission_manifest": pl.DataFrame(),
            }
            patches = {
                **paths,
                "DATA_DIR": root,
            }
            with (
                patch.multiple(stage2_tft, **patches),
                patch.object(
                    stage2_tft,
                    "parquet_embedding_dim",
                    return_value=4,
                ),
                patch.object(stage2_tft, "require_paths"),
                patch.object(stage2_tft, "select_device", return_value="cpu"),
                patch.object(
                    stage2_tft,
                    "classify_covariates",
                    return_value=(
                        tuple(stage2_tft.TFT_TEMPORAL_COLUMNS),
                        (),
                        (),
                        ("ret_1",),
                    ),
                ),
                patch.object(
                    stage2_tft,
                    "price_encoder_settings",
                    side_effect=AssertionError(
                        "pretrained settings must not be resolved"
                    ),
                ),
                patch.object(
                    stage2_tft,
                    "generate_frozen_price_latents",
                    side_effect=AssertionError(
                        "pretrained latents must not be generated"
                    ),
                ),
                patch.object(
                    stage2_tft,
                    "run_walk_forward_fusion",
                    return_value=reports,
                ) as run_fusion,
            ):
                result = stage2_tft.run_tft_pipeline(
                    args,
                    output_dir=root / "output",
                )

            self.assertIs(result, reports)
            call = run_fusion.call_args.kwargs
            self.assertEqual(call["price_encoder"], "none")
            self.assertEqual(
                call["train_price_latents"].columns,
                ["row_id", "date", "ticker"],
            )
            self.assertEqual(
                call["test_price_latents"].columns,
                ["row_id", "date", "ticker"],
            )
            metadata = json.loads(
                (root / "output" / "feature_groups.json").read_text()
            )
            self.assertFalse(metadata["pretrained_price_model_enabled"])
            self.assertIsNone(metadata["frozen_price_encoder"])
            self.assertIsNone(metadata["price_encoder_model_id"])
            self.assertEqual(metadata["price_encoder_input_columns"], [])


if __name__ == "__main__":
    unittest.main()
