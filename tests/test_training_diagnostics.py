import inspect
import tempfile
import unittest
from pathlib import Path

import polars as pl

from latent_fusion import fit_raw_fusion_model, _write_training_diagnostics
from model_config import EARLY_STOPPING_MIN_EPOCHS, FUSION_EPOCHS


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_default_epoch_budget_reaches_early_stopping_minimum(self):
        parameters = inspect.signature(
            fit_raw_fusion_model
        ).parameters
        self.assertEqual(parameters["epochs"].default, FUSION_EPOCHS)
        self.assertGreaterEqual(
            parameters["epochs"].default,
            EARLY_STOPPING_MIN_EPOCHS,
        )

    def test_full_only_final_refit_writes_rows_and_plot(self):
        history = pl.DataFrame({
            "epoch": [1.0, 2.0],
            "bce": [0.70, 0.65],
            "train_bce": [0.70, 0.65],
            "train_positive_rate": [0.51, 0.51],
            "best_epoch": [2.0, 2.0],
            "is_best_epoch": [0.0, 1.0],
            "stopped_early": [0.0, 0.0],
            "fold": pl.Series([0, 0], dtype=pl.Int8),
            "model": ["timesfm_tft", "timesfm_tft"],
            "variant": ["all_families", "all_families"],
            "scope": ["final_full_training", "final_full_training"],
            "phase": ["final_full_refit", "final_full_refit"],
        })

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            stale_csv = (
                output_dir
                / "old_all_families_fold_learning_curves.csv"
            )
            stale_png = (
                output_dir
                / "old_all_families_fold_learning_curves.png"
            )
            stale_csv.write_text("stale")
            stale_png.write_bytes(b"stale")
            result = _write_training_diagnostics(
                {"timesfm_tft": [history]},
                output_dir,
            )

            self.assertEqual(result.height, 2)
            combined = pl.read_csv(
                output_dir / "all_fold_learning_curves.csv"
            )
            self.assertEqual(combined.height, 2)
            self.assertEqual(
                combined["phase"].unique().to_list(),
                ["final_full_refit"],
            )
            self.assertTrue(
                (
                    output_dir
                    / "timesfm_tft_fold_learning_curves.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "timesfm_tft_fold_learning_curves.png"
                ).exists()
            )
            self.assertFalse(stale_csv.exists())
            self.assertFalse(stale_png.exists())

    def test_full_only_diagnostics_include_inner_validation_bce(self):
        inner_history = pl.DataFrame({
            "epoch": [1.0, 2.0],
            "train_bce": [0.70, 0.65],
            "validation_bce": [0.72, 0.68],
            "validation_accuracy": [0.51, 0.54],
            "validation_balanced_accuracy": [0.50, 0.53],
            "fold": pl.Series([-1, -1], dtype=pl.Int8),
            "model": ["timesfm_tft", "timesfm_tft"],
            "variant": ["qwen", "qwen"],
            "scope": ["final_full_training", "final_full_training"],
            "phase": [
                "inner_validation_selection",
                "inner_validation_selection",
            ],
        })
        final_history = pl.DataFrame({
            "epoch": [1.0, 2.0],
            "train_bce": [0.69, 0.64],
            "fold": pl.Series([0, 0], dtype=pl.Int8),
            "model": ["timesfm_tft", "timesfm_tft"],
            "variant": ["qwen", "qwen"],
            "scope": ["final_full_training", "final_full_training"],
            "phase": ["final_full_refit", "final_full_refit"],
        })

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = _write_training_diagnostics(
                {"timesfm_tft": [inner_history, final_history]},
                output_dir,
            )

            self.assertEqual(result.height, 4)
            self.assertEqual(
                result.filter(
                    pl.col("phase") == "inner_validation_selection"
                )["validation_bce"].to_list(),
                [0.72, 0.68],
            )
            self.assertEqual(
                result.filter(
                    pl.col("phase") == "final_full_refit"
                )["validation_bce"].null_count(),
                2,
            )
            persisted = pl.read_csv(
                output_dir / "all_fold_learning_curves.csv"
            )
            self.assertEqual(
                persisted["validation_bce"].drop_nulls().len(),
                2,
            )


if __name__ == "__main__":
    unittest.main()
