import unittest
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl
import torch
from torch import nn

from chronos2_lora_fusion import (
    Chronos2LoRATextFusion,
    ChronosContextStore,
    TopLoRALinear,
    fit_lora_fusion,
    inject_toplora,
)
from latent_fusion import (
    _early_stopping_reached,
    _select_decision_threshold,
    fit_raw_fusion_model,
)


class FakePatch(nn.Module):
    def forward(self, values):
        return values.reshape(values.shape[0], 2, 2)


class FakeChronosCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = FakePatch()
        self.lora_scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, *, context, group_ids):
        del group_ids
        patches = self.patch(torch.nan_to_num(context, nan=0.0)).mean(dim=-1)
        hidden = patches.unsqueeze(-1).repeat(1, 1, 4) * self.lora_scale
        special = torch.full(
            (len(context), 2, 4), 1000.0, device=context.device
        )
        return (
            (torch.cat([hidden, special], dim=1),),
            None,
            None,
            hidden.shape[1],
        )


class FakeRawStore:
    input_dim = 3

    def gather(self, indices):
        values = np.ones((*indices.shape, self.input_dim), dtype=np.float32)
        values[indices < 0] = 0.0
        return values


class Chronos2LoRAFusionTests(unittest.TestCase):
    def test_fixed_decision_threshold_does_not_require_validation_scores(self):
        threshold = _select_decision_threshold(
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.float32),
            calibrate=False,
            fixed_threshold=0.5,
        )
        self.assertEqual(threshold, 0.5)

    def test_fixed_decision_threshold_rejects_endpoints(self):
        for threshold in (0.0, 1.0):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError, "strictly between 0 and 1"
                ):
                    _select_decision_threshold(
                        np.empty(0, dtype=np.int8),
                        np.empty(0, dtype=np.float32),
                        calibrate=False,
                        fixed_threshold=threshold,
                    )

    def test_early_stopping_respects_minimum_epoch_and_patience(self):
        self.assertFalse(_early_stopping_reached(
            epoch=4,
            epochs_without_improvement=10,
            min_epochs=5,
            patience=3,
        ))
        self.assertFalse(_early_stopping_reached(
            epoch=5,
            epochs_without_improvement=2,
            min_epochs=5,
            patience=3,
        ))
        self.assertTrue(_early_stopping_reached(
            epoch=5,
            epochs_without_improvement=3,
            min_epochs=5,
            patience=3,
        ))

    def test_chronos_checkpoint_selection_starts_at_minimum_epoch(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
        features = pl.DataFrame({
            "row_id": list(range(6)),
            "ticker": ["AAA"] * 6,
            "date": dates,
            "first": [0, 1, 2, 3, 4, 5],
            "second": [10, 11, 12, 13, 14, 15],
        })
        contexts = ChronosContextStore.from_features(
            features,
            ("first", "second"),
            horizon=1,
            lookback=4,
            min_context=3,
        )
        index = features.filter(pl.col("row_id").is_in([3, 4])).select(
            ["row_id", "date", "ticker"]
        )
        model = Chronos2LoRATextFusion(
            FakeChronosCore(),
            n_variates=2,
            chronos_dim=4,
            covariate_dim=2,
            family_dims={"family": 3},
            field_count=2,
            hidden_dim=4,
            text_dim=4,
            fusion_depth=1,
            text_attention_heads=2,
            dropout=0.0,
        )
        with patch(
            "chronos2_lora_fusion.predict_lora_fusion",
            return_value=np.full(2, 0.5, dtype=np.float32),
        ):
            _, history = fit_lora_fusion(
                model,
                train_index=index,
                contexts=contexts,
                train_covariates=np.ones((2, 2), dtype=np.float32),
                train_text_indices={
                    "family": np.zeros((2, 2), dtype=np.int64)
                },
                stores={"family": FakeRawStore()},
                train_target=np.array([0.0, 1.0], dtype=np.float32),
                device="cpu",
                epochs=6,
                batch_size=2,
                learning_rate=1e-3,
                lora_learning_rate=1e-3,
                adapter_learning_rate_multiplier=0.1,
                weight_decay=0.0,
                seed=42,
                validation_index=index,
                validation_covariates=np.ones((2, 2), dtype=np.float32),
                validation_text_indices={
                    "family": np.zeros((2, 2), dtype=np.int64)
                },
                validation_target=np.array([0.0, 1.0], dtype=np.float32),
                early_stopping_min_epochs=3,
                early_stopping_patience=1,
            )

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["best_epoch"], 3.0)
        self.assertEqual(history[-1]["stopped_early"], 1.0)

    def test_raw_fusion_checkpoint_selection_starts_at_minimum_epoch(self):
        values = np.ones((4, 2), dtype=np.float32)
        target = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        text_indices = {
            "family": np.zeros((4, 2), dtype=np.int64)
        }
        validation_metrics = {
            "validation_bce": 0.5,
            "validation_accuracy": 0.5,
            "validation_balanced_accuracy": 0.5,
            "validation_positive_rate": 0.5,
        }
        with patch(
            "latent_fusion._raw_fusion_validation_metrics",
            return_value=validation_metrics,
        ):
            _, history = fit_raw_fusion_model(
                values,
                values,
                text_indices,
                {"family": FakeRawStore()},
                target,
                "cpu",
                text_dim=4,
                hidden_dim=4,
                market_depth=1,
                fusion_depth=1,
                expansion=1,
                dropout=0.0,
                epochs=6,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=42,
                text_attention_heads=2,
                validation_price=values,
                validation_covariates=values,
                validation_text_indices=text_indices,
                validation_target=target,
                early_stopping_min_epochs=3,
                early_stopping_patience=1,
            )

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["best_epoch"], 3.0)
        self.assertEqual(history[-1]["stopped_early"], 1.0)

    def test_toplora_starts_as_the_frozen_base_map(self):
        base = nn.Linear(4, 3, bias=False)
        layer = TopLoRALinear(
            base, rank=2, alpha=4.0, dropout=0.0
        )
        values = torch.randn(2, 5, 4)

        torch.testing.assert_close(layer(values), base(values))
        self.assertFalse(base.weight.requires_grad)

    def test_toplora_matches_tokenwise_projected_update(self):
        torch.manual_seed(7)
        layer = TopLoRALinear(
            nn.Linear(4, 3, bias=False),
            rank=2,
            alpha=2.0,
            dropout=0.0,
        )
        nn.init.normal_(layer.lora_B.weight)
        values = torch.randn(2, 3, 4)

        actual = layer(values)
        token_scale = torch.exp(
            layer.toplora_norm(values @ layer.toplora_projector)
        )
        expected = layer.base_layer(values) + layer.lora_B(
            layer.lora_A(values) * token_scale
        ) * layer.scaling
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNotNone(layer.toplora_projector.grad)
        self.assertTrue(torch.isfinite(layer.toplora_projector.grad).all())

    def test_toplora_injection_replaces_only_named_targets(self):
        class AttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attention = nn.Module()
                self.self_attention.q = nn.Linear(4, 4)
                self.self_attention.k = nn.Linear(4, 4)
                self.other = nn.Linear(4, 4)

        model = AttentionBlock()
        replaced = inject_toplora(
            model,
            target_modules=("self_attention.q", "self_attention.k"),
            rank=2,
            alpha=4,
            dropout=0.0,
        )

        self.assertEqual(
            replaced, ["self_attention.q", "self_attention.k"]
        )
        self.assertIsInstance(model.self_attention.q, TopLoRALinear)
        self.assertIsInstance(model.self_attention.k, TopLoRALinear)
        self.assertIsInstance(model.other, nn.Linear)

    def test_context_store_preserves_order_and_left_pads(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
        features = pl.DataFrame({
            "row_id": list(range(6)),
            "ticker": ["AAA"] * 6,
            "date": dates,
            "first": [0, 1, 2, 3, 4, 5],
            "second": [10, 11, 12, 13, 14, 15],
        })
        store = ChronosContextStore.from_features(
            features,
            ("first", "second"),
            horizon=1,
            lookback=4,
            min_context=3,
        )

        valid = store.valid_index(features.select(["row_id", "date", "ticker"]))
        self.assertEqual(valid["row_id"].to_list(), [3, 4, 5])
        batch = store.batch(np.array([3]))
        self.assertEqual(batch.shape, (1, 2, 4))
        self.assertTrue(np.isnan(batch[0, :, 0]).all())
        np.testing.assert_array_equal(
            batch[0, :, 1:],
            np.array([[1, 2, 3], [11, 12, 13]], dtype=np.float32),
        )

    def test_market_pooling_ignores_padding_and_special_tokens(self):
        core = FakeChronosCore()
        model = Chronos2LoRATextFusion(
            core,
            n_variates=2,
            chronos_dim=4,
            covariate_dim=2,
            family_dims={"family": 3},
            field_count=2,
            hidden_dim=4,
            text_dim=4,
            fusion_depth=1,
            text_attention_heads=2,
            dropout=0.0,
        )
        contexts = torch.tensor([[
            [float("nan"), float("nan"), 1.0, 3.0],
            [2.0, 4.0, 6.0, 8.0],
        ]])

        pooled = model.encode_market(contexts)

        expected = torch.tensor([[
            2.0, 2.0, 2.0, 2.0,
            5.0, 5.0, 5.0, 5.0,
        ]])
        torch.testing.assert_close(pooled, expected)

    def test_direction_loss_reaches_chronos_lora_parameter(self):
        core = FakeChronosCore()
        model = Chronos2LoRATextFusion(
            core,
            n_variates=2,
            chronos_dim=4,
            covariate_dim=2,
            family_dims={"family": 3},
            field_count=2,
            hidden_dim=4,
            text_dim=4,
            fusion_depth=1,
            text_attention_heads=2,
            dropout=0.0,
        )
        contexts = torch.tensor([
            [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
        ])
        logits = model(
            contexts,
            torch.ones(2, 2),
            {"family": torch.ones(2, 2, 3)},
            {"family": torch.ones(2, 2, dtype=torch.bool)},
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.0, 1.0])
        )
        loss.backward()

        self.assertIsNotNone(core.lora_scale.grad)
        self.assertTrue(torch.isfinite(core.lora_scale.grad))

    def test_one_epoch_joint_training_updates_lora_path(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
        features = pl.DataFrame({
            "row_id": list(range(6)),
            "ticker": ["AAA"] * 6,
            "date": dates,
            "first": [0, 1, 2, 3, 4, 5],
            "second": [10, 11, 12, 13, 14, 15],
        })
        contexts = ChronosContextStore.from_features(
            features,
            ("first", "second"),
            horizon=1,
            lookback=4,
            min_context=3,
        )
        index = features.filter(pl.col("row_id").is_in([3, 4])).select(
            ["row_id", "date", "ticker"]
        )
        model = Chronos2LoRATextFusion(
            FakeChronosCore(),
            n_variates=2,
            chronos_dim=4,
            covariate_dim=2,
            family_dims={"family": 3},
            field_count=2,
            hidden_dim=4,
            text_dim=4,
            fusion_depth=1,
            text_attention_heads=2,
            dropout=0.0,
        )
        before = model.chronos_model.lora_scale.detach().clone()

        model, history = fit_lora_fusion(
            model,
            train_index=index,
            contexts=contexts,
            train_covariates=np.ones((2, 2), dtype=np.float32),
            train_text_indices={
                "family": np.zeros((2, 2), dtype=np.int64)
            },
            stores={"family": FakeRawStore()},
            train_target=np.array([0.0, 1.0], dtype=np.float32),
            device="cpu",
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            lora_learning_rate=1e-3,
            adapter_learning_rate_multiplier=0.1,
            weight_decay=0.0,
            seed=42,
            early_stopping_min_epochs=1,
        )

        self.assertEqual(len(history), 1)
        self.assertFalse(torch.equal(
            before, model.chronos_model.lora_scale.detach()
        ))


if __name__ == "__main__":
    unittest.main()
