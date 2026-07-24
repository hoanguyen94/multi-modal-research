import unittest
from datetime import date, timedelta

import numpy as np
import polars as pl
import torch

from latent_fusion import (
    _multivariate_context_array,
    _pooled_chronos2_hidden,
    _prepared_multivariate_history_lookup,
)


class FakeChronos2Pipeline:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.calls = []

    def embed(self, inputs, *, batch_size, context_length):
        self.calls.append((inputs, batch_size, context_length))
        loc_scale = [
            (
                torch.zeros(embedding.shape[0]),
                torch.ones(embedding.shape[0]),
            )
            for embedding in self.embeddings
        ]
        return self.embeddings, loc_scale


class Chronos2LatentTests(unittest.TestCase):
    def test_pooling_excludes_special_tokens_and_preserves_variate_order(self):
        # Two variates, two observed patch tokens, hidden width two. Chronos-2
        # appends [REG] and masked-future tokens as the final two positions.
        embedding = torch.tensor([
            [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0], [200.0, 200.0]],
            [[2.0, 4.0], [6.0, 8.0], [300.0, 300.0], [400.0, 400.0]],
        ])
        pipeline = FakeChronos2Pipeline([embedding])
        context = np.arange(16, dtype=np.float32).reshape(2, 8)

        result = _pooled_chronos2_hidden(
            pipeline,
            [context],
            input_columns=("ret_1", "ret_20"),
            lookback=8,
            batch_size=1,
        )

        np.testing.assert_allclose(
            result,
            np.array([[3.0, 5.0, 4.0, 6.0]], dtype=np.float32),
        )
        self.assertEqual(result.shape, (1, 4))
        _, used_batch_size, used_context_length = pipeline.calls[0]
        self.assertEqual(used_batch_size, 2)
        self.assertEqual(used_context_length, 8)

    def test_history_lookup_and_context_keep_configured_column_order(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
        features = pl.DataFrame({
            "ticker": ["AAA"] * 6,
            "date": dates,
            "second": [10, 11, 12, 13, 14, 15],
            "first": [0, 1, 2, 3, 4, 5],
        })

        histories, positions = _prepared_multivariate_history_lookup(
            features, ("first", "second")
        )
        context = _multivariate_context_array(
            histories["AAA"],
            positions["AAA"][dates[-1]],
            horizon=1,
            lookback=3,
            min_context=3,
        )

        np.testing.assert_array_equal(
            context,
            np.array([[3, 4, 5], [13, 14, 15]], dtype=np.float32),
        )

    def test_pooling_rejects_wrong_variate_count(self):
        pipeline = FakeChronos2Pipeline([])
        with self.assertRaisesRegex(ValueError, "configured variates"):
            _pooled_chronos2_hidden(
                pipeline,
                [np.ones((1, 8), dtype=np.float32)],
                input_columns=("a", "b"),
                lookback=8,
                batch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
