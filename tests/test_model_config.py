import unittest

from model_config import (
    ADAPTER_LEARNING_RATE_MULTIPLIER,
    FUSION_DEPTH,
    FUSION_HIDDEN_DIM,
    MARKET_DEPTH,
    OPTUNA_FUSION_DEPTH_CANDIDATES,
    OPTUNA_HIDDEN_DIM_CANDIDATES,
    OPTUNA_LEARNING_RATE_MAX,
    OPTUNA_LEARNING_RATE_MIN,
    OPTUNA_MARKET_DEPTH_CANDIDATES,
)


class TftRegularizationConfigTests(unittest.TestCase):
    def test_reduced_capacity_search_space(self):
        self.assertEqual(FUSION_HIDDEN_DIM, 128)
        self.assertEqual(MARKET_DEPTH, 1)
        self.assertEqual(FUSION_DEPTH, 1)
        self.assertEqual(OPTUNA_HIDDEN_DIM_CANDIDATES, (64, 128))
        self.assertEqual(OPTUNA_MARKET_DEPTH_CANDIDATES, (1, 2))
        self.assertEqual(OPTUNA_FUSION_DEPTH_CANDIDATES, (1, 2))

    def test_reduced_head_and_adapter_learning_rates(self):
        self.assertEqual(OPTUNA_LEARNING_RATE_MIN, 2e-6)
        self.assertEqual(OPTUNA_LEARNING_RATE_MAX, 6e-5)
        self.assertAlmostEqual(
            OPTUNA_LEARNING_RATE_MIN * ADAPTER_LEARNING_RATE_MULTIPLIER,
            2e-7,
        )
        self.assertAlmostEqual(
            OPTUNA_LEARNING_RATE_MAX * ADAPTER_LEARNING_RATE_MULTIPLIER,
            6e-6,
        )


if __name__ == "__main__":
    unittest.main()
