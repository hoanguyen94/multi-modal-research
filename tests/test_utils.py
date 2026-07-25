import unittest

import numpy as np

from utils import directional_classification_metrics


class DirectionalClassificationMetricsTests(unittest.TestCase):
    def test_reports_binary_cross_entropy(self):
        truth = np.array([0, 1], dtype=np.int8)
        probability = np.array([0.25, 0.75], dtype=np.float64)

        metrics = directional_classification_metrics(
            truth, probability, threshold=0.5
        )

        self.assertAlmostEqual(
            metrics["bce"], -np.log(0.75), places=12
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
