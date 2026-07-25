import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "stage2_tft_forecasts_submission.ipynb"


class SubmissionNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text())

    def test_configuration_matches_source_except_full_only_mode(self):
        expected = (ROOT / "model_config.py").read_text().replace(
            'DATA_DIR = Path("data")',
            "# Change this path if the submitted data directory is mounted "
            "elsewhere.\n"
            'DATA_DIR = Path("data")',
        ).replace(
            'TRAINING_MODES = ("nested-folds", "full-only")',
            'TRAINING_MODES = ("full-only",)',
        )
        actual = "".join(self.notebook["cells"][3]["source"])
        self.assertEqual(actual, expected)

    def test_execution_is_hard_wired_to_full_only(self):
        runner_source = "".join(
            self.notebook["cells"][13]["source"]
        )
        run_config_source = "".join(
            self.notebook["cells"][15]["source"]
        )
        self.assertIn(
            'args.training_mode = "full-only"',
            runner_source,
        )
        self.assertIn(
            'training_mode="full-only"',
            run_config_source,
        )

    def test_code_cells_have_no_project_local_imports(self):
        local_modules = {
            "latent_fusion",
            "model_config",
            "stage2_pretrained_forecasts",
            "stage2_tft_forecasts",
            "utils",
        }
        violations = []
        for cell_index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code" or cell_index == 0:
                continue
            tree = ast.parse(
                "".join(cell["source"]),
                filename=f"notebook_cell_{cell_index}",
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in local_modules:
                            violations.append(
                                (cell_index, alias.name)
                            )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split(".")[0] in local_modules
                ):
                    violations.append((cell_index, node.module))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
