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
            "# Kaggle inputs are read-only; generated artifacts belong in "
            "/kaggle/working.\n"
            'DATA_DIR = Path("/kaggle/working/data")',
        ).replace(
            'TRAINING_MODES = ("nested-folds", "full-only")',
            'TRAINING_MODES = ("full-only",)',
        ).replace(
            'DEFAULT_TEXT_FAMILIES = ("linq", "qwen")',
            'DEFAULT_TEXT_FAMILIES = ("qwen",)',
        )
        actual = "".join(self.notebook["cells"][3]["source"])
        self.assertEqual(actual, expected)

    def test_execution_is_hard_wired_to_full_only_and_qwen(self):
        all_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
        )
        self.assertIn(
            'args.training_mode = "full-only"',
            all_source,
        )
        self.assertIn(
            'training_mode="full-only"',
            all_source,
        )
        self.assertIn('FAMILIES = ["qwen"]', all_source)
        self.assertIn(
            "USE_PRETRAINED_PRICE_MODEL =",
            all_source,
        )
        self.assertIn(
            "no_pretrained_model=not USE_PRETRAINED_PRICE_MODEL",
            all_source,
        )

    def test_kaggle_input_staging_requires_only_qwen(self):
        all_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
        )
        self.assertIn(
            'KAGGLE_INPUT_ROOT = Path("/kaggle/input")',
            all_source,
        )
        self.assertIn('"qwen_textemb.parquet"', all_source)
        staging_source = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
            and "REQUIRED_INPUT_FILENAMES" in "".join(cell["source"])
        )
        self.assertNotIn("linq_textemb.parquet", staging_source)

    def test_submission_is_written_to_kaggle_working(self):
        all_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
        )
        self.assertIn(
            'FINAL_SUBMISSION_PATH = Path('
            '"/kaggle/working/submission.csv")',
            all_source,
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
