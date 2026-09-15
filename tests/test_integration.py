import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiment import run


class IntegrationTests(unittest.TestCase):
    def test_proposed_smoke_run_emits_auditable_outputs(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tiny.npz"
            x_train = rng.normal(size=(120, 6)).astype(np.float32)
            x_test = rng.normal(size=(40, 6)).astype(np.float32)
            np.savez(
                data, x_train=x_train, x_test=x_test,
                y1_train=rng.integers(0, 3, 120), y1_test=rng.integers(0, 3, 40),
                y2_train=rng.integers(0, 2, 120), y2_test=rng.integers(0, 2, 40),
            )
            output = root / "result"
            manifest = run(data, "proposed", 11, 2, 6, 2, output)
            self.assertTrue((output / "round_metrics.csv").exists())
            self.assertTrue((output / "participation.csv").exists())
            self.assertGreater(manifest["payload_bytes"], 0)
            self.assertGreater(manifest["residual_bytes"], 0)

    def test_regression_and_all_baselines_execute(self):
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "regression.npz"
            x_train = rng.normal(size=(100, 4)).astype(np.float32)
            x_test = rng.normal(size=(20, 4)).astype(np.float32)
            np.savez(
                data, x_train=x_train, x_test=x_test,
                y1_train=rng.normal(size=100), y1_test=rng.normal(size=20),
                y2_train=rng.normal(size=100), y2_test=rng.normal(size=20),
                task_type=np.asarray(["regression", "regression"]),
            )
            for method in ("fedavg", "fedprox", "fmtl", "topksgd", "scaffold", "proposed"):
                manifest = run(data, method, 7, 1, 5, 2, root / method)
                self.assertEqual(tuple(manifest["task_types"]), ("regression", "regression"))


if __name__ == "__main__":
    unittest.main()
