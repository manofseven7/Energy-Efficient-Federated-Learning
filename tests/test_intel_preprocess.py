import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.preprocess_intel import preprocess


class IntelPreprocessTests(unittest.TestCase):
    def test_chronological_split_and_train_only_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "data.txt.gz"; output = root / "intel.npz"
            lines = []
            for sensor in (1, 2):
                for epoch in range(1, 31):
                    lines.append(f"2004-01-01 00:00:00 {epoch} {sensor} {20+epoch/10} {40+epoch/10} 100 2.8\n")
            with gzip.open(source, "wt") as handle:
                handle.writelines(lines)
            manifest = preprocess(source, output)
            data = np.load(output)
            self.assertGreater(manifest["train_rows"], 0)
            self.assertGreater(manifest["test_rows"], 0)
            np.testing.assert_allclose(data["x_train"].mean(axis=0), 0, atol=1e-5)
            self.assertTrue(np.all(np.isfinite(data["x_test"])))


if __name__ == "__main__":
    unittest.main()

