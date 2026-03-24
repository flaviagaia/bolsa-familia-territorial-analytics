from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline


class PipelineTest(unittest.TestCase):
    def test_pipeline_runs_and_outputs_summary(self):
        summary = run_pipeline()
        self.assertEqual(summary["municipios_cobertos"], 102)
        self.assertEqual(summary["anos_cobertos"], 19)
        self.assertEqual(summary["linhas_operacionais"], 2448)
        self.assertGreater(summary["taxa_media_saque_pct"], 90)
        self.assertIn("regression_r2", summary)
        self.assertIn("clusters", summary)
        self.assertIn("anomaly_rows", summary)


if __name__ == "__main__":
    unittest.main()
