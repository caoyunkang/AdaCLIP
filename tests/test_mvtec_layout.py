import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "dataset" / "mvtec_layout.py"
SPEC = importlib.util.spec_from_file_location("mvtec_layout_under_test", MODULE_PATH)
MVTEC_LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MVTEC_LAYOUT)


class MVTecLayoutTest(unittest.TestCase):
    def test_builds_metadata_for_good_and_anomalous_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = [
                root / "candle" / "train" / "good" / "0000.JPG",
                root / "candle" / "test" / "good" / "0001.JPG",
                root / "candle" / "test" / "bad" / "0002.JPG",
                root / "candle" / "ground_truth" / "bad" / "0002_mask.png",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            meta = MVTEC_LAYOUT.build_mvtec_style_meta(root, ["candle"])

            self.assertEqual(meta["train"]["candle"][0]["anomaly"], 0)
            self.assertEqual(meta["test"]["candle"][0]["img_path"], "candle/test/bad/0002.JPG")
            self.assertEqual(meta["test"]["candle"][0]["anomaly"], 1)
            self.assertEqual(
                meta["test"]["candle"][0]["mask_path"],
                "candle/ground_truth/bad/0002_mask.png",
            )
            self.assertEqual(meta["test"]["candle"][1]["img_path"], "candle/test/good/0001.JPG")
            self.assertEqual(meta["test"]["candle"][1]["mask_path"], "")
