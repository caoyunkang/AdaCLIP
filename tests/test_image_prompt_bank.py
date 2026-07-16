import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "method" / "image_prompt_bank.py"
SPEC = importlib.util.spec_from_file_location("image_prompt_bank_under_test", MODULE_PATH)
IMAGE_PROMPT_BANK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMAGE_PROMPT_BANK
SPEC.loader.exec_module(IMAGE_PROMPT_BANK)

ImagePromptBank = IMAGE_PROMPT_BANK.ImagePromptBank
merge_abnormal_prompt_lists = IMAGE_PROMPT_BANK.merge_abnormal_prompt_lists


class ImagePromptBankTest(unittest.TestCase):
    def _bank(self, payload, fallback="default"):
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "prompts.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        bank = ImagePromptBank(str(path), fallback=fallback, debug_limit=0)
        self.addCleanup(tmpdir.cleanup)
        return bank

    def test_mvtec_relative_key(self):
        key = ImagePromptBank.make_relative_key(
            "toothbrush",
            "/content/mvtec-test/mvtec-data/toothbrush/test/defective/000.png",
        )
        self.assertEqual(key, "test/defective/000.png")

    def test_visa_relative_key(self):
        key = ImagePromptBank.make_relative_key(
            "candle",
            "/content/visa-test/VisA_pytorch/1cls/candle/test/bad/000.JPG",
        )
        self.assertEqual(key, "test/bad/000.JPG")

    def test_direct_prompts(self):
        bank = self._bank({"candle": {"test/bad/000.JPG": {
            "difference_prompts": ["a candle with a chipped edge"]
        }}})
        info = bank.get_difference_prompts(
            "candle",
            "/content/visa-test/VisA_pytorch/1cls/candle/test/bad/000.JPG",
        )
        self.assertTrue(info["found"])
        self.assertEqual(info["source_format"], "direct")
        self.assertEqual(info["prompts"], ["a candle with a chipped edge"])

    def test_parsed_prompts(self):
        bank = self._bank({"toothbrush": {"test/defective/000.png": {
            "parsed": {"difference_prompts": ["a toothbrush with deformed bristles"]}
        }}})
        info = bank.get_difference_prompts(
            "toothbrush",
            "/content/mvtec-test/mvtec-data/toothbrush/test/defective/000.png",
        )
        self.assertTrue(info["found"])
        self.assertEqual(info["source_format"], "parsed")
        self.assertEqual(info["prompts"], ["a toothbrush with deformed bristles"])

    def test_parsed_none_fallback_and_error(self):
        payload = {"toothbrush": {"test/defective/000.png": {"parsed": None}}}
        bank = self._bank(payload, fallback="default")
        info = bank.get_difference_prompts(
            "toothbrush",
            "/content/mvtec-test/mvtec-data/toothbrush/test/defective/000.png",
        )
        self.assertFalse(info["found"])
        self.assertTrue(info["fallback_used"])

        bank_error = self._bank(payload, fallback="error")
        with self.assertRaises(KeyError):
            bank_error.get_difference_prompts(
                "toothbrush",
                "/content/mvtec-test/mvtec-data/toothbrush/test/defective/000.png",
            )

    def test_replace_and_append(self):
        self.assertEqual(
            merge_abnormal_prompt_lists(["default abnormal"], ["json abnormal"], "replace"),
            ["json abnormal"],
        )
        self.assertEqual(
            merge_abnormal_prompt_lists(["default abnormal"], ["json abnormal"], "append"),
            ["default abnormal", "json abnormal"],
        )


if __name__ == "__main__":
    unittest.main()
