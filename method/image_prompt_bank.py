import json
import os
from typing import Any, Dict, List


def merge_abnormal_prompt_lists(default_prompts: List[str], json_prompts: List[str], mode: str) -> List[str]:
    """Resolve replace/append abnormal prompt lists while preserving order."""
    if mode not in ("replace", "append"):
        raise ValueError("abnormal_prompt_mode must be 'replace' or 'append'")

    prompts = json_prompts if mode == "replace" else default_prompts + json_prompts
    cleaned = []
    seen = set()
    for prompt in prompts:
        prompt = str(prompt).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        cleaned.append(prompt)
    return cleaned


class ImagePromptBank:
    """Load per-image difference prompts and resolve them by class/image path."""

    def __init__(
            self,
            json_path: str,
            fallback: str = "default",
            debug_limit: int = 5,
            phase: str = "test",
    ):
        if fallback not in ("default", "error"):
            raise ValueError("fallback must be 'default' or 'error'")
        if not json_path:
            raise ValueError("A JSON prompt path is required for image prompt bank.")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Image prompt JSON does not exist: {json_path}")

        self.json_path = json_path
        self.fallback = fallback
        self.debug_limit = debug_limit
        self.phase = phase
        self._debug_count = 0

        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                self.prompt_bank = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse image prompt JSON: {json_path}") from exc

        if not isinstance(self.prompt_bank, dict):
            raise ValueError(f"Image prompt JSON must be a class-keyed object: {json_path}")

    @staticmethod
    def _normalize_path(path: str) -> str:
        return str(path).replace("\\", "/")

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        return str(class_name).strip()

    @staticmethod
    def _clean_prompts(prompts: Any) -> List[str]:
        if not isinstance(prompts, list):
            return []

        cleaned = []
        seen = set()
        for prompt in prompts:
            prompt = str(prompt).strip()
            if not prompt or prompt in seen:
                continue
            seen.add(prompt)
            cleaned.append(prompt)
        return cleaned

    @classmethod
    def make_relative_key(cls, class_name: str, image_path: str) -> str:
        """Return a JSON key like test/defective/000.png when it is identifiable."""
        class_name = cls._normalize_class_name(class_name)
        path = cls._normalize_path(image_path)
        parts = [part for part in path.split("/") if part]

        for idx in range(len(parts) - 1):
            if parts[idx] == class_name and idx + 1 < len(parts) and parts[idx + 1] == "test":
                return "/".join(parts[idx + 1:])

        test_indices = [idx for idx, part in enumerate(parts) if part == "test"]
        if test_indices:
            return "/".join(parts[test_indices[-1]:])

        return ""

    def _get_class_items(self, class_name: str):
        class_name = self._normalize_class_name(class_name)
        if class_name in self.prompt_bank:
            return self.prompt_bank[class_name]

        matches = [
            value for key, value in self.prompt_bank.items()
            if str(key).lower() == class_name.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _lookup_item(class_items: Dict[str, Any], relative_key: str):
        if not isinstance(class_items, dict) or not relative_key:
            return None
        if relative_key in class_items:
            return class_items[relative_key]

        matches = [
            value for key, value in class_items.items()
            if str(key).replace("\\", "/").lower() == relative_key.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _extract_prompts(self, item: Any):
        if not isinstance(item, dict):
            return [], "missing"

        prompts = self._clean_prompts(item.get("difference_prompts"))
        if prompts:
            return prompts, "direct"

        parsed = item.get("parsed")
        if isinstance(parsed, dict):
            prompts = self._clean_prompts(parsed.get("difference_prompts"))
            if prompts:
                return prompts, "parsed"

        return [], "missing"

    def _debug(self, class_name, image_path, relative_key, prompts, found, fallback_used, source_format):
        if self._debug_count >= self.debug_limit:
            return
        self._debug_count += 1

        print("[ImagePromptBank]")
        print(f"phase: {self.phase}")
        print(f"class_name: {class_name}")
        print(f"image_path: {image_path}")
        print(f"relative_key: {relative_key}")
        print(f"found: {found}")
        print(f"source_format: {source_format}")
        print("json_prompts:")
        for prompt in prompts:
            print(f"  - {prompt}")
        print(f"fallback_used: {fallback_used}")

    def get_difference_prompts(self, class_name: str, image_path: str) -> Dict[str, Any]:
        relative_key = self.make_relative_key(class_name, image_path)
        class_items = self._get_class_items(class_name)
        item = self._lookup_item(class_items, relative_key)
        prompts, source_format = self._extract_prompts(item)
        found = bool(prompts)
        fallback_used = not found and self.fallback == "default"

        self._debug(class_name, image_path, relative_key, prompts, found, fallback_used, source_format)

        if not found and self.fallback == "error":
            raise KeyError(
                "Could not resolve JSON difference_prompts for "
                f"class_name={class_name}, image_path={image_path}, relative_key={relative_key}, "
                f"json_path={self.json_path}"
            )

        return {
            "prompts": prompts,
            "relative_key": relative_key,
            "found": found,
            "fallback_used": fallback_used,
            "source_format": source_format,
        }
