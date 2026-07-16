import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "path_tools.py"
SPEC = importlib.util.spec_from_file_location("path_tools_under_test", MODULE_PATH)
PATH_TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATH_TOOLS)


class PathToolsTest(unittest.TestCase):
    def test_relative_path_is_resolved_from_repo_root(self):
        resolved = PATH_TOOLS.resolve_repo_path("workspaces/example")
        expected = (PATH_TOOLS.REPO_ROOT / "workspaces" / "example").resolve()
        self.assertEqual(Path(resolved), expected)

    def test_absolute_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(Path(PATH_TOOLS.resolve_repo_path(tmpdir)), Path(tmpdir).resolve())

    def test_empty_path_is_preserved(self):
        self.assertEqual(PATH_TOOLS.resolve_repo_path(""), "")
