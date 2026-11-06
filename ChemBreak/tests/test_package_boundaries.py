import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageBoundaryTests(unittest.TestCase):
    def test_removed_runtime_terms_are_absent_from_code_and_config(self):
        selected = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.yaml"))
        text = "\n".join(path.read_text().lower() for path in selected)
        self.assertNotIn("fall" + "back", text)
        self.assertNotIn("a" + "sr", text)
        self.assertNotIn("chcs", text)
        self.assertNotIn("proto" + "type", text)


if __name__ == "__main__":
    unittest.main()
