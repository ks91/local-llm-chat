import tempfile
import unittest
from pathlib import Path

from local_llm_chat.chat import load_instructions


class InstructionsTests(unittest.TestCase):
    def test_loads_existing_instructions_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instructions.md"
            path.write_text("You are local.\n", encoding="utf-8")

            self.assertEqual(load_instructions(path), "You are local.")

    def test_missing_instructions_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_instructions(Path(tmp) / "missing.md"), "")


if __name__ == "__main__":
    unittest.main()
