import json
import unittest
import tempfile
from pathlib import Path

from local_llm_chat.cli import (
    append_output_log,
    build_parser,
    configure_line_editing,
    is_file_command,
    is_paste_command,
    read_message_file,
    read_paste_input,
    resolve_base_url,
)


class FakeReadline:
    def __init__(self):
        self.parse_and_bind_calls = []
        self.history_length = None

    def parse_and_bind(self, value):
        self.parse_and_bind_calls.append(value)

    def set_history_length(self, value):
        self.history_length = value


class CliTests(unittest.TestCase):
    def test_parser_enables_line_editing_and_emoji_by_default(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.line_editing)
        self.assertTrue(args.show_emoji)

    def test_parser_can_disable_line_editing_and_emoji(self):
        args = build_parser().parse_args(["--no-line-editing", "--no-show-emoji"])

        self.assertFalse(args.line_editing)
        self.assertFalse(args.show_emoji)

    def test_parser_keeps_positive_line_editing_and_emoji_flags(self):
        args = build_parser().parse_args(["--line-editing", "--show-emoji"])

        self.assertTrue(args.line_editing)
        self.assertTrue(args.show_emoji)

    def test_default_base_url_uses_port_8080(self):
        self.assertEqual(resolve_base_url(port=8080, base_url=None), "http://127.0.0.1:8080")

    def test_port_changes_default_localhost_url(self):
        self.assertEqual(resolve_base_url(port=8081, base_url=None), "http://127.0.0.1:8081")

    def test_explicit_base_url_takes_precedence(self):
        self.assertEqual(
            resolve_base_url(port=8081, base_url="http://192.168.1.10:9000"),
            "http://192.168.1.10:9000",
        )

    def test_configure_line_editing_is_disabled_by_default(self):
        readline = FakeReadline()

        self.assertFalse(configure_line_editing(readline, enabled=False))

        self.assertEqual(readline.parse_and_bind_calls, [])
        self.assertIsNone(readline.history_length)

    def test_configure_line_editing_can_enable_common_bindings(self):
        readline = FakeReadline()

        self.assertTrue(configure_line_editing(readline, enabled=True))

        self.assertIn("tab: complete", readline.parse_and_bind_calls)
        self.assertIn("set editing-mode emacs", readline.parse_and_bind_calls)
        self.assertEqual(readline.history_length, 200)

    def test_configure_line_editing_tolerates_missing_readline(self):
        self.assertFalse(configure_line_editing(None, enabled=True))

    def test_is_paste_command(self):
        self.assertTrue(is_paste_command(" /paste "))
        self.assertFalse(is_paste_command("/paste now"))

    def test_read_paste_input_reads_until_send(self):
        lines = iter(["first", "second", "/send", "ignored"])
        messages = []

        text = read_paste_input(input_func=lambda: next(lines), print_func=messages.append)

        self.assertEqual(text, "first\nsecond")
        self.assertEqual(
            messages,
            ["Paste multi-line input. Finish with /send or /end on its own line."],
        )

    def test_read_paste_input_reads_until_end(self):
        lines = iter(["first", "/end"])

        self.assertEqual(
            read_paste_input(input_func=lambda: next(lines), print_func=lambda _: None),
            "first",
        )

    def test_is_file_command(self):
        self.assertTrue(is_file_command("/file prompt.txt"))
        self.assertFalse(is_file_command("/filename prompt.txt"))

    def test_read_message_file_reads_relative_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "message.txt"
            path.write_text("long\nmessage\n", encoding="utf-8")

            self.assertEqual(
                read_message_file("/file message.txt", base_dir=Path(tmp)),
                "long\nmessage\n",
            )

    def test_read_message_file_supports_quoted_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long message.txt"
            path.write_text("hello", encoding="utf-8")

            self.assertEqual(
                read_message_file('/file "long message.txt"', base_dir=Path(tmp)),
                "hello",
            )

    def test_read_message_file_rejects_missing_path(self):
        with self.assertRaisesRegex(RuntimeError, "Usage: /file"):
            read_message_file("/file")

    def test_append_output_log_writes_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.jsonl"

            append_output_log(
                path,
                user_text="こんにちは",
                raw_answer="raw\nanswer",
                display_output="LLM>\ndisplay",
            )

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["event"], "assistant_response")
            self.assertEqual(records[0]["user_text"], "こんにちは")
            self.assertEqual(records[0]["raw_answer"], "raw\nanswer")
            self.assertEqual(records[0]["display_output"], "LLM>\ndisplay")
            self.assertIn("timestamp", records[0])

    def test_append_output_log_appends_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.jsonl"

            append_output_log(path, user_text="one", raw_answer="a", display_output="LLM>\na")
            append_output_log(path, user_text="two", raw_answer="b", display_output="LLM>\nb")

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["user_text"] for record in records], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
