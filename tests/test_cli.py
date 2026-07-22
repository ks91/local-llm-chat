import json
import unittest
import tempfile
from pathlib import Path

from local_llm_chat.cli import (
    append_output_log,
    build_pdf_multimodal_user_text,
    build_pdf_user_text,
    build_parser,
    configure_input,
    configure_line_editing,
    extract_pdf_text_with_pdftotext,
    is_file_command,
    is_paste_command,
    read_message_file,
    read_pdf_text,
    read_paste_input,
    render_pdf_pages_to_image_urls,
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


class FakePromptToolkit:
    prompt_calls = []

    class InMemoryHistory:
        pass

    @classmethod
    def prompt(cls, message="", *, history=None):
        cls.prompt_calls.append({"message": message, "history": history})
        return f"prompted: {message}"


class FakePdfPage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfModule:
    class PdfReader:
        def __init__(self, path):
            self.path = path
            self.pages = [
                FakePdfPage("first page"),
                FakePdfPage(" "),
                FakePdfPage("second page"),
            ]


class FakeCompletedProcess:
    def __init__(self, *, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


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

    def test_parser_accepts_optional_pdf_path(self):
        args = build_parser().parse_args(["paper.pdf"])

        self.assertEqual(args.pdf, Path("paper.pdf"))

    def test_parser_disables_pdf_images_by_default(self):
        args = build_parser().parse_args(["paper.pdf"])

        self.assertFalse(args.pdf_images)
        self.assertEqual(args.pdf_image_dpi, 144)
        self.assertEqual(args.pdf_image_max_pages, 8)

    def test_parser_can_enable_pdf_images(self):
        args = build_parser().parse_args(
            [
                "--pdf-images",
                "--pdf-image-dpi",
                "96",
                "--pdf-image-max-pages",
                "3",
                "paper.pdf",
            ]
        )

        self.assertTrue(args.pdf_images)
        self.assertEqual(args.pdf_image_dpi, 96)
        self.assertEqual(args.pdf_image_max_pages, 3)

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

    def test_configure_input_uses_basic_input_when_line_editing_is_disabled(self):
        readline = FakeReadline()
        input_func = lambda prompt="": prompt

        selected = configure_input(
            line_editing=False,
            input_func=input_func,
            readline_module=readline,
            prompt_toolkit_module=FakePromptToolkit,
        )

        self.assertIs(selected, input_func)
        self.assertEqual(readline.parse_and_bind_calls, [])

    def test_configure_input_prefers_prompt_toolkit_when_available(self):
        readline = FakeReadline()
        input_func = lambda prompt="": prompt
        FakePromptToolkit.prompt_calls = []

        selected = configure_input(
            line_editing=True,
            input_func=input_func,
            readline_module=readline,
            prompt_toolkit_module=FakePromptToolkit,
        )

        self.assertEqual(selected("You> "), "prompted: You> ")
        self.assertEqual(readline.parse_and_bind_calls, [])
        self.assertIsInstance(
            FakePromptToolkit.prompt_calls[0]["history"],
            FakePromptToolkit.InMemoryHistory,
        )

    def test_configure_input_reuses_prompt_toolkit_history(self):
        FakePromptToolkit.prompt_calls = []

        selected = configure_input(
            line_editing=True,
            input_func=lambda prompt="": prompt,
            readline_module=FakeReadline(),
            prompt_toolkit_module=FakePromptToolkit,
        )

        selected("first> ")
        selected("second> ")

        self.assertIs(
            FakePromptToolkit.prompt_calls[0]["history"],
            FakePromptToolkit.prompt_calls[1]["history"],
        )

    def test_configure_input_falls_back_to_readline_without_prompt_toolkit(self):
        readline = FakeReadline()
        input_func = lambda prompt="": prompt

        selected = configure_input(
            line_editing=True,
            input_func=input_func,
            readline_module=readline,
            prompt_toolkit_module=None,
        )

        self.assertIs(selected, input_func)
        self.assertIn("set editing-mode emacs", readline.parse_and_bind_calls)

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

    def test_read_pdf_text_extracts_text_from_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            self.assertEqual(
                read_pdf_text(path, pypdf_module=FakePdfModule),
                "first page\n\nsecond page",
            )

    def test_extract_pdf_text_with_pdftotext_runs_layout_mode(self):
        calls = []

        def runner(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return FakeCompletedProcess(stdout="pdftotext output\n")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            self.assertEqual(
                extract_pdf_text_with_pdftotext(
                    path,
                    pdftotext_command="pdftotext",
                    runner=runner,
                ),
                "pdftotext output",
            )

        self.assertEqual(calls[0]["args"], ["pdftotext", "-layout", str(path), "-"])
        self.assertEqual(calls[0]["kwargs"]["encoding"], "utf-8")

    def test_extract_pdf_text_with_pdftotext_returns_none_when_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            self.assertIsNone(
                extract_pdf_text_with_pdftotext(path, pdftotext_command=None)
            )

    def test_extract_pdf_text_with_pdftotext_returns_none_on_failure(self):
        def runner(args, **kwargs):
            return FakeCompletedProcess(stderr="bad pdf", returncode=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            self.assertIsNone(
                extract_pdf_text_with_pdftotext(
                    path,
                    pdftotext_command="pdftotext",
                    runner=runner,
                )
            )

    def test_read_pdf_text_prefers_pdftotext(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            command = Path(tmp) / "fake-pdftotext"
            command.write_text("#!/bin/sh\nprintf 'from pdftotext\\n'\n", encoding="utf-8")
            command.chmod(0o755)

            self.assertEqual(
                read_pdf_text(
                    path,
                    pypdf_module=FakePdfModule,
                    pdftotext_command=command,
                ),
                "from pdftotext",
            )

    def test_read_pdf_text_rejects_non_pdf_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.txt"
            path.write_text("not pdf", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Expected a PDF"):
                read_pdf_text(path, pypdf_module=FakePdfModule)

    def test_read_pdf_text_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                read_pdf_text(Path(tmp) / "missing.pdf", pypdf_module=FakePdfModule)

    def test_read_pdf_text_reports_missing_pypdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            with self.assertRaisesRegex(RuntimeError, "requires pdftotext or pypdf"):
                read_pdf_text(path, pypdf_module=None)

    def test_build_pdf_user_text_includes_path_and_extracted_text(self):
        text = build_pdf_user_text(Path("paper.pdf"), "body")

        self.assertIn("Read the following PDF text", text)
        self.assertIn("PDF file: paper.pdf", text)
        self.assertTrue(text.endswith("body"))

    def test_build_pdf_multimodal_user_text_includes_text_and_image_instruction(self):
        text = build_pdf_multimodal_user_text(Path("paper.pdf"), "body")

        self.assertIn("Use both the extracted text and the attached page images", text)
        self.assertIn("PDF file: paper.pdf", text)
        self.assertTrue(text.endswith("body"))

    def test_build_pdf_multimodal_user_text_handles_missing_text(self):
        text = build_pdf_multimodal_user_text(Path("paper.pdf"), "")

        self.assertIn("No extractable text was found", text)
        self.assertIn("attached page images", text)

    def test_render_pdf_pages_to_image_urls_runs_pdftoppm(self):
        calls = []

        def runner(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            Path(f"{args[-1]}-2.png").write_bytes(b"page2")
            Path(f"{args[-1]}-1.png").write_bytes(b"page1")
            return FakeCompletedProcess()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            urls = render_pdf_pages_to_image_urls(
                path,
                dpi=96,
                max_pages=2,
                pdftoppm_command="pdftoppm",
                runner=runner,
            )

        self.assertEqual(
            calls[0]["args"][:-2],
            ["pdftoppm", "-png", "-r", "96", "-f", "1", "-l", "2"],
        )
        self.assertEqual(calls[0]["args"][-2], str(path))
        self.assertEqual(
            urls,
            [
                "data:image/png;base64,cGFnZTE=",
                "data:image/png;base64,cGFnZTI=",
            ],
        )

    def test_render_pdf_pages_to_image_urls_requires_pdftoppm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            with self.assertRaisesRegex(RuntimeError, "requires pdftoppm"):
                render_pdf_pages_to_image_urls(path, pdftoppm_command=None)

    def test_render_pdf_pages_to_image_urls_rejects_bad_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            with self.assertRaisesRegex(RuntimeError, "dpi"):
                render_pdf_pages_to_image_urls(path, dpi=0, pdftoppm_command="pdftoppm")
            with self.assertRaisesRegex(RuntimeError, "max-pages"):
                render_pdf_pages_to_image_urls(
                    path,
                    max_pages=0,
                    pdftoppm_command="pdftoppm",
                )

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
