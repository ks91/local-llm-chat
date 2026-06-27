import json
import unittest
from unittest.mock import patch

from local_llm_chat.chat import (
    ChatSession,
    DEFAULT_MAX_TOKENS,
    OpenAICompletionClient,
    format_assistant_output,
    is_stop_command,
    make_terminal_unicode_safe,
    render_unicode_escapes,
    sanitize_user_input,
    sanitize_terminal_text,
    strip_restarted_prompt,
    strip_thinking,
    wrap_terminal_text,
)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, prompt, *, max_tokens, temperature, stop):
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
        )
        return self.replies.pop(0)


class FakeContinuationClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_with_metadata(self, prompt, *, max_tokens, temperature, stop):
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
        )
        return self.replies.pop(0)


class ChatSessionTests(unittest.TestCase):
    def test_build_prompt_includes_instructions_and_history(self):
        session = ChatSession(instructions="Be concise.")
        session.add_turn("hello", "hi")

        prompt = session.build_prompt("what now?")

        self.assertIn("System instructions:\nBe concise.", prompt)
        self.assertIn("User: hello\nAssistant: hi", prompt)
        self.assertTrue(prompt.endswith("User: what now?\nAssistant:"))

    def test_ask_preserves_context_for_lifetime_of_session(self):
        client = FakeClient(["hello", "second answer"])
        session = ChatSession(instructions="Use Japanese.", client=client)

        first = session.ask("one")
        second = session.ask("two")

        self.assertEqual(first, "hello")
        self.assertEqual(second, "second answer")
        self.assertIn("User: one\nAssistant: hello", client.calls[1]["prompt"])
        self.assertIn("User: two\nAssistant:", client.calls[1]["prompt"])
        self.assertEqual(client.calls[0]["max_tokens"], DEFAULT_MAX_TOKENS)

    def test_ask_strips_thinking_from_history_by_default(self):
        client = FakeClient(["<think>private</think>\nfinal"])
        session = ChatSession(client=client)

        self.assertEqual(session.ask("one"), "final")
        self.assertEqual(session.turns, [("one", "final")])

    def test_ask_strips_restarted_system_instructions_from_history(self):
        client = FakeClient(["final answer\n\nSystem instructions:\nDo not leak."])
        session = ChatSession(client=client)

        self.assertEqual(session.ask("one"), "final answer")
        self.assertEqual(session.turns, [("one", "final answer")])

    def test_default_stop_sequences_include_prompt_restart_labels(self):
        client = FakeClient(["hello"])
        session = ChatSession(client=client)

        session.ask("one")

        self.assertIn("\nUser:", client.calls[0]["stop"])
        self.assertIn("\nAssistant:", client.calls[0]["stop"])
        self.assertIn("\nSystem instructions:", client.calls[0]["stop"])

    def test_ask_can_preserve_thinking_when_enabled(self):
        client = FakeClient(["<think>private</think>\nfinal"])
        session = ChatSession(client=client, show_thinking=True)

        self.assertEqual(session.ask("one"), "<think>private</think>\nfinal")
        self.assertEqual(session.turns, [("one", "<think>private</think>\nfinal")])

    def test_stop_commands(self):
        self.assertTrue(is_stop_command("/quit"))
        self.assertTrue(is_stop_command(" /exit "))
        self.assertTrue(is_stop_command("/bye"))
        self.assertFalse(is_stop_command("quit"))

    def test_format_assistant_output_preserves_real_newlines(self):
        self.assertEqual(
            format_assistant_output("first\nsecond\nthird"),
            "LLM>\nfirst\nsecond\nthird",
        )

    def test_strip_thinking_removes_standard_think_block(self):
        self.assertEqual(
            strip_thinking("<think>\nprivate\n</think>\nanswer"),
            "answer",
        )

    def test_strip_thinking_removes_self_closing_think_tag(self):
        self.assertEqual(strip_thinking("<think/>\nanswer"), "answer")

    def test_strip_thinking_handles_case_insensitive_tags(self):
        self.assertEqual(strip_thinking("<THINK>private</Think>\nanswer"), "answer")

    def test_strip_thinking_removes_unclosed_think_suffix(self):
        self.assertEqual(strip_thinking("<think>\nlong private reasoning"), "")

    def test_strip_thinking_keeps_text_before_unclosed_think_suffix(self):
        self.assertEqual(strip_thinking("answer\n<think>\ntruncated"), "answer")

    def test_ask_strips_unclosed_thinking_from_history_by_default(self):
        client = FakeClient(["<think>long private reasoning"])
        session = ChatSession(client=client)

        self.assertEqual(session.ask("one"), "")
        self.assertEqual(session.turns, [("one", "")])

    def test_ask_preserves_unclosed_thinking_when_enabled(self):
        client = FakeClient(["<think>long private reasoning"])
        session = ChatSession(client=client, show_thinking=True)

        self.assertEqual(session.ask("one"), "<think>long private reasoning")
        self.assertEqual(session.turns, [("one", "<think>long private reasoning")])

    def test_ask_continues_when_finish_reason_is_length(self):
        client = FakeContinuationClient(
            [
                {"text": "<think>long", "finish_reason": "length"},
                {"text": " private</think>\nanswer", "finish_reason": "stop"},
            ]
        )
        session = ChatSession(client=client, max_continuations=2)

        self.assertEqual(session.ask("one"), "answer")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("<think>long", client.calls[1]["prompt"])

    def test_ask_stops_after_max_continuations(self):
        client = FakeContinuationClient(
            [
                {"text": "part 1", "finish_reason": "length"},
                {"text": " part 2", "finish_reason": "length"},
            ]
        )
        session = ChatSession(client=client, max_continuations=1, show_thinking=True)

        self.assertEqual(session.ask("one"), "part 1 part 2")
        self.assertEqual(len(client.calls), 2)

    def test_strip_restarted_prompt_removes_system_instructions_suffix(self):
        self.assertEqual(
            strip_restarted_prompt("answer\n\nSystem instructions:\nBe terse."),
            "answer",
        )

    def test_strip_restarted_prompt_removes_japanese_answer_suffix(self):
        text = (
            "ありがとうございます。私もより良いアシスタントになるために、"
            "引き続き改善に努めます。何かお手伝いできることがあれば、"
            "何でもお気軽にお問い合わせください。 \n"
            "System instructions:\n"
            "You are a concise, practical assistant.\n\n"
            "Answer in the same language as the user unless asked otherwise."
        )

        self.assertEqual(
            strip_restarted_prompt(text),
            "ありがとうございます。私もより良いアシスタントになるために、"
            "引き続き改善に努めます。何かお手伝いできることがあれば、"
            "何でもお気軽にお問い合わせください。",
        )

    def test_strip_restarted_prompt_removes_wrapped_system_instructions_suffix(self):
        text = (
            "メタ認知はまた、問題解決や意思決定にも関与します。これは、"
            "あなたが自分の思考プロセスを監視し、必要に応じて修正することを"
            "可能にします。 System \n"
            "instructions:\n"
            "You are a concise, practical assistant.\n\n"
            "Answer in the same language as the user unless asked otherwise."
        )

        self.assertEqual(
            strip_restarted_prompt(text),
            "メタ認知はまた、問題解決や意思決定にも関与します。これは、"
            "あなたが自分の思考プロセスを監視し、必要に応じて修正することを"
            "可能にします。",
        )

    def test_strip_restarted_prompt_removes_user_suffix(self):
        self.assertEqual(strip_restarted_prompt("answer\nUser: next"), "answer")

    def test_strip_restarted_prompt_removes_assistant_suffix(self):
        self.assertEqual(strip_restarted_prompt("answer\nAssistant: duplicate"), "answer")

    def test_strip_restarted_prompt_removes_qwen_duplicate_assistant_suffix(self):
        text = (
            "カー解は角運動量を持つブラックホールの時空構造を記述します。\n\n"
            "Assistant:\n\n"
            "ングシンギュラリティ）です。"
        )

        self.assertEqual(
            strip_restarted_prompt(text),
            "カー解は角運動量を持つブラックホールの時空構造を記述します。",
        )

    def test_strip_restarted_prompt_keeps_inline_assistant_mentions(self):
        self.assertEqual(
            strip_restarted_prompt("The token Assistant: is shown inline."),
            "The token Assistant: is shown inline.",
        )

    def test_strip_restarted_prompt_keeps_inline_mentions(self):
        self.assertEqual(
            strip_restarted_prompt("The phrase System instructions: is shown inline."),
            "The phrase System instructions: is shown inline.",
        )

    def test_format_assistant_output_strips_restarted_prompt_before_display(self):
        self.assertEqual(
            format_assistant_output("answer\nSystem instructions:\nsecret"),
            "LLM>\nanswer",
        )

    def test_format_assistant_output_hides_thinking_by_default(self):
        self.assertEqual(
            format_assistant_output("<think>private</think>\nanswer"),
            "LLM>\nanswer",
        )

    def test_format_assistant_output_can_show_thinking(self):
        self.assertEqual(
            format_assistant_output("<think>private</think>\nanswer", show_thinking=True),
            "LLM>\n<think>private</think>\nanswer",
        )

    def test_format_assistant_output_hides_unclosed_thinking_by_default(self):
        self.assertEqual(
            format_assistant_output("<think>long private reasoning"),
            "LLM>\n",
        )

    def test_format_assistant_output_expands_escaped_newlines_for_display(self):
        self.assertEqual(
            format_assistant_output("first\\nsecond\\nthird"),
            "LLM>\nfirst\nsecond\nthird",
        )

    def test_format_assistant_output_removes_ansi_control_sequences(self):
        self.assertEqual(
            format_assistant_output("\x1b[31mred\x1b[0m plain"),
            "LLM>\nred plain",
        )

    def test_sanitize_terminal_text_removes_osc_sequences(self):
        self.assertEqual(
            sanitize_terminal_text("before\x1b]0;bad title\x07after"),
            "beforeafter",
        )

    def test_sanitize_terminal_text_removes_raw_control_chars(self):
        self.assertEqual(
            sanitize_terminal_text("a\x00b\x08c\r\nd\re\tf"),
            "abc\nd\ne\tf",
        )

    def test_sanitize_user_input_removes_terminal_controls_before_prompt(self):
        self.assertEqual(
            sanitize_user_input("hello\x1b[2J world\x00"),
            "hello world",
        )

    def test_make_terminal_unicode_safe_keeps_japanese_text(self):
        self.assertEqual(
            make_terminal_unicode_safe("日本語 カタカナ ひらがな"),
            "日本語 カタカナ ひらがな",
        )

    def test_make_terminal_unicode_safe_removes_format_controls(self):
        self.assertEqual(
            make_terminal_unicode_safe("a\u200db\u202ec"),
            "abc",
        )

    def test_make_terminal_unicode_safe_escapes_non_bmp_chars(self):
        self.assertEqual(
            make_terminal_unicode_safe("ok \U0001f600"),
            "ok \\U0001f600",
        )

    def test_make_terminal_unicode_safe_can_keep_non_bmp_chars(self):
        self.assertEqual(
            make_terminal_unicode_safe("ok \U0001f9e9", allow_non_bmp=True),
            "ok \U0001f9e9",
        )

    def test_format_assistant_output_makes_unicode_safe_for_display(self):
        self.assertEqual(
            format_assistant_output("ok \U0001f600"),
            "LLM>\nok \\U0001f600",
        )

    def test_render_unicode_escapes_turns_qwen_style_emoji_into_text(self):
        self.assertEqual(render_unicode_escapes("piece \\U0001f9e9"), "piece \U0001f9e9")

    def test_render_unicode_escapes_ignores_invalid_codepoints(self):
        self.assertEqual(render_unicode_escapes("bad \\Uffffffff"), "bad \\Uffffffff")

    def test_format_assistant_output_can_show_qwen_style_emoji(self):
        self.assertEqual(
            format_assistant_output("piece \\U0001f9e9", show_emoji=True),
            "LLM>\npiece \U0001f9e9",
        )

    def test_format_assistant_output_keeps_qwen_style_emoji_escaped_by_default(self):
        self.assertEqual(
            format_assistant_output("piece \\U0001f9e9"),
            "LLM>\npiece \\U0001f9e9",
        )

    def test_wrap_terminal_text_wraps_long_lines(self):
        self.assertEqual(
            wrap_terminal_text("abcdefghij", width=4),
            "abcd\nefgh\nij",
        )

    def test_wrap_terminal_text_preserves_existing_line_breaks(self):
        self.assertEqual(
            wrap_terminal_text("abcde\nfghij", width=3),
            "abc\nde\nfgh\nij",
        )

    def test_format_assistant_output_wraps_long_lines(self):
        self.assertEqual(
            format_assistant_output("abcdefghij", width=4),
            "LLM>\nabcd\nefgh\nij",
        )


class OpenAICompletionClientTests(unittest.TestCase):
    def test_complete_posts_to_v1_completions_and_returns_text(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"choices": [{"text": " answer "}]}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["headers"] = dict(request.header_items())
            return FakeResponse()

        client = OpenAICompletionClient("http://127.0.0.1:8080")

        with patch("urllib.request.urlopen", fake_urlopen):
            text = client.complete(
                "User: hi\nAssistant:",
                max_tokens=64,
                temperature=0.2,
                stop=["\nUser:"],
            )

        self.assertEqual(text, "answer")
        self.assertEqual(captured["url"], "http://127.0.0.1:8080/v1/completions")
        self.assertEqual(captured["timeout"], 120)
        self.assertEqual(captured["body"]["model"], "local")
        self.assertEqual(captured["body"]["prompt"], "User: hi\nAssistant:")
        self.assertEqual(captured["body"]["max_tokens"], 64)
        self.assertEqual(captured["body"]["temperature"], 0.2)
        self.assertEqual(captured["body"]["stop"], ["\nUser:"])
        self.assertEqual(captured["headers"]["Content-type"], "application/json")

    def test_complete_with_metadata_returns_finish_reason(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"text": " answer ", "finish_reason": "length"}]}
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            return FakeResponse()

        client = OpenAICompletionClient("http://127.0.0.1:8080")

        with patch("urllib.request.urlopen", fake_urlopen):
            result = client.complete_with_metadata(
                "User: hi\nAssistant:",
                max_tokens=64,
                temperature=0.2,
                stop=["\nUser:"],
            )

        self.assertEqual(result, {"text": "answer", "finish_reason": "length"})

    def test_complete_wraps_timeout_error(self):
        def fake_urlopen(request, timeout):
            raise TimeoutError("timed out")

        client = OpenAICompletionClient("http://127.0.0.1:8080", timeout=3)

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(
                RuntimeError,
                "LLM server request timed out after 3 seconds",
            ):
                client.complete(
                    "User: hi\nAssistant:",
                    max_tokens=64,
                    temperature=0.2,
                    stop=["\nUser:"],
                )


if __name__ == "__main__":
    unittest.main()
