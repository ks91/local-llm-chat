import json
import unittest
from unittest.mock import patch

from local_llm_chat.chat import (
    ChatSession,
    OpenAICompletionClient,
    format_assistant_output,
    is_stop_command,
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

    def test_stop_commands(self):
        self.assertTrue(is_stop_command("/quit"))
        self.assertTrue(is_stop_command(" /exit "))
        self.assertFalse(is_stop_command("quit"))

    def test_format_assistant_output_preserves_real_newlines(self):
        self.assertEqual(
            format_assistant_output("first\nsecond\nthird"),
            "LLM>\nfirst\nsecond\nthird",
        )

    def test_format_assistant_output_expands_escaped_newlines_for_display(self):
        self.assertEqual(
            format_assistant_output("first\\nsecond\\nthird"),
            "LLM>\nfirst\nsecond\nthird",
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


if __name__ == "__main__":
    unittest.main()
