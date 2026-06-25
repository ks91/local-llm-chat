import unittest

from local_llm_chat.cli import configure_line_editing, resolve_base_url


class FakeReadline:
    def __init__(self):
        self.parse_and_bind_calls = []
        self.history_length = None

    def parse_and_bind(self, value):
        self.parse_and_bind_calls.append(value)

    def set_history_length(self, value):
        self.history_length = value


class CliTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
