import unittest

from local_llm_chat.cli import resolve_base_url


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


if __name__ == "__main__":
    unittest.main()
