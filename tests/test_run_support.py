import unittest

from local_llm_chat.run_support import model_log_name


class RunSupportTests(unittest.TestCase):
    def test_model_log_name_uses_basename_for_model_path(self):
        self.assertEqual(
            model_log_name("/Users/me/models/hermes4-14b/Hermes-4-14B-Q8_0.gguf"),
            "Hermes-4-14B",
        )

    def test_model_log_name_strips_common_quantization_suffixes(self):
        self.assertEqual(model_log_name("Qwen3.5-122B-A10B-Q4_K_M.gguf"), "Qwen3.5-122B-A10B")
        self.assertEqual(model_log_name("Model-IQ4_NL.gguf"), "Model")
        self.assertEqual(model_log_name("Model-F16.gguf"), "Model")

    def test_model_log_name_strips_split_gguf_shard_suffix(self):
        self.assertEqual(
            model_log_name(
                "/Volumes/ks91home/ks91/models/qwen3.5-122b/"
                "Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
            ),
            "Qwen3.5-122B-A10B",
        )

    def test_model_log_name_keeps_non_quantization_model_details(self):
        self.assertEqual(model_log_name("Qwen3.5-122B-A10B"), "Qwen3.5-122B-A10B")

    def test_model_log_name_sanitizes_for_filename(self):
        self.assertEqual(model_log_name("team/model name:latest"), "model_name_latest")

    def test_model_log_name_falls_back_to_local(self):
        self.assertEqual(model_log_name(""), "local")


if __name__ == "__main__":
    unittest.main()
