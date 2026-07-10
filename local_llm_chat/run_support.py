from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import PurePosixPath


QUANTIZATION_SUFFIX_RE = re.compile(
    r"(?i)(?:[-_.](?:"
    r"q[2-8](?:_[0-9a-z]+)*"
    r"|iq[1-4](?:_[0-9a-z]+)*"
    r"|f(?:16|32)"
    r"))$"
)
GGUF_SHARD_SUFFIX_RE = re.compile(r"(?i)-\d{5}-of-\d{5}$")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def model_log_name(model_name: str) -> str:
    name = str(model_name).strip()
    if not name:
        return "local"

    name = PurePosixPath(name.replace("\\", "/")).name
    for suffix in (".gguf", ".bin", ".safetensors"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break

    name = GGUF_SHARD_SUFFIX_RE.sub("", name)
    name = QUANTIZATION_SUFFIX_RE.sub("", name)
    name = SAFE_FILENAME_RE.sub("_", name).strip("._-")
    return name or "local"


def discover_model_name(base_url: str, *, timeout: int = 2) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=timeout) as response:
        body = json.load(response)

    models = body.get("data", [])
    if not models:
        return "local"

    first = models[0]
    if isinstance(first, dict):
        return str(first.get("id") or first.get("name") or "local")
    return str(first)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m local_llm_chat.run_support BASE_URL", file=sys.stderr)
        return 2

    try:
        model_name = discover_model_name(args[0])
    except Exception:
        model_name = "local"
    print(model_log_name(model_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
