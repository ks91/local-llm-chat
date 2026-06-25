from __future__ import annotations

import json
import re
import shutil
import textwrap
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


STOP_COMMANDS = {"/bye", "/exit", "/quit"}
DEFAULT_MAX_TOKENS = 2048
DEFAULT_STOP_SEQUENCES = ["\nUser:", "\nSystem instructions:"]

TERMINAL_ESCAPE_RE = re.compile(
    r"\x1b(?:"
    r"\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC: set title, clipboard, hyperlinks, etc.
    r"|P.*?(?:\x1b\\)?"  # DCS.
    r"|\[[0-?]*[ -/]*[@-~]"  # CSI: SGR, cursor movement, erase, etc.
    r"|[@-_]"  # Other 7-bit C1 controls.
    r")",
    re.DOTALL,
)
RAW_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
THINKING_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
SELF_CLOSING_THINK_RE = re.compile(r"<think\s*/\s*>", re.IGNORECASE)
UNICODE_ESCAPE_RE = re.compile(r"\\U([0-9a-fA-F]{8})|\\u([0-9a-fA-F]{4})")
RESTARTED_SYSTEM_LABEL_RE = re.compile(r"(?is)(^|\s)(system\s+instructions\s*:)")


class CompletionClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> str:
        ...


def load_instructions(path: str | Path) -> str:
    instruction_path = Path(path)
    if not instruction_path.exists():
        return ""
    return instruction_path.read_text(encoding="utf-8").strip()


def is_stop_command(text: str) -> bool:
    return text.strip() in STOP_COMMANDS


def sanitize_terminal_text(text: str) -> str:
    text = TERMINAL_ESCAPE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return RAW_CONTROL_RE.sub("", text)


def sanitize_user_input(text: str) -> str:
    return sanitize_terminal_text(text)


def strip_thinking(text: str) -> str:
    text = THINKING_RE.sub("", text)
    text = SELF_CLOSING_THINK_RE.sub("", text)
    return text.strip()


def strip_restarted_prompt(text: str) -> str:
    for system_match in RESTARTED_SYSTEM_LABEL_RE.finditer(text):
        label_start = system_match.start(2)
        line_start = text.rfind("\n", 0, label_start) + 1
        line_prefix = text[line_start:label_start]
        label_text = system_match.group(2)
        if not line_prefix.strip() or "\n" in label_text:
            return text[: system_match.start()].strip()

    kept_lines: list[str] = []
    for line in text.splitlines():
        normalized = line.strip().casefold()
        if normalized.startswith("user:"):
            break
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def render_unicode_escapes(text: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        hex_value = match.group(1) or match.group(2)
        try:
            return chr(int(hex_value, 16))
        except (OverflowError, ValueError):
            return match.group(0)

    return UNICODE_ESCAPE_RE.sub(replace_match, text)


def make_terminal_unicode_safe(text: str, *, allow_non_bmp: bool = False) -> str:
    safe_chars: list[str] = []
    for char in text:
        if char in {"\n", "\t"}:
            safe_chars.append(char)
            continue

        codepoint = ord(char)
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue
        if codepoint > 0xFFFF and not allow_non_bmp:
            safe_chars.append(f"\\U{codepoint:08x}")
            continue
        safe_chars.append(char)
    return "".join(safe_chars)


def terminal_wrap_width() -> int:
    columns = shutil.get_terminal_size(fallback=(100, 24)).columns
    return max(40, min(columns, 120))


def wrap_terminal_text(text: str, *, width: int | None = None) -> str:
    wrap_width = terminal_wrap_width() if width is None else width
    wrapped_lines: list[str] = []
    for line in text.split("\n"):
        if len(line) <= wrap_width:
            wrapped_lines.append(line)
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=wrap_width,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )
    return "\n".join(wrapped_lines)


def format_assistant_output(
    text: str,
    *,
    width: int | None = None,
    show_thinking: bool = False,
    show_emoji: bool = False,
) -> str:
    display_text = text
    if not show_thinking:
        display_text = strip_thinking(display_text)
    display_text = strip_restarted_prompt(display_text)
    if show_emoji:
        display_text = render_unicode_escapes(display_text)
    if "\n" not in display_text:
        display_text = display_text.replace("\\r\\n", "\n").replace("\\n", "\n")
    display_text = sanitize_terminal_text(display_text)
    display_text = make_terminal_unicode_safe(display_text, allow_non_bmp=show_emoji)
    display_text = wrap_terminal_text(display_text, width=width)
    return f"LLM>\n{display_text}"


class OpenAICompletionClient:
    def __init__(self, base_url: str, *, model: str = "local", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"LLM server request timed out after {self.timeout} seconds"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM server request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM server returned invalid JSON") from exc

        try:
            choice = body["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM server response did not include choices") from exc

        if "text" in choice:
            return str(choice["text"]).strip()
        if "message" in choice and isinstance(choice["message"], dict):
            return str(choice["message"].get("content", "")).strip()
        raise RuntimeError("LLM server response choice did not include text")


@dataclass
class ChatSession:
    instructions: str = ""
    client: CompletionClient | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.7
    show_thinking: bool = False
    stop: list[str] = field(default_factory=lambda: list(DEFAULT_STOP_SEQUENCES))
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.turns.append((user_text, assistant_text))

    def build_prompt(self, user_text: str) -> str:
        parts: list[str] = []
        if self.instructions:
            parts.append(f"System instructions:\n{self.instructions}")
        if self.turns:
            parts.append(self._format_turns(self.turns))
        parts.append(f"User: {user_text}\nAssistant:")
        return "\n\n".join(parts)

    def ask(self, user_text: str) -> str:
        if self.client is None:
            raise RuntimeError("ChatSession requires a completion client")

        prompt = self.build_prompt(user_text)
        answer = self.client.complete(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop,
        )
        if not self.show_thinking:
            answer = strip_thinking(answer)
        answer = strip_restarted_prompt(answer)
        self.add_turn(user_text, answer)
        return answer

    @staticmethod
    def _format_turns(turns: Iterable[tuple[str, str]]) -> str:
        return "\n".join(
            f"User: {user_text}\nAssistant: {assistant_text}"
            for user_text, assistant_text in turns
        )
