from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


STOP_COMMANDS = {"/quit", "/exit"}


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


def format_assistant_output(text: str) -> str:
    display_text = text
    if "\n" not in display_text:
        display_text = display_text.replace("\\r\\n", "\n").replace("\\n", "\n")
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
    max_tokens: int = 512
    temperature: float = 0.7
    stop: list[str] = field(default_factory=lambda: ["\nUser:"])
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
        self.add_turn(user_text, answer)
        return answer

    @staticmethod
    def _format_turns(turns: Iterable[tuple[str, str]]) -> str:
        return "\n".join(
            f"User: {user_text}\nAssistant: {assistant_text}"
            for user_text, assistant_text in turns
        )
