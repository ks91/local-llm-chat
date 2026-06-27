from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .chat import (
    ChatSession,
    DEFAULT_MAX_TOKENS,
    OpenAICompletionClient,
    format_assistant_output,
    is_stop_command,
    load_instructions,
    sanitize_user_input,
)


_READLINE_AUTO = object()
PASTE_COMMAND = "/paste"
PASTE_END_COMMANDS = {"/end", "/send"}
FILE_COMMAND = "/file"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a local OpenAI-compatible completion server."
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--base-url")
    parser.add_argument("--instructions", default="instructions.md")
    parser.add_argument("--model", default="local")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-continuations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--line-editing",
        action="store_true",
        help="enable readline/libedit history and cursor bindings",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="show model thinking tags such as <think>...</think>",
    )
    parser.add_argument(
        "--show-emoji",
        action="store_true",
        help="render non-BMP Unicode and \\U00000000-style emoji escapes",
    )
    return parser


def resolve_base_url(*, port: int, base_url: str | None) -> str:
    if base_url:
        return base_url
    return f"http://127.0.0.1:{port}"


def configure_line_editing(readline_module=_READLINE_AUTO, *, enabled: bool = False) -> bool:
    if not enabled:
        return False
    if readline_module is _READLINE_AUTO:
        try:
            import readline as readline_module
        except ImportError:
            return False
    if readline_module is None:
        return False

    readline_module.parse_and_bind("set editing-mode emacs")
    readline_module.parse_and_bind("tab: complete")
    readline_module.set_history_length(200)
    return True


def is_paste_command(text: str) -> bool:
    return text.strip() == PASTE_COMMAND


def is_file_command(text: str) -> bool:
    return text.strip().startswith(f"{FILE_COMMAND} ")


def read_paste_input(input_func=input, print_func=print) -> str:
    print_func("Paste multi-line input. Finish with /send or /end on its own line.")
    lines: list[str] = []
    while True:
        try:
            line = input_func()
        except EOFError:
            break
        if line.strip() in PASTE_END_COMMANDS:
            break
        lines.append(line)
    return "\n".join(lines)


def read_message_file(command: str, *, base_dir: Path | None = None) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"Invalid /file command: {exc}") from exc

    if len(parts) != 2 or parts[0] != FILE_COMMAND:
        raise RuntimeError("Usage: /file path/to/message.txt")

    path = Path(parts[1]).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_line_editing(enabled=args.line_editing)
    instructions = load_instructions(Path(args.instructions))
    base_url = resolve_base_url(port=args.port, base_url=args.base_url)
    client = OpenAICompletionClient(base_url, model=args.model, timeout=args.timeout)
    session = ChatSession(
        instructions=instructions,
        client=client,
        max_tokens=args.max_tokens,
        max_continuations=args.max_continuations,
        temperature=args.temperature,
        show_thinking=args.show_thinking,
    )

    print("Local LLM chat. Type /bye, /quit, or /exit to stop gracefully.")
    print("Use /paste for multi-line input or /file path for long input.")
    if instructions:
        print(f"Loaded instructions from {args.instructions}.")
    else:
        print(f"No instructions loaded from {args.instructions}.")

    while True:
        try:
            user_text = input("\nYou> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            return 130

        user_text = sanitize_user_input(user_text)
        if not user_text.strip():
            continue
        if is_stop_command(user_text):
            print("Bye.")
            return 0
        if is_paste_command(user_text):
            user_text = sanitize_user_input(read_paste_input())
            if not user_text.strip():
                continue
        elif is_file_command(user_text):
            try:
                user_text = sanitize_user_input(read_message_file(user_text))
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                continue
            if not user_text.strip():
                continue

        try:
            answer = session.ask(user_text)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("You can retry, use --timeout to wait longer, or type /bye to exit.")
            continue

        print()
        print(
            format_assistant_output(
                answer,
                show_thinking=args.show_thinking,
                show_emoji=args.show_emoji,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
