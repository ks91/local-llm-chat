from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chat import (
    ChatSession,
    OpenAICompletionClient,
    format_assistant_output,
    is_stop_command,
    load_instructions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a local OpenAI-compatible completion server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--instructions", default="instructions.md")
    parser.add_argument("--model", default="local")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instructions = load_instructions(Path(args.instructions))
    client = OpenAICompletionClient(args.base_url, model=args.model, timeout=args.timeout)
    session = ChatSession(
        instructions=instructions,
        client=client,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("Local LLM chat. Type /quit or /exit to stop gracefully.")
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

        if not user_text.strip():
            continue
        if is_stop_command(user_text):
            print("Bye.")
            return 0

        try:
            answer = session.ask(user_text)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print()
        print(format_assistant_output(answer))


if __name__ == "__main__":
    raise SystemExit(main())
