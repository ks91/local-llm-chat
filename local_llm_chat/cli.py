from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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
_PROMPT_TOOLKIT_AUTO = object()
PASTE_COMMAND = "/paste"
PASTE_END_COMMANDS = {"/end", "/send"}
FILE_COMMAND = "/file"
_PYPDF_AUTO = object()
_PDFTOTEXT_AUTO = object()


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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable readline/libedit history and cursor bindings",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="show model thinking tags such as <think>...</think>",
    )
    parser.add_argument(
        "--show-emoji",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render non-BMP Unicode and \\U00000000-style emoji escapes",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        help="append raw and display-ready assistant output to a JSON Lines file",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="read a text PDF once, ask the model, print the answer, and exit",
    )
    return parser


def resolve_base_url(*, port: int, base_url: str | None) -> str:
    if base_url:
        return base_url
    return f"http://127.0.0.1:{port}"


def configure_line_editing(readline_module=_READLINE_AUTO, *, enabled: bool = True) -> bool:
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


def configure_input(
    *,
    line_editing: bool = True,
    input_func=input,
    readline_module=_READLINE_AUTO,
    prompt_toolkit_module=_PROMPT_TOOLKIT_AUTO,
):
    if not line_editing:
        return input_func

    if prompt_toolkit_module is _PROMPT_TOOLKIT_AUTO:
        try:
            from prompt_toolkit import prompt
            from prompt_toolkit.history import InMemoryHistory
        except ImportError:
            prompt = None
            InMemoryHistory = None
    elif prompt_toolkit_module is None:
        prompt = None
        InMemoryHistory = None
    else:
        prompt = prompt_toolkit_module.prompt
        InMemoryHistory = prompt_toolkit_module.InMemoryHistory

    if prompt is not None:
        history = InMemoryHistory()

        def prompt_with_history(message=""):
            return prompt(message, history=history)

        return prompt_with_history

    configure_line_editing(readline_module, enabled=True)
    return input_func


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


def extract_pdf_text_with_pdftotext(
    path: Path,
    *,
    pdftotext_command=_PDFTOTEXT_AUTO,
    runner=subprocess.run,
) -> str | None:
    if pdftotext_command is _PDFTOTEXT_AUTO:
        pdftotext_command = shutil.which("pdftotext")
    if not pdftotext_command:
        return None

    try:
        result = runner(
            [str(pdftotext_command), "-layout", str(path), "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def extract_pdf_text_with_pypdf(path: Path, *, pypdf_module=_PYPDF_AUTO) -> str:
    if pypdf_module is _PYPDF_AUTO:
        try:
            import pypdf as pypdf_module
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pdftotext or pypdf. Run scripts/install.sh, "
                "install poppler, or install pypdf."
            ) from exc
    if pypdf_module is None:
        raise RuntimeError("PDF support requires pdftotext or pypdf.")

    try:
        reader = pypdf_module.PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from {path}: {exc}") from exc

    return "\n\n".join(pages).strip()


def read_pdf_text(
    path: Path,
    *,
    pypdf_module=_PYPDF_AUTO,
    pdftotext_command=_PDFTOTEXT_AUTO,
) -> str:
    if path.suffix.lower() != ".pdf":
        raise RuntimeError(f"Expected a PDF file: {path}")
    if not path.exists():
        raise RuntimeError(f"PDF file does not exist: {path}")

    extracted = extract_pdf_text_with_pdftotext(
        path,
        pdftotext_command=pdftotext_command,
    )
    if extracted is None:
        extracted = extract_pdf_text_with_pypdf(path, pypdf_module=pypdf_module)

    if not extracted:
        raise RuntimeError(f"No extractable text found in {path}")
    return extracted


def build_pdf_user_text(path: Path, text: str) -> str:
    return (
        f"Read the following PDF text according to the system instructions.\n\n"
        f"PDF file: {path}\n\n"
        f"{text}"
    )


def append_output_log(
    path: Path,
    *,
    user_text: str,
    raw_answer: str,
    display_output: str,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "assistant_response",
        "user_text": user_text,
        "raw_answer": raw_answer,
        "display_output": display_output,
    }
    try:
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())
    except OSError as exc:
        raise RuntimeError(f"Could not write output log {path}: {exc}") from exc


def display_and_log_answer(
    *,
    answer: str,
    user_text: str,
    show_thinking: bool,
    show_emoji: bool,
    log_output: Path | None,
) -> None:
    display_output = format_assistant_output(
        answer,
        show_thinking=show_thinking,
        show_emoji=show_emoji,
    )
    if log_output:
        try:
            append_output_log(
                log_output,
                user_text=user_text,
                raw_answer=answer,
                display_output=display_output,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)

    print()
    print(display_output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

    if args.pdf:
        try:
            user_text = build_pdf_user_text(args.pdf, read_pdf_text(args.pdf))
            answer = session.ask(user_text)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        display_and_log_answer(
            answer=answer,
            user_text=user_text,
            show_thinking=args.show_thinking,
            show_emoji=args.show_emoji,
            log_output=args.log_output,
        )
        return 0

    input_func = configure_input(line_editing=args.line_editing)
    print("Local LLM chat. Type /bye, /quit, or /exit to stop gracefully.")
    print("Use /paste for multi-line input or /file path for long input.")
    if instructions:
        print(f"Loaded instructions from {args.instructions}.")
    else:
        print(f"No instructions loaded from {args.instructions}.")

    while True:
        try:
            user_text = input_func("\nYou> ")
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
            user_text = sanitize_user_input(read_paste_input(input_func=input_func))
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

        display_and_log_answer(
            answer=answer,
            user_text=user_text,
            show_thinking=args.show_thinking,
            show_emoji=args.show_emoji,
            log_output=args.log_output,
        )


if __name__ == "__main__":
    raise SystemExit(main())
