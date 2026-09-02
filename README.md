# local-llm-chat

Terminal chat for a local OpenAI-compatible completion server.

This is meant for a `llama-server` process that exposes the OpenAI-compatible
completion API at `/v1/completions`. Conversation history is kept only in the
running Python process. If you stop and start this client again, the chat starts
with no previous context.

The client works with only the Python standard library. If the optional
`prompt_toolkit` package is installed, it is used automatically for better
interactive line editing.

## Requirements

- Python 3.10 or newer
- A local OpenAI-compatible completion server

Optional, for more reliable cursor movement, wrapping, and backspace behavior
with long or non-ASCII input:

```sh
scripts/install.sh
```

The install script creates `.venv` in this repository and installs
`prompt_toolkit` and `pypdf` there. It also installs Homebrew `poppler` when
`pdftotext` or `pdftoppm` is not already available. It does not modify the
Homebrew/system Python environment.

## Start the model server

Example `llama-server` command:

```sh
llama-server \
  -m ~/models/hermes4-14b/Hermes-4-14B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -ngl 999
```

The chat client defaults to `http://127.0.0.1:8080`.

## Start chatting

From this repository:

```sh
scripts/run.sh
```

The run script uses `.venv/bin/python` when it exists, otherwise it falls back
to `python3`. Any options passed to the script are passed through to
`local_llm_chat`.

By default, `scripts/run.sh` enables `--log-output` and writes JSON Lines logs
under `logs/` using this filename shape:

```text
logs/<model>-log-YYYYMMDD-HHMMSS-pid<PID>.jsonl
```

The model name is read from the server's OpenAI-compatible `/v1/models`
endpoint when available. If the returned name is a model file path, the log name
uses only the basename. Common file extensions and quantization suffixes such
as `.gguf`, `Q8_0`, and `Q4_K_M` are removed. If the server is not running or
does not expose that endpoint, the script uses `local`.

Pass `--log-output path/to/file.jsonl` yourself to override the default log
path.

You can still run the module directly:

```sh
python3 -m local_llm_chat
```

Show the client version:

```sh
python3 -m local_llm_chat --version
```

## Interactive Chat

You will see a prompt like this:

```text
You>
```

Input uses the terminal's normal line editing. Backspace edits the current line
before Python receives it. `readline`/libedit history and cursor bindings are
enabled by default, including Emacs-style bindings where supported. Chat input
history is not saved after the process exits.

If `prompt_toolkit` is installed, the client uses it instead of Python's
`readline` module. This usually gives better display behavior for wrapped lines
and Japanese text. Without it, the client falls back to the standard-library
`readline`/libedit support.

During one running session, previous input can be recalled with Up/Down or
Ctrl-P/Ctrl-N. Input history is in memory only and is discarded when the client
exits.

Disable in-process line editing if you need the most conservative terminal
interaction:

```sh
python3 -m local_llm_chat --no-line-editing
```

Very long single-line input may hit the terminal driver's line length limit
before Python receives it. For long prompts, use multi-line paste mode:

```text
You> /paste
Paste multi-line input. Finish with /send or /end on its own line.
...paste or type text here...
/send
```

Or put the prompt in a UTF-8 text file and send it with:

```text
You> /file path/to/message.txt
```

Assistant responses are printed under an `LLM>` label. Multi-line responses are
displayed across multiple terminal lines. Long lines are wrapped before display
to avoid stressing terminal rendering with very wide generated lines.

Terminal control sequences in model output, such as ANSI color codes, OSC title
changes, cursor movement, and raw control characters, are removed before display.
This keeps generated text from changing terminal state or triggering terminal
rendering bugs.

For Apple Terminal stability, display output also removes invisible Unicode
format/control characters. Non-BMP Unicode and `\U00000000`-style emoji escapes
are rendered by default.

Disable emoji rendering if a terminal has trouble with those glyphs:

```sh
python3 -m local_llm_chat --no-show-emoji
```

To debug terminal rendering crashes, save each response before it is printed:

```sh
python3 -m local_llm_chat --log-output output.jsonl
```

The log is appended as UTF-8 JSON Lines. Each record contains the sanitized user
message, the raw assistant response, and the display-ready text that was about
to be printed. The client flushes the file after each response so the last
record is likely to survive even if the terminal application crashes.

Emoji rendering also converts model output such as `\U0001f9e9` into the
corresponding emoji for display.

User input is also sanitized before it is added to the in-memory conversation
history, so pasted terminal control sequences are not sent back to the model in
later prompts.

## Read a PDF

Pass a PDF path to read it once and exit:

```sh
scripts/run.sh paper.pdf
```

The client extracts text with `pdftotext -layout` when available, falling back
to `pypdf` otherwise. `pdftotext` is usually more reliable for Japanese PDFs.
The extracted text is sent as a single user message with the configured
instructions, the answer is printed without the interactive `LLM>` label, and
the process exits.

Use a custom instruction file:

```sh
scripts/run.sh --instructions summarize.md paper.pdf
```

For a multimodal model served by `llama-server`, also send rendered page images:

```sh
scripts/run.sh --pdf-images --instructions summarize.md paper.pdf
```

With `--pdf-images`, the client still extracts text first, then renders pages
with `pdftoppm` and sends both the extracted text and PNG page images through
`/v1/chat/completions`. This helps with scanned PDFs, tables, figures, or PDFs
whose text extraction is unreliable. The default image render limit is the first
8 pages at 144 DPI:

```sh
scripts/run.sh --pdf-images --pdf-image-max-pages 12 --pdf-image-dpi 120 paper.pdf
```

`--pdf-images` requires a vision-capable model and a server that accepts
OpenAI-style `image_url` content. `scripts/install.sh` installs Homebrew
`poppler` when needed, which provides both `pdftotext` and `pdftoppm`.

## Batch PDF Processing

Process every top-level PDF file in a directory and write one Markdown file per
PDF:

```sh
scripts/run.sh --batch-pdf-dir input-pdfs --batch-output-dir outputs \
  --instructions summarize.md
```

If `--batch-pdf-dir` is provided without a value, `inputs/` is used. The
default output directory is `outputs/`:

```sh
scripts/run.sh --batch-pdf-dir --instructions summarize.md
```

For each `name.pdf`, the client writes `outputs/name.md`. Existing output files
with the same name are overwritten. Each PDF is processed independently with
the configured instructions; chat context is not carried from one PDF to the
next. If one PDF fails, the batch continues and exits with status `1` after the
remaining files have been attempted.

Batch processing can also use rendered PDF page images:

```sh
scripts/run.sh --batch-pdf-dir input-pdfs --batch-output-dir outputs \
  --pdf-images --instructions summarize.md
```

## Initial Instructions

At startup, the client reads `instructions.md` from the current directory. Edit
that file to change the system-style instructions sent at the start of every
prompt.

Use a different instructions file:

```sh
python3 -m local_llm_chat --instructions my-instructions.md
```

If the file does not exist, the client starts without initial instructions.

## Connection Options

Use the default local server on port 8080:

```sh
python3 -m local_llm_chat
```

Use another local port:

```sh
python3 -m local_llm_chat --port 8081
```

Use a full base URL for another host or scheme:

```sh
python3 -m local_llm_chat --base-url http://192.168.1.10:8080
```

If `--base-url` is set, it takes precedence over `--port`.

## Generation Options

The client sends these completion parameters:

```sh
python3 -m local_llm_chat \
  --model local \
  --max-tokens 2048 \
  --max-continuations 2 \
  --temperature 0.7 \
  --timeout 120 \
  --line-editing \
  --no-line-editing \
  --show-thinking \
  --show-emoji \
  --no-show-emoji \
  --log-output output.jsonl \
  --batch-pdf-dir input-pdfs \
  --batch-output-dir outputs
```

Options:

- `--model`: model name sent in the JSON payload. Default: `local`
- `--max-tokens`: maximum tokens requested for each assistant response. Default: `2048`
- `--max-continuations`: additional completion requests when the server reports
  `finish_reason: length`. Default: `2`
- `--temperature`: sampling temperature. Default: `0.7`
- `--timeout`: HTTP request timeout in seconds. Default: `120`
- `--line-editing` / `--no-line-editing`: enable or disable in-process
  readline/libedit input history and cursor bindings. Default: on
- `--show-thinking`: show model thinking tags such as `<think>...</think>`.
  Default: off
- `--show-emoji` / `--no-show-emoji`: render or escape non-BMP Unicode and
  `\U00000000`-style emoji escapes such as `\U0001f9e9`. Default: on
- `--log-output`: append raw and display-ready assistant output to a UTF-8 JSON
  Lines file for terminal crash debugging. Default: off
- `--batch-pdf-dir`: process every top-level PDF file in a directory and write
  Markdown outputs. If the option is present without a value, the input
  directory is `inputs`. Default: off
- `--batch-output-dir`: output directory for batch Markdown files. Default:
  `outputs`

## Thinking Tags

Some models, including Qwen-style reasoning models, may return thinking markup
such as:

```text
<think>
private reasoning
</think>
final answer
```

By default, `<think>...</think>` blocks, self-closing `<think/>` tags, and
truncated unclosed `<think>` blocks are removed before the response is displayed
or saved in the in-memory conversation history. This keeps future prompts
focused on the visible answer.

To show and preserve those tags during the current session:

```sh
python3 -m local_llm_chat --show-thinking
```

When `--show-thinking` is enabled, the client also adds a short instruction that
asks the model to use a brief `<think>...</think>` block before the final answer
when useful, and to close `</think>` before writing the final answer.

The default stop sequences include `\nUser:`, `\nAssistant:`, and
`\nSystem instructions:` so the server should stop before generating another
prompt segment. The client also removes restarted prompt labels from the tail of
a response. This helps with completion models that occasionally continue by
echoing the prompt scaffold instead of stopping after the answer.

If a local model is slow, increase the timeout:

```sh
python3 -m local_llm_chat --timeout 300
```

Timeouts and other request errors are printed without a Python traceback, and
the chat prompt stays open so you can retry or exit with `/bye`.

## Stop Gracefully

Inside the chat prompt, type any of these commands:

```text
/bye
/quit
/exit
```

Other exits:

- Ctrl-D exits cleanly with status `0`
- Ctrl-C exits with status `130`

## API Shape

Requests are sent as `POST` JSON to:

```text
<base-url>/v1/completions
```

Payload fields:

```json
{
  "model": "local",
  "prompt": "...",
  "max_tokens": 2048,
  "temperature": 0.7,
  "stop": ["\nUser:"]
}
```

The client reads `choices[0].text` from the response. It also accepts
`choices[0].message.content` as a fallback.

## Test

Run all tests:

```sh
python3 -m unittest discover -s tests
```

Compile-check the code:

```sh
python3 -m compileall local_llm_chat tests
```
