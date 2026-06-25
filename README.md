# local-llm-chat

Terminal chat for a local OpenAI-compatible completion server.

This is meant for a `llama-server` process that exposes the OpenAI-compatible
completion API at `/v1/completions`. Conversation history is kept only in the
running Python process. If you stop and start this client again, the chat starts
with no previous context.

The client uses only the Python standard library.

## Requirements

- Python 3.10 or newer
- A local OpenAI-compatible completion server

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
python3 -m local_llm_chat
```

You will see a prompt like this:

```text
You>
```

Input uses the terminal's normal line editing. Backspace edits the current line
before Python receives it. Optional `readline`/libedit history and cursor
bindings can be enabled with `--line-editing`, but they are off by default to
keep terminal interaction conservative. Chat input history is not saved after
the process exits.

Assistant responses are printed under an `LLM>` label. Multi-line responses are
displayed across multiple terminal lines. Long lines are wrapped before display
to avoid stressing terminal rendering with very wide generated lines.

Terminal control sequences in model output, such as ANSI color codes, OSC title
changes, cursor movement, and raw control characters, are removed before display.
This keeps generated text from changing terminal state or triggering terminal
rendering bugs.

For Apple Terminal stability, display output also removes invisible Unicode
format/control characters and escapes non-BMP Unicode such as many emoji as
literal `\U00000000`-style text. Common Japanese text remains unchanged.

If your terminal handles emoji reliably, enable emoji rendering:

```sh
python3 -m local_llm_chat --show-emoji
```

This also converts model output such as `\U0001f9e9` into the corresponding
emoji for display.

User input is also sanitized before it is added to the in-memory conversation
history, so pasted terminal control sequences are not sent back to the model in
later prompts.

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
  --temperature 0.7 \
  --timeout 120 \
  --line-editing \
  --show-thinking \
  --show-emoji
```

Options:

- `--model`: model name sent in the JSON payload. Default: `local`
- `--max-tokens`: maximum tokens requested for each assistant response. Default: `2048`
- `--temperature`: sampling temperature. Default: `0.7`
- `--timeout`: HTTP request timeout in seconds. Default: `120`
- `--line-editing`: enable optional in-process readline/libedit input history
  and cursor bindings. Default: off
- `--show-thinking`: show model thinking tags such as `<think>...</think>`.
  Default: off
- `--show-emoji`: render non-BMP Unicode and `\U00000000`-style emoji escapes
  such as `\U0001f9e9`. Default: off

## Thinking Tags

Some models, including Qwen-style reasoning models, may return thinking markup
such as:

```text
<think>
private reasoning
</think>
final answer
```

By default, `<think>...</think>` blocks and self-closing `<think/>` tags are
removed before the response is displayed or saved in the in-memory conversation
history. This keeps future prompts focused on the visible answer.

To show and preserve those tags during the current session:

```sh
python3 -m local_llm_chat --show-thinking
```

The default stop sequence is `\nUser:` so the server should stop before
generating the next user turn.

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
