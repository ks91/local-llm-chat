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

Assistant responses are printed under an `LLM>` label. Multi-line responses are
displayed across multiple terminal lines.

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
  --max-tokens 512 \
  --temperature 0.7 \
  --timeout 120
```

Options:

- `--model`: model name sent in the JSON payload. Default: `local`
- `--max-tokens`: maximum tokens requested for each assistant response. Default: `512`
- `--temperature`: sampling temperature. Default: `0.7`
- `--timeout`: HTTP request timeout in seconds. Default: `120`

The default stop sequence is `\nUser:` so the server should stop before
generating the next user turn.

## Stop Gracefully

Inside the chat prompt, type either command:

```text
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
  "max_tokens": 512,
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
