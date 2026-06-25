# local-llm-chat

Terminal chat for a local OpenAI-compatible completion server such as
`llama-server` on port 8080.

## Start the model server

```sh
llama-server \
  -m ~/models/hermes4-14b/Hermes-4-14B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -ngl 999
```

## Chat

```sh
python3 -m local_llm_chat
```

The process keeps conversation context in memory only. Restarting the program
starts a fresh conversation.

Initial behavior is read from `instructions.md`. You can point at another file:

```sh
python3 -m local_llm_chat --instructions my-instructions.md
```

Graceful stop commands:

```text
/quit
/exit
```

Ctrl-D also exits cleanly. Ctrl-C exits with status 130.

## Test

```sh
python3 -m unittest discover -s tests
```
