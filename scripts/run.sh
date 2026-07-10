#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

if [ -x "$repo_dir/.venv/bin/python" ]; then
  python_bin="$repo_dir/.venv/bin/python"
else
  python_bin=${PYTHON:-python3}
fi

base_url=""
port="8080"
has_log_output=0
next_is_base_url=0
next_is_port=0

for arg in "$@"; do
  if [ "$next_is_base_url" -eq 1 ]; then
    base_url=$arg
    next_is_base_url=0
    continue
  fi
  if [ "$next_is_port" -eq 1 ]; then
    port=$arg
    next_is_port=0
    continue
  fi

  case "$arg" in
    --base-url)
      next_is_base_url=1
      ;;
    --base-url=*)
      base_url=${arg#--base-url=}
      ;;
    --port)
      next_is_port=1
      ;;
    --port=*)
      port=${arg#--port=}
      ;;
    --log-output|--log-output=*)
      has_log_output=1
      ;;
  esac
done

if [ -z "$base_url" ]; then
  base_url="http://127.0.0.1:$port"
fi

safe_model=$("$python_bin" -m local_llm_chat.run_support "$base_url")
timestamp=$(date '+%Y%m%d-%H%M%S')
log_path="$repo_dir/logs/${safe_model}-log-${timestamp}-pid$$.jsonl"

mkdir -p "$repo_dir/logs"
cd "$repo_dir"

if [ "$has_log_output" -eq 1 ]; then
  exec "$python_bin" -m local_llm_chat "$@"
fi

exec "$python_bin" -m local_llm_chat --log-output "$log_path" "$@"
