#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
venv_dir="$repo_dir/.venv"
python_bin=${PYTHON:-python3}

if [ ! -x "$venv_dir/bin/python" ]; then
  "$python_bin" -m venv "$venv_dir"
fi

"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install prompt_toolkit pypdf

if ! command -v pdftotext >/dev/null 2>&1 || ! command -v pdftoppm >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install poppler
  else
    printf '%s\n' 'pdftotext or pdftoppm was not found. Install poppler for PDF text and image extraction.' >&2
  fi
fi

printf 'Installed optional dependencies in %s\n' "$venv_dir"
printf 'Run with: %s\n' "$repo_dir/scripts/run.sh"
