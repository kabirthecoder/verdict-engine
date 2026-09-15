#!/bin/bash
# Runs real cases on this Mac (needs internet + Ollama or another OpenAI-compatible LLM).
# Double-click from Finder or run: bash scripts/run-cases.command
# Everything is logged to .verdict-sync/run-<timestamp>.log next to the repo.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SYNC="$(dirname "$REPO")/.verdict-sync"
mkdir -p "$SYNC"
LOG="$SYNC/run-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== verdict-engine real run · $(date)"
echo "== repo: $REPO"
cd "$REPO" || exit 1

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== syncing dependencies"
uv sync --all-extras --python 3.12 || exit 1

if [ ! -f .env ]; then
  cp .env.example .env
fi

# Pick a model: prefer what is already pulled in Ollama, else pull a modest tool-capable one.
if command -v ollama >/dev/null 2>&1; then
  echo "== ollama models:"
  ollama list || true
  MODEL=""
  for m in qwen3:32b qwen3:14b qwen2.5:14b llama3.3:70b llama3.1:8b qwen3:8b qwen2.5:7b mistral:7b; do
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m"; then MODEL="$m"; break; fi
  done
  if [ -z "$MODEL" ]; then
    MODEL="qwen3:8b"
    echo "== no tool-capable model found, pulling $MODEL (~5 GB, one time)"
    ollama pull "$MODEL" || exit 1
  fi
  echo "== using ollama model $MODEL"
  export VERDICT_LLM_BASE_URL="http://localhost:11434/v1"
  export VERDICT_LLM_API_KEY="ollama"
  export VERDICT_LLM_MODEL="$MODEL"
else
  echo "== ollama not found; using whatever .env says (VERDICT_LLM_*)"
fi

export VERDICT_DATABASE_URL="sqlite:///$SYNC/verdict.db"

echo; echo "== doctor"; uv run verdict doctor
echo; echo "== case 1"; uv run verdict ask "Is requests 2.28.0 safe to adopt?" -c ecosystem=pypi -c package=requests -c version=2.28.0
echo; echo "== case 2"; uv run verdict ask "Is event-stream 3.3.6 safe to adopt?" -c ecosystem=npm -c package=event-stream -c version=3.3.6
echo; echo "== case 3"; uv run verdict ask "Is requests 2.32.4 safe to adopt?" -c ecosystem=pypi -c package=requests -c version=2.32.4
echo; echo "== list"; uv run verdict list
echo; echo "== done · log: $LOG"
