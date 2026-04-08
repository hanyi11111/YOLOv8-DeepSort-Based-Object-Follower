#!/usr/bin/env bash
set -e

PYTHON_BIN=""
SCRIPT_PATH=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --script)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "missing --python" >&2
  exit 1
fi

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "missing --script" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_PATH" "${PASS_ARGS[@]}"
