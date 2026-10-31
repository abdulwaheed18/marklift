#!/usr/bin/env sh
# One-time setup for macOS / Linux: creates .venv and installs the requirements.
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    echo "Python 3.10+ was not found. Install it from https://www.python.org/downloads/"
    exit 1
fi

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt -q
mkdir -p inbox

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "NOTE: ffmpeg was not found. Images work without it, videos need it:"
    echo "    brew install ffmpeg      (macOS)"
    echo "    sudo apt install ffmpeg  (Debian/Ubuntu)"
fi

echo
echo "MarkLift is ready. Try:"
echo "    .venv/bin/python marklift.py clean inbox"
