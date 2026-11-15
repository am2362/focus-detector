#!/bin/bash
# Starts the Focus Composer server and the camera script together.
# Usage:  bash run.sh              (with the Arduino)
#         bash run.sh --no-arduino (extra flags go to the camera script)
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "No .venv here. See README.md."; exit 1; }

# stop any old server still holding port 8000
OLD=$(lsof -ti :8000)
if [ -n "$OLD" ]; then
  echo "Stopping old server on port 8000..."
  kill $OLD 2>/dev/null
  sleep 1
fi

python server.py --audio --rise 10 &
SERVER=$!
trap 'echo; echo "Stopping server..."; kill $SERVER 2>/dev/null' EXIT INT TERM

# wait until the server answers (up to 15 seconds)
for i in $(seq 1 30); do
  curl -s http://127.0.0.1:8000/status >/dev/null && break
  sleep 0.5
done
if ! curl -s http://127.0.0.1:8000/status >/dev/null; then
  echo "Server did not start. Check the messages above."
  exit 1
fi
echo "Dashboard: http://127.0.0.1:8000"

python focus_buddy_composer.py --phone-grace 1 "$@"
