#!/bin/bash
# Double-click to start Focus Composer.
cd "$(dirname "$0")" || exit 1
open "http://127.0.0.1:8000" >/dev/null 2>&1 &
./run.sh
echo
echo "Stopped. You can close this window."
