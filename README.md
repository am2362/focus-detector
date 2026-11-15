# Focus Composer

DurHack 2026, theme: Composition.

A desk installation that turns distraction into music. A webcam watches for phone use, looking away and drowsiness. An escalation level from 0 to 4 builds a piece of music, adds spoken lines, and drives an Arduino Nano Every box (LEDs, buzzer, focus timer). When you refocus, the piece resolves.

## Parts

- `focus_buddy_composer.py` camera script (MediaPipe face landmarks + YOLO phone detection). Sends state to the server and the Nano.
- `server.py` FastAPI server and dashboard. Runs the escalation, music, and voice.
- `audio.py` music sections, ElevenLabs voice lines, ducking. `spotify_player.py` optional Spotify playback.
- `focus_buddy_composer/focus_buddy_composer.ino` Nano sketch (LEDs, buzzer, 7-segment timer).
- `focus_buddy.py` and `focus_buddy/focus_buddy.ino` are the original standalone detector and sketch.

## Run

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_composer.txt
cp .env.example .env     # add your ElevenLabs key
```

Put your own music at `music/track.mp3`. Upload the `.ino` sketch to the Nano.

Terminal 1:
```
python server.py --audio --rise 10
```
Terminal 2:
```
python focus_buddy_composer.py --phone-grace 1
```
Open http://127.0.0.1:8000 and press Start session once the camera says it has calibrated. Use `--no-arduino` to run without the board, and the Simulate buttons on the dashboard to test without the camera.

## Notes

- If port 8000 is busy: `lsof -ti :8000 | xargs kill`
- First run downloads the face model and YOLO weights. If the download fails on a Mac with a certificate error, download `face_landmarker.task` by hand into this folder.
