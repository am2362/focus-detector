#!/usr/bin/env python3
"""
Focus Buddy - laptop side
=========================
Watches you through the webcam and tells the Arduino whether you're focused.

  * MediaPipe Face Landmarker : is a face there? which way is the head pointing?
                                are the eyes closed or looking down?
  * YOLO (Ultralytics)        : is there a phone in the picture?

Everything runs on your laptop. No video is saved or sent anywhere.

Set up once:
    python -m venv .venv
    .venv\\Scripts\\activate            (Windows)
    source .venv/bin/activate          (Mac / Linux)
    pip install -r requirements.txt

Run:
    python focus_buddy.py                # finds the Arduino by itself
    python focus_buddy.py --port COM5    # or tell it which port
    python focus_buddy.py --no-arduino   # try it without the hardware

Keys (click the video window first):
    c = recalibrate    p = pause/resume    m = mute/unmute buzzer    q = quit
"""

import argparse
import csv
import math
import platform
import sys
import json
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FACE_MODEL = HERE / "face_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
COCO_PHONE = 67          # "cell phone" in the COCO dataset YOLO was trained on
WINDOW = "Focus Buddy"

# The first letter of every message sent to the Arduino
FOCUSED, WARNING, PHONE, LOOKING, SLEEPY, AWAY, CALIBRATING, PAUSED = "FWPLSACZ"
ALERTS = {PHONE, LOOKING, SLEEPY, AWAY}
LABELS = {
    FOCUSED: "Focused",
    WARNING: "Eyes back on your work?",
    PHONE: "PHONE SPOTTED",
    LOOKING: "Looking away",
    SLEEPY: "Eyes closed",
    AWAY: "Away from desk",
    CALIBRATING: "Calibrating: look at your work as normal",
    PAUSED: "Paused (press p)",
}
COLOURS = {  # BGR
    FOCUSED: (80, 175, 60), WARNING: (0, 170, 255), PHONE: (40, 40, 220),
    LOOKING: (40, 40, 220), SLEEPY: (40, 40, 220), AWAY: (40, 40, 220),
    CALIBRATING: (200, 140, 30), PAUSED: (120, 120, 120),
}


@dataclass
class Settings:
    yaw_limit: float = 30.0        # degrees you can turn left/right from your normal pose
    down_limit: float = 20.0       # degrees you can tilt your head down
    up_limit: float = 25.0         # degrees you can tilt your head up
    eyes_down_extra: float = 0.35  # how much further your eyes can look down (0-1 score)
    eyes_closed: float = 0.55      # eye-blink score that counts as "closed"
    grace: dict = field(default_factory=lambda: {
        PHONE: 2.0,     # seconds before each one sets off the alarm
        LOOKING: 5.0,
        SLEEPY: 3.0,
        AWAY: 10.0,
    })
    warn_after: float = 1.5        # seconds before the red LED starts blinking
    recover_after: float = 1.0     # seconds of focus needed to clear an alarm
    calibrate_secs: float = 3.0


@dataclass
class Observation:
    face: bool = False
    yaw: float = 0.0               # + = head turned one way, - = the other
    pitch: float = 0.0             # - = head tilted down
    eyes_closed: float = 0.0       # 0 = wide open, 1 = shut
    eyes_down: float = 0.0         # 0 = straight ahead, 1 = looking right down
    phone: bool = False
    phone_boxes: list = field(default_factory=list)


@dataclass
class Baseline:
    """Your normal 'working' pose, measured during calibration."""
    yaw: float = 0.0
    pitch: float = 0.0
    eyes_down: float = 0.0
    eyes_closed: float = 0.0


# ---------------------------------------------------------------- seeing
def head_angles(matrix):
    """Yaw and pitch (degrees) from MediaPipe's 4x4 face transformation matrix.

    Column 2 of the rotation is the direction the face is pointing in camera
    space. Pitch is negative when the head tilts down.
    """
    fx, fy, fz = np.asarray(matrix, dtype=float)[:3, 2]
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
    return yaw, pitch


def ensure_face_model():
    if FACE_MODEL.exists():
        return
    print("Downloading the face model (about 3.7 MB)...")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
    except Exception as err:
        FACE_MODEL.unlink(missing_ok=True)
        sys.exit(f"Couldn't download the face model: {err}\n"
                 f"Download it in your browser from\n  {FACE_MODEL_URL}\n"
                 f"and save it next to this script as {FACE_MODEL.name}.\n"
                 "(On a Mac with python.org Python, running 'Install Certificates.command' "
                 "in the Python folder usually fixes download errors.)")


class Detector:
    def __init__(self, yolo_weights="yolo26n.pt", phone_every=3, phone_conf=0.4):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        from ultralytics import YOLO

        self.mp = mp
        ensure_face_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.face = vision.FaceLandmarker.create_from_options(options)
        print(f"Loading {yolo_weights} (downloads automatically the first time)...")
        self.yolo = YOLO(yolo_weights)
        self.phone_every = max(1, phone_every)
        self.phone_conf = phone_conf
        self.frame_no = 0
        self.last_ts = -1
        self.last_phone = -1e9
        self.phone_boxes = []

    def analyse(self, frame_bgr, now):
        obs = Observation()

        # 1. Face: landmarks, head direction and "blendshapes" (facial expression scores)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(int(now * 1000), self.last_ts + 1)   # VIDEO mode needs rising timestamps
        self.last_ts = ts
        result = self.face.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)
        if result.face_landmarks:
            obs.face = True
            if result.facial_transformation_matrixes:
                obs.yaw, obs.pitch = head_angles(result.facial_transformation_matrixes[0])
            if result.face_blendshapes:
                score = {c.category_name: c.score for c in result.face_blendshapes[0]}
                obs.eyes_closed = (score.get("eyeBlinkLeft", 0) + score.get("eyeBlinkRight", 0)) / 2
                obs.eyes_down = (score.get("eyeLookDownLeft", 0) + score.get("eyeLookDownRight", 0)) / 2

        # 2. Phone: YOLO is heavier, so only run it every few frames
        if self.frame_no % self.phone_every == 0:
            found = self.yolo.predict(frame_bgr, classes=[COCO_PHONE], conf=self.phone_conf,
                                      imgsz=480, verbose=False)[0]
            self.phone_boxes = found.boxes.xyxy.cpu().numpy().astype(int).tolist()
            if self.phone_boxes:
                self.last_phone = now
        self.frame_no += 1
        obs.phone = now - self.last_phone < 1.0   # keep it for 1 s so one missed frame doesn't reset it
        obs.phone_boxes = self.phone_boxes
        return obs

    def close(self):
        self.face.close()


# ---------------------------------------------------------------- deciding
class Calibrator:
    def __init__(self, seconds):
        self.seconds = seconds
        self.started = None
        self.samples = []

    @property
    def active(self):
        return self.started is not None

    def start(self, now):
        self.started, self.samples = now, []

    def add(self, obs, now):
        """Collect your normal pose. Returns a Baseline once it has enough."""
        if obs.face and not obs.phone:
            self.samples.append((obs.yaw, obs.pitch, obs.eyes_down, obs.eyes_closed))
        if now - self.started >= self.seconds and len(self.samples) >= 10:
            self.started = None
            return Baseline(*(float(v) for v in np.median(self.samples, axis=0)))
        return None


def what_is_wrong(obs, base, s):
    """Why you look distracted in this frame, or None if you look focused."""
    if obs.phone:
        return PHONE
    if not obs.face:
        return AWAY
    if obs.eyes_closed > max(s.eyes_closed, base.eyes_closed + 0.3):
        return SLEEPY
    turn = obs.yaw - base.yaw
    tilt = obs.pitch - base.pitch          # negative = tilted further down than normal
    if abs(turn) > s.yaw_limit or tilt < -s.down_limit or tilt > s.up_limit:
        return LOOKING
    if obs.eyes_down - base.eyes_down > s.eyes_down_extra:
        return LOOKING
    return None


class FocusTracker:
    """Turns frame-by-frame guesses into a calm, sensible state.

    A quick glance away does nothing. Keep it up past the warning time and
    the red LED blinks; past the grace time the alarm goes off and the focus
    timer rewinds to when you first looked away. You have to look focused for
    a moment before it calms down again.
    """

    def __init__(self, settings):
        self.s = settings
        self.state = FOCUSED
        self.bad_since = None      # when the current "looks distracted" spell began
        self.good_since = None     # when you started looking focused again
        self.spell_focus = 0.0     # focus time counted since the spell began
        self.focus_secs = 0.0
        self.streak_secs = 0.0
        self.best_streak = 0.0
        self.alerts = Counter()

    def reset_spell(self):
        self.best_streak = max(self.best_streak, self.streak_secs)
        self.state, self.bad_since, self.good_since = FOCUSED, None, None
        self.spell_focus = 0.0

    def update(self, reason, now, dt):
        s = self.s
        if reason is None:
            if self.bad_since is None:
                self.state = FOCUSED
            else:
                if self.good_since is None:
                    self.good_since = now
                needed = s.recover_after if self.state in ALERTS else 0.5
                if now - self.good_since >= needed:
                    self.reset_spell()
        else:
            self.good_since = None
            if self.bad_since is None:
                self.bad_since = now
            spell = now - self.bad_since
            if spell >= s.grace[reason]:
                if self.state not in ALERTS:
                    # a new alarm: the lead-up to it wasn't really focus time
                    self.focus_secs -= self.spell_focus
                    self.best_streak = max(self.best_streak, self.streak_secs - self.spell_focus)
                    self.streak_secs = self.spell_focus = 0.0
                if self.state != reason:
                    self.alerts[reason] += 1
                self.state = reason
            elif self.state not in ALERTS and spell >= s.warn_after:
                self.state = WARNING

        if self.state in (FOCUSED, WARNING):
            self.focus_secs += dt
            self.streak_secs += dt
            if self.bad_since is not None:
                self.spell_focus += dt
            else:
                self.best_streak = max(self.best_streak, self.streak_secs)
        return self.state


# ---------------------------------------------------------------- Composer server
# Our server (server.py) plays the music, speaks the lines and runs the dashboard.
# This sends it your current state about 5 times a second, and reads back the
# escalation level (0-4) so the Arduino can follow it.
SERVER_STATE = {FOCUSED: "focused", WARNING: "focused", CALIBRATING: "focused",
                PAUSED: "focused", PHONE: "on phone", LOOKING: "looking away",
                SLEEPY: "drowsy", AWAY: "away"}


class ComposerLink:
    def __init__(self, url, enabled=True):
        self.url = url.rstrip("/")
        self.enabled = enabled
        self.level = 0
        self.ok = False
        self._state = "focused"
        self._stop = False
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def set_state(self, letter):
        self._state = SERVER_STATE.get(letter, "focused")

    def _call(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2.0) as r:
            return json.loads(r.read().decode())

    def _run(self):
        while not self._stop:
            try:
                self._call("/state", {"state": self._state})
                self.level = int(self._call("/status").get("level", 0))
                self.ok = True
            except Exception:
                self.ok, self.level = False, 0
            time.sleep(0.2)

    def close(self):
        self._stop = True


# ---------------------------------------------------------------- Arduino
class ArduinoLink:
    def __init__(self, port=None, enabled=True):
        self.ser = None
        self.port = None
        self.muted = False
        self.last_line = None
        self.last_sent = 0.0
        if not enabled:
            print("Running without the Arduino.")
            return
        import serial
        port = port or self.find_port()
        if not port:
            print("Couldn't find the Arduino. Plug it in, or run with --port COM5 "
                  "(Windows) / --port /dev/cu.usbmodemXXXX (Mac). Carrying on without it.")
            return
        try:
            self.ser = serial.Serial(port, 115200, timeout=0, write_timeout=0.5)
            time.sleep(2)                      # give the board a moment after opening
            self.ser.reset_input_buffer()
            self.port = port
            print(f"Connected to the Arduino on {port}")
        except Exception as err:
            print(f"Couldn't open {port}: {err}\n(Close the Arduino IDE's Serial Monitor - "
                  "only one program can use the port at a time.) Carrying on without it.")
            self.ser = None

    @staticmethod
    def find_port():
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        for p in ports:                         # 0x2341 = Arduino's USB vendor ID
            text = f"{p.description} {p.manufacturer} {p.product}".lower()
            if p.vid == 0x2341 or "arduino" in text or "nano every" in text:
                return p.device
        likely = [p.device for p in ports
                  if any(k in p.device.lower() for k in ("usbmodem", "ttyacm", "usbserial"))]
        return likely[0] if len(likely) == 1 else None

    def send(self, state, focus_secs, force=False):
        line = f"{state}{int(focus_secs)}" if state in (FOCUSED, WARNING) else state
        now = time.monotonic()
        if force or line != self.last_line or now - self.last_sent >= 1.0:
            self._write(line)                   # resend every second as a heartbeat
            self.last_line, self.last_sent = line, now

    def send_level(self, level):
        """E0..E4: escalation level from the Composer server (resent every second)."""
        now = time.monotonic()
        if level != getattr(self, "last_level", None) or now - getattr(self, "last_level_sent", 0.0) >= 1.0:
            self._write(f"E{int(level)}")
            self.last_level, self.last_level_sent = level, now

    def set_mute(self, muted):
        self.muted = muted
        self._write("M0" if muted else "M1")

    def _write(self, text):
        if self.ser is None:
            return
        try:
            self.ser.write((text + "\n").encode())
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)   # ignore what the Arduino prints
        except Exception as err:
            print(f"Lost the Arduino ({err}). Carrying on without it.")
            self.ser = None

    def close(self):
        if self.ser is not None:
            self.ser.close()


# ---------------------------------------------------------------- display
def fmt(secs):
    secs = int(secs)
    return f"{secs // 3600}:{secs // 60 % 60:02d}:{secs % 60:02d}" if secs >= 3600 else f"{secs // 60}:{secs % 60:02d}"


def draw(frame, obs, state, tracker, base, link, settings, composer):
    """Mirror the picture (feels natural) and draw the status on it."""
    view = cv2.flip(frame, 1)
    h, w = view.shape[:2]
    for x1, y1, x2, y2 in obs.phone_boxes:
        cv2.rectangle(view, (w - x2, y1), (w - x1, y2), (40, 40, 220), 3)
        cv2.putText(view, "phone", (w - x2, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 220), 2)

    cv2.rectangle(view, (0, 0), (w, 54), COLOURS[state], -1)
    cv2.putText(view, LABELS[state], (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    lines = [f"Focused {fmt(tracker.focus_secs)}   streak {fmt(tracker.streak_secs)}"]
    if base is not None and obs.face:
        lines.append(f"turn {obs.yaw - base.yaw:+.0f}/{settings.yaw_limit:.0f}  "
                     f"tilt {obs.pitch - base.pitch:+.0f}/-{settings.down_limit:.0f}  "
                     f"eyes down {obs.eyes_down - base.eyes_down:+.2f}  closed {obs.eyes_closed:.2f}")
    for i, text in enumerate(lines):
        cv2.putText(view, text, (12, 82 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(view, text, (12, 82 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    board = f"Arduino {link.port}" if link.ser else "no Arduino"
    board += "  |  server L%d" % composer.level if composer.ok else "  |  no server"
    footer = f"c recalibrate  p pause  m {'unmute' if link.muted else 'mute'}  q quit   |  {board}"
    cv2.rectangle(view, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(view, footer, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return view


# ---------------------------------------------------------------- session log
def save_session(started, tracker):
    total = (datetime.now() - started).total_seconds()
    if total < 30:
        return
    focus_pct = 100 * tracker.focus_secs / total
    a = tracker.alerts
    print(f"\nSession: {fmt(total)}  |  focused {fmt(tracker.focus_secs)} ({focus_pct:.0f}%)"
          f"  |  best streak {fmt(tracker.best_streak)}")
    print(f"Alerts: phone {a[PHONE]}, looking away {a[LOOKING]}, eyes closed {a[SLEEPY]}, away {a[AWAY]}")
    log = HERE / "focus_log.csv"
    new_file = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["date", "start", "minutes", "focused_minutes", "focus_percent",
                             "best_streak_minutes", "phone", "looking_away", "eyes_closed", "away"])
        writer.writerow([started.strftime("%Y-%m-%d"), started.strftime("%H:%M"), round(total / 60, 1),
                         round(tracker.focus_secs / 60, 1), round(focus_pct), round(tracker.best_streak / 60, 1),
                         a[PHONE], a[LOOKING], a[SLEEPY], a[AWAY]])
    print(f"Saved to {log.name}")


# ---------------------------------------------------------------- main
def open_camera(source):
    if str(source).isdigit():
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cam = cv2.VideoCapture(int(source), backend)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cam = cv2.VideoCapture(str(source))    # a video file, handy for testing
    if not cam.isOpened():
        sys.exit(f"Couldn't open camera {source}. Try --camera 1. On a Mac, allow camera "
                 "access for your terminal in System Settings > Privacy & Security > Camera.")
    return cam


def parse_args():
    p = argparse.ArgumentParser(description="Webcam focus tracker that talks to an Arduino.")
    p.add_argument("--port", help="Arduino serial port, e.g. COM5 or /dev/cu.usbmodem1101")
    p.add_argument("--no-arduino", action="store_true", help="run without the hardware")
    p.add_argument("--camera", default="0", help="webcam number (0, 1, ...) or a video file")
    p.add_argument("--mute", action="store_true", help="start with the buzzer off")
    p.add_argument("--no-window", action="store_true", help="don't show the video window")
    p.add_argument("--yaw", type=float, default=30, help="degrees left/right allowed (default 30)")
    p.add_argument("--down", type=float, default=20, help="degrees head-down allowed (default 20)")
    p.add_argument("--look-grace", type=float, default=5, help="seconds looking away before the alarm")
    p.add_argument("--phone-grace", type=float, default=2, help="seconds of phone before the alarm")
    p.add_argument("--away-grace", type=float, default=10, help="seconds away from the desk before the alarm")
    p.add_argument("--phone-conf", type=float, default=0.4, help="how sure YOLO must be it's a phone (0-1)")
    p.add_argument("--phone-every", type=int, default=3, help="check for phones every N frames")
    p.add_argument("--yolo", default="yolo26n.pt", help="YOLO weights (yolo11n.pt also works)")
    p.add_argument("--server", default="http://127.0.0.1:8000", help="Composer server address")
    p.add_argument("--no-server", action="store_true", help="don't talk to the Composer server")
    p.add_argument("--max-seconds", type=float, default=0, help=argparse.SUPPRESS)  # for testing
    return p.parse_args()


def main():
    args = parse_args()
    s = Settings(yaw_limit=args.yaw, down_limit=args.down)
    s.grace.update({LOOKING: args.look_grace, PHONE: args.phone_grace, AWAY: args.away_grace})

    link = ArduinoLink(args.port, enabled=not args.no_arduino)
    composer = ComposerLink(args.server, enabled=not args.no_server)
    link.set_mute(args.mute)
    cam = open_camera(args.camera)
    from_file = not str(args.camera).isdigit()
    detector = Detector(args.yolo, args.phone_every, args.phone_conf)
    tracker = FocusTracker(s)
    calibrator = Calibrator(s.calibrate_secs)
    baseline = None
    paused = False
    started = datetime.now()
    last = None
    show = not args.no_window
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Look at your work as normal for a few seconds while it calibrates...")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("No more frames from the camera.")
                break
            # real time for a webcam; the video's own clock for a file
            now = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000 if from_file else time.monotonic()
            if last is None:
                calibrator.start(now)
                last = now
            dt = min(max(now - last, 0.0), 0.5)
            last = now

            obs = detector.analyse(frame, now)
            if paused:
                state = PAUSED
            elif calibrator.active:
                state = CALIBRATING
                result = calibrator.add(obs, now)
                if result:
                    baseline = result
                    tracker.reset_spell()
                    print(f"Calibrated (turn {baseline.yaw:+.0f}, tilt {baseline.pitch:+.0f}). Go!")
            else:
                state = tracker.update(what_is_wrong(obs, baseline, s), now, dt)
            link.send(state, tracker.focus_secs)
            composer.set_state(state)
            if composer.ok:
                link.send_level(composer.level)

            if show:
                cv2.imshow(WINDOW, draw(frame, obs, state, tracker, baseline, link, s, composer))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    calibrator.start(now)
                elif key == ord("p"):
                    paused = not paused
                    tracker.reset_spell()
                elif key == ord("m"):
                    link.set_mute(not link.muted)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_seconds and now >= args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        composer.set_state(PAUSED)
        composer.close()
        link.send(PAUSED, tracker.focus_secs, force=True)
        link.close()
        detector.close()
        cam.release()
        cv2.destroyAllWindows()
        save_session(started, tracker)


if __name__ == "__main__":
    main()
