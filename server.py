#!/usr/bin/env python3
"""Focus Composer backend: session, escalation, live events, summary, sound settings.

Run:   python server.py              (dashboard at http://localhost:8000)
       python server.py --audio      (also play music + voice through this computer)

Where the state comes from
    Anything can report the current state by POSTing it:
        POST /state   {"state": "focused" | "on phone" | "looking away" | "away" | "drowsy"}
    The dashboard's "Simulate" buttons do this, and so can the detection script, e.g.
        requests.post("http://localhost:8000/state", json={"state": "on phone"})
    (call it whenever the state changes, or every frame; both work).

Endpoints
    GET  /               the dashboard
    POST /start          begin a session (short calibration, then tracking)
    POST /stop           end the session and keep its summary
    GET  /status         current state, level, score, session info
    GET  /events         live stream (server-sent events) of status updates
    GET  /summary        totals, percentages and timeline (live session or the last one)
    POST /state          report the current state
    GET  /settings       voice lines, voice, volume, music mode
    POST /settings       change them (new lines are generated as voice files)
    GET  /voices         ElevenLabs voices (needs ELEVENLABS_API_KEY)
    POST /preview        speak a line now (needs --audio and an API key)
"""
import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from escalation import Escalator

def _load_env():
    """Read KEY=value lines from a .env file next to this script (no extra package needed)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()


ROOT = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(ROOT, "settings.json")
STATES = ["focused", "on phone", "looking away", "away", "drowsy"]
DISTRACTED = {"on phone", "looking away", "drowsy"}   # "away" (left the desk) does not escalate
CALIBRATE_SECS = 3.0
TICK = 0.05
RISE = 20.0                              # seconds of distraction to reach level 4 (--rise)
MUSIC_EXT = (".mp3", ".wav", ".ogg")
MAX_MUSIC_BYTES = 25 * 1024 * 1024
SETTING_KEYS = {"lines", "voice_id", "model_id", "volume", "duck", "music_mode",
                "music_file", "music_sections", "gap_after", "music_source", "spotify_uri",
                "spotify_start", "spotify_end"}


class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.running = False
        self.t0 = 0.0
        self.t_end = None
        self.state = "focused"          # latest reported state (kept between sessions)
        self.shown_state = "focused"
        self.level = 0
        self.peak = 0
        self.events = []                # {"time","state","level","t"}
        self.segments = []              # {"state","start","end"}
        self.seg_start = 0.0
        self.durations = {s: 0.0 for s in STATES}
        self.pickups = 0
        self.esc = Escalator(rise_time=RISE)

    def elapsed(self):
        if not self.t0:
            return 0.0
        return (self.t_end or time.time()) - self.t0

    def calibrating(self):
        return self.running and self.elapsed() < CALIBRATE_SECS


S = Session()
subscribers = []                        # asyncio queues for /events
audio = None                            # AudioEngine when --audio is on
reported = {"state": "focused"}         # what the source last said (kept across sessions)


def status():
    return {
        "running": S.running, "calibrating": S.calibrating(),
        "state": S.shown_state, "level": S.level, "score": round(S.esc.score, 3),
        "elapsed": round(S.elapsed(), 1), "peak": S.peak, "started": int(S.t0),
        "calibrate_secs": CALIBRATE_SECS, "audio": audio is not None,
    }


def summary():
    total = sum(S.durations.values())
    segs = list(S.segments)
    if S.running and S.shown_state:
        segs.append({"state": S.shown_state, "start": round(S.seg_start, 1), "end": round(S.elapsed(), 1)})
    return {
        "duration": round(S.elapsed(), 1), "peak_level": S.peak, "phone_pickups": S.pickups, "started": int(S.t0),
        "seconds": {k: round(v, 1) for k, v in S.durations.items()},
        "percent": {k: (round(100 * v / total, 1) if total else 0.0) for k, v in S.durations.items()},
        "timeline": segs, "events": S.events[-200:],
    }


def publish():
    msg = json.dumps({"status": status(), "latest": S.events[-1] if S.events else None})
    for q in list(subscribers):
        if q.qsize() < 50:
            q.put_nowait(msg)


def log_event(state, level):
    S.events.append({"time": datetime.now().strftime("%H:%M:%S"), "state": state,
                     "level": level, "t": round(S.elapsed(), 1)})


async def tick_loop():
    last = time.time()
    last_pub = 0.0
    while True:
        await asyncio.sleep(TICK)
        now = time.time()
        dt, last = now - last, now
        if S.running:
            if S.calibrating():
                S.shown_state, distracted = "focused", False
            else:
                S.shown_state = S.state
                distracted = S.state in DISTRACTED
                S.durations[S.shown_state] += dt
            prev_level, prev_state = S.level, getattr(S, "_logged_state", None)
            S.level = S.esc.update(distracted, dt)
            S.peak = max(S.peak, S.level)
            if S.shown_state != prev_state or S.level != prev_level:
                if S.shown_state != prev_state:
                    t = round(S.elapsed(), 1)
                    if prev_state is not None:
                        S.segments.append({"state": prev_state, "start": round(S.seg_start, 1), "end": t})
                    S.seg_start = t
                    if S.shown_state == "on phone":
                        S.pickups += 1
                S._logged_state = S.shown_state
                log_event(S.shown_state, S.level)
                publish()
                last_pub = now
        if audio is not None:
            try:
                if S.running and S.level != audio.level:
                    print(f"[audio] level {audio.level} -> {S.level}", flush=True)
                    audio.set_level(S.level)
                if not S.running and audio.level != 0:
                    audio.set_level(0)
                audio.update(dt)
            except Exception:
                import traceback
                print("[audio] ERROR:", flush=True)
                traceback.print_exc()
                await asyncio.sleep(1.0)
        if now - last_pub >= 1.0:
            publish()
            last_pub = now


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(tick_loop())
    yield
    task.cancel()
    if audio is not None:
        audio.stop()


app = FastAPI(title="Focus Composer", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "dashboard.html"))


@app.post("/start")
def start():
    print("[server] Start pressed", flush=True)
    S.reset()
    S.state = reported["state"]
    S.running, S.t0 = True, time.time()
    S._logged_state = None
    publish()
    return status()


@app.post("/stop")
def stop():
    if S.running:
        t = round(S.elapsed(), 1)
        S.t_end, S.running = time.time(), False
        if getattr(S, "_logged_state", None):
            S.segments.append({"state": S._logged_state, "start": round(S.seg_start, 1), "end": t})
        S.level = 0
        publish()
    return summary()


@app.get("/status")
def get_status():
    return status()


@app.get("/summary")
def get_summary():
    return summary()


@app.post("/state")
def set_state(body: dict):
    state = str(body.get("state", "")).strip().lower()
    if state not in STATES:
        raise HTTPException(400, f"state must be one of {STATES}")
    reported["state"] = state
    S.state = state
    return {"state": state}


@app.get("/events")
async def events():
    q = asyncio.Queue()
    subscribers.append(q)
    q.put_nowait(json.dumps({"status": status(), "latest": None}))

    async def gen():
        try:
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/settings")
def get_settings():
    with open(SETTINGS, encoding="utf-8") as f:
        return json.load(f)


@app.post("/settings")
async def post_settings(body: dict):
    import audio as audio_mod
    with open(SETTINGS, encoding="utf-8") as f:
        cur = json.load(f)
    for k, v in body.items():
        if k in SETTING_KEYS:
            cur[k] = v
    lines = cur.get("lines", {})
    if not isinstance(lines, dict) or any(not isinstance(v, list) for v in lines.values()):
        raise HTTPException(400, "lines must be {level: [text, ...]}")
    audio_mod.save_settings(SETTINGS, cur)
    result = {"saved": True, "generated": None, "error": None}
    try:
        result["generated"] = await asyncio.to_thread(audio_mod.generate_voice, cur, os.path.join(ROOT, "voice"), lambda *_: None)
    except Exception as e:                          # settings stay saved; voice files may be missing
        result["error"] = str(e)
    if audio is not None:
        await asyncio.to_thread(audio.reload)
    return result


def music_info():
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, "music", "track.*")))
    info = {"file": os.path.basename(files[0]) if files else None,
            "size_mb": round(os.path.getsize(files[0]) / 1e6, 1) if files else 0}
    if audio is not None:
        info.update(audio.info())
    return info


@app.get("/music")
def get_music():
    return music_info()


@app.post("/music")
async def upload_music(request: Request, name: str = "track.mp3"):
    """Body is the raw audio file. Replaces music/track.* and rebuilds the level sections."""
    ext = os.path.splitext(name)[1].lower()
    if ext not in MUSIC_EXT:
        raise HTTPException(400, "Use an MP3, WAV or OGG file.")
    data = await request.body()
    if not data:
        raise HTTPException(400, "The file was empty.")
    if len(data) > MAX_MUSIC_BYTES:
        raise HTTPException(413, "File is too large (limit 25 MB). Trim it first.")
    music_dir = os.path.join(ROOT, "music")
    os.makedirs(music_dir, exist_ok=True)
    tmp = os.path.join(music_dir, "_upload" + ext)
    with open(tmp, "wb") as f:
        f.write(data)
    if audio is not None:                              # make sure it really plays before replacing the old track
        try:
            audio.pg.mixer.Sound(tmp)
        except Exception as e:
            os.remove(tmp)
            raise HTTPException(400, f"Could not read that audio file: {e}")
    for old in os.listdir(music_dir):
        if old.startswith("track."):
            os.remove(os.path.join(music_dir, old))
    final = os.path.join(music_dir, "track" + ext)
    os.replace(tmp, final)
    import audio as audio_mod
    cur = json.load(open(SETTINGS, encoding="utf-8"))
    cur["music_file"] = "music/track" + ext
    audio_mod.save_settings(SETTINGS, cur)
    if audio is not None:
        await asyncio.to_thread(audio.reload)
    return music_info()


@app.get("/voices")
async def voices():
    import audio as audio_mod
    try:
        return [{"id": i, "name": n} for i, n in await asyncio.to_thread(audio_mod.list_voices)]
    except Exception as e:
        raise HTTPException(502, str(e))


@app.post("/preview")
async def preview(body: dict):
    if audio is None:
        raise HTTPException(503, "Start the server with --audio to preview voices")
    try:
        await asyncio.to_thread(audio.preview, str(body.get("text", "")))
    except Exception as e:
        raise HTTPException(502, str(e))
    return {"ok": True}


def main():
    global audio, RISE
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true", help="play music and voice on this computer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rise", type=float, default=20.0, help="seconds of distraction to reach level 4")
    a = ap.parse_args()
    RISE = a.rise
    S.esc = Escalator(rise_time=RISE)
    if a.audio:
        from audio import AudioEngine
        audio = AudioEngine("settings.json", root=ROOT)
    import uvicorn
    print(f"Dashboard: http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
