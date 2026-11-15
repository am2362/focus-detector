#!/usr/bin/env python3
"""Audio module: music that follows the level, plus spoken lines (ElevenLabs), with ducking.

Use from the backend:
    from audio import AudioEngine
    engine = AudioEngine("settings.json")
    engine.set_level(2)        # whenever the level changes
    engine.update(dt)          # call ~20-30 times a second
    engine.preview("Hello")    # speak any text now (for the dashboard Preview button)

Command line:
    python audio.py --voices      list ElevenLabs voices (needs ELEVENLABS_API_KEY)
    python audio.py --generate    make voice files from settings.json (only changed lines)
    python audio.py --info        show track, section and line lengths
    python audio.py --test        type 0-4 + Enter to set a level; s = speak; q = quit
"""
import argparse
import glob
import hashlib
import json
import os
import queue
import random
import re
import threading
import time

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


LEVEL_VOLUME = {0: 0.0, 1: 0.25, 2: 0.45, 3: 0.70, 4: 1.0}
SLEW = 0.5                         # max music volume change per second
MIN_GAP = 2.0                      # min silence after any spoken line ends
GAP_AFTER = {2: 8.0, 3: 5.0, 4: 3.0}  # silence after a line ends before the next one, per level
FADE_MS = 500                      # crossfade between music sections
MAX_LINE_CHARS = 100
API = "https://api.elevenlabs.io/v1"
CACHE_NAME = re.compile(r"^L\d_[0-9a-f]{8}\.mp3$")


# ---------- settings ----------
def load_settings(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_settings(path, settings):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ---------- ElevenLabs ----------
def _key():
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set ELEVENLABS_API_KEY first (export ELEVENLABS_API_KEY=...)")
    return key


def _check(resp):
    if resp.status_code >= 400:
        detail = resp.text[:300].replace("\n", " ")
        hint = {
            401: "key rejected or missing a permission",
            402: "payment required: out of credits, a paid-only voice, or the key's credit limit",
            429: "too many requests, wait a moment",
        }.get(resp.status_code, "request failed")
        raise RuntimeError(f"ElevenLabs error {resp.status_code} ({hint}): {detail}")


def list_voices():
    import requests
    r = requests.get(f"{API}/voices", headers={"xi-api-key": _key()}, timeout=30)
    _check(r)
    return [(v["voice_id"], v["name"]) for v in r.json().get("voices", [])]


def tts(text, voice_id, model_id):
    """Returns MP3 bytes for the text."""
    import requests
    r = requests.post(
        f"{API}/text-to-speech/{voice_id}",
        headers={"xi-api-key": _key(), "Content-Type": "application/json"},
        json={"text": text, "model_id": model_id},
        timeout=60,
    )
    _check(r)
    return r.content


def line_path(voice_dir, level, text, voice_id, model_id):
    h = hashlib.sha1(f"{voice_id}|{model_id}|{text}".encode("utf-8")).hexdigest()[:8]
    return os.path.join(voice_dir, f"L{level}_{h}.mp3")


def generate_voice(settings, voice_dir="voice", log=print):
    """Make an MP3 for every line that doesn't have one yet; remove files for lines no longer used."""
    os.makedirs(voice_dir, exist_ok=True)
    wanted = set()
    for level, lines in settings.get("lines", {}).items():
        for text in lines:
            text = text.strip()[:MAX_LINE_CHARS]
            if not text:
                continue
            path = line_path(voice_dir, level, text, settings["voice_id"], settings["model_id"])
            wanted.add(os.path.basename(path))
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                continue
            log(f"Generating L{level}: {text}")
            data = tts(text, settings["voice_id"], settings["model_id"])
            with open(path, "wb") as f:
                f.write(data)
    for name in os.listdir(voice_dir):
        if CACHE_NAME.match(name) and name not in wanted:
            os.remove(os.path.join(voice_dir, name))
    log(f"Voice files ready: {len(wanted)}")
    return len(wanted)


# ---------- playback ----------
class AudioEngine:
    def __init__(self, settings_path="settings.json", root="."):
        import pygame
        self.pg = pygame
        self.root = root
        self.settings_path = settings_path
        self.s = load_settings(os.path.join(root, settings_path))
        pygame.mixer.init(frequency=44100)
        pygame.mixer.set_num_channels(4)
        self.music_chs = [pygame.mixer.Channel(0), pygame.mixer.Channel(2)]
        self.cur_ch = 0
        self.voice_ch = pygame.mixer.Channel(1)
        self.level = 0
        self.vol = 0.0
        self.voice_end = -1e9
        self.last_clip = None
        self.music = None
        self.sections = {}               # level -> Sound (a slice of the track)
        self.section_times = {}          # level -> (start_s, end_s)
        self.track_len = 0.0
        self.cur_section = None
        self.spotify = None
        self.clips = {}
        self.reload()

    # -- loading --
    def reload(self):
        """Re-read settings and files (call after saving settings or generating voice)."""
        pg = self.pg
        self.s = load_settings(os.path.join(self.root, self.settings_path))
        if self.spotify is not None:
            self.spotify.stop()
            self.spotify = None
        if self.s.get("music_source") == "spotify":
            try:
                from spotify_player import SpotifyMusic
                self.spotify = SpotifyMusic(self.s.get("spotify_uri", ""),
                                            start=self.s.get("spotify_start", 0),
                                            end=self.s.get("spotify_end", 0))
                print("Music: Spotify")
            except Exception as e:
                print(f"Spotify not used ({e}). Using the local music file.")
        self.music = None
        path = os.path.join(self.root, self.s.get("music_file", ""))
        if not os.path.isfile(path):
            found = glob.glob(os.path.join(self.root, "music", "track.*"))
            path = found[0] if found else None
        if path:
            try:
                self.music = pg.mixer.Sound(path)
            except Exception as e:
                print(f"Could not load music {path}: {e}")
        else:
            print("No music file found (music/track.mp3). Running without music.")
        self._build_sections()
        self.clips = {}
        voice_dir = os.path.join(self.root, "voice")
        for level, lines in self.s.get("lines", {}).items():
            for text in lines:
                text = text.strip()[:MAX_LINE_CHARS]
                p = line_path(voice_dir, level, text, self.s["voice_id"], self.s["model_id"])
                if os.path.isfile(p) and os.path.getsize(p) > 0:
                    try:
                        self.clips.setdefault(int(level), []).append(pg.mixer.Sound(p))
                    except Exception as e:
                        print(f"Could not load {p}: {e}")
        if not self.clips:
            print("No voice files yet. Run: python audio.py --generate")

    def _build_sections(self):
        """Split the track into one slice per level (1-4), sized from the track's real length."""
        self.sections, self.section_times, self.track_len = {}, {}, 0.0
        for ch in self.music_chs:
            ch.stop()
        self.cur_section = None
        if self.music is None:
            return
        freq, size, chans = self.pg.mixer.get_init()
        frame = (abs(size) // 8) * chans
        raw = self.music.get_raw()
        total = len(raw) // frame
        self.track_len = total / freq
        custom = self.s.get("music_sections") or {}
        for lv in range(1, 5):
            if str(lv) in custom:
                a, b = custom[str(lv)]
            else:
                a, b = self.track_len * (lv - 1) / 4, self.track_len * lv / 4
            a, b = max(0.0, float(a)), min(self.track_len, float(b))
            if b - a < 0.5:
                continue
            chunk = raw[int(a * freq) * frame:int(b * freq) * frame]
            self.sections[lv] = self.pg.mixer.Sound(buffer=chunk)
            self.section_times[lv] = (round(a, 1), round(b, 1))

    def info(self):
        """Lengths used for pacing (for --info and the dashboard's sound settings page)."""
        lines = {}
        for lv, clips in self.clips.items():
            lines[lv] = [round(c.get_length(), 1) for c in clips]
        return {"track_seconds": round(self.track_len, 1),
                "mode": self.s.get("music_mode", "sections"),
                "sections": self.section_times, "line_seconds": lines}

    # -- control --
    def set_level(self, n):
        n = max(0, min(4, int(n)))
        rising = n > self.level
        self.level = n
        if rising and n >= 1:
            self.speak(n)

    def speak(self, level=None, force=False):
        now = time.time()
        if not force and (self.voice_ch.get_busy() or now < self.voice_end + MIN_GAP):
            return False
        level = self.level if level is None else level
        pool = None
        for lv in range(level, 0, -1):          # fall back to a lower level if this one is empty
            if self.clips.get(lv):
                pool = self.clips[lv]
                break
        if not pool:
            return False
        choices = [c for c in pool if c is not self.last_clip] or pool
        clip = random.choice(choices)
        self.last_clip = clip
        clip.set_volume(min(1.0, float(self.s.get("volume", 0.8)) + 0.2))
        self.voice_ch.play(clip)
        self.voice_end = now + clip.get_length()
        return True

    def preview(self, text):
        """Speak any text right now (generates it live; needs the internet)."""
        text = text.strip()[:MAX_LINE_CHARS]
        if not text:
            return
        os.makedirs(os.path.join(self.root, "voice"), exist_ok=True)
        p = os.path.join(self.root, "voice", "_preview.mp3")
        data = tts(text, self.s["voice_id"], self.s["model_id"])
        with open(p, "wb") as f:
            f.write(data)
        clip = self.pg.mixer.Sound(p)
        self.voice_ch.play(clip)
        self.voice_end = time.time() + clip.get_length()

    def update(self, dt):
        master = float(self.s.get("volume", 0.8))
        target = LEVEL_VOLUME[self.level] * master
        if self.voice_ch.get_busy():
            target *= float(self.s.get("duck", 0.3))     # duck the music under the voice
        step = SLEW * dt
        self.vol += max(-step, min(step, target - self.vol))
        if self.spotify is not None:
            self.spotify.set(self.level, round(100 * target))   # target already includes the voice ducking
        elif self.music is not None:
            if self.s.get("music_mode", "sections") == "sections" and self.sections:
                want = min(self.level, max(self.sections)) if self.level >= 1 else None
                if want is not None and want not in self.sections:
                    want = max(k for k in self.sections if k <= want)
                if want is not None and want != self.cur_section:
                    self.music_chs[self.cur_ch].fadeout(FADE_MS)
                    self.cur_ch = 1 - self.cur_ch
                    self.music_chs[self.cur_ch].play(self.sections[want], loops=-1, fade_ms=FADE_MS)
                    self.cur_section = want
            elif self.level >= 1 and not self.music_chs[0].get_busy():
                self.music_chs[0].play(self.music, loops=-1)
            ch = self.music_chs[self.cur_ch]
            if ch.get_busy():
                ch.set_volume(self.vol)
        gaps = {int(k): float(v) for k, v in (self.s.get("gap_after") or {}).items()} or GAP_AFTER
        gap = gaps.get(self.level)
        if gap is not None and not self.voice_ch.get_busy() and time.time() - self.voice_end >= gap:
            self.speak(self.level)

    def stop(self):
        if self.spotify is not None:
            self.spotify.stop()
        self.pg.mixer.stop()
        self.vol = 0.0


# ---------- command line ----------
def _test_mode(engine):
    cmds = queue.Queue()

    def reader():
        while True:
            try:
                cmds.put(input().strip().lower())
            except EOFError:
                cmds.put("q")
                return

    threading.Thread(target=reader, daemon=True).start()
    print("Type 0-4 then Enter to set the level. s = speak now, r = reload files, q = quit.")
    last = time.time()
    while True:
        now = time.time()
        dt, last = now - last, now
        while not cmds.empty():
            c = cmds.get()
            if c in list("01234"):
                engine.set_level(int(c))
                print(f"level {c}")
            elif c == "s":
                engine.speak(force=True)
            elif c == "r":
                engine.reload()
            elif c == "q":
                engine.stop()
                return
        engine.update(dt)
        time.sleep(0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="settings.json")
    ap.add_argument("--voices", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--info", action="store_true", help="show track/section/line lengths")
    a = ap.parse_args()
    if a.voices:
        for vid, name in list_voices():
            print(f"{vid}  {name}")
    elif a.generate:
        generate_voice(load_settings(a.settings))
    elif a.info:
        print(json.dumps(AudioEngine(a.settings).info(), indent=2))
    elif a.test:
        _test_mode(AudioEngine(a.settings))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
