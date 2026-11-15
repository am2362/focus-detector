#!/usr/bin/env python3
"""Spotify playback for Focus Composer (Premium needed).

One-time login (opens your browser, saves a refresh token to spotify_token.json):
    python spotify_player.py --login
Quick check that a link plays at each level:
    python spotify_player.py --try "https://open.spotify.com/track/..."

In settings.json:
    "music_source": "spotify",
    "spotify_uri": "https://open.spotify.com/track/..."   (a track, playlist or album link)

Spotify has no per-section looping or fades through its web API, so in this mode the music
is one track whose volume follows the level, and it pauses at level 0.
Spotify's Web API is rate limited, so commands are sent from a background thread, newest
wish only, at most about two calls per second.
"""
import argparse
import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
import webbrowser

import requests

CLIENT_ID = "e83488a4f0cc4f28b339a5f7a1d8fbb5"
REDIRECT = "http://127.0.0.1:8888/callback"
SCOPES = "user-read-private user-read-playback-state user-modify-playback-state"
AUTH = "https://accounts.spotify.com/authorize"
TOKEN = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(HERE, "spotify_token.json")


def to_seconds(v):
    """Accepts 75, "75", "1:15" or "" and returns seconds as a float (0 if empty)."""
    if v in (None, "", 0):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def to_uri(text):
    """Accepts a Spotify link or URI and returns (uri, kind), kind being track/playlist/album/artist."""
    text = (text or "").strip()
    m = re.match(r"spotify:(track|playlist|album|artist):([A-Za-z0-9]+)", text)
    if not m:
        m = re.search(r"open\.spotify\.com/(?:intl-[a-z]+/)?(track|playlist|album|artist)/([A-Za-z0-9]+)", text)
    if not m:
        raise ValueError("Not a Spotify track, playlist or album link.")
    return f"spotify:{m.group(1)}:{m.group(2)}", m.group(1)


class Spotify:
    def __init__(self, client_id=CLIENT_ID, token_file=TOKEN_FILE):
        self.client_id = client_id
        self.token_file = token_file
        self.access = None
        self.expires = 0.0
        self.refresh = None
        self.blocked_until = 0.0
        self.device_id = None
        if os.path.isfile(token_file):
            try:
                self.refresh = json.load(open(token_file)).get("refresh_token")
            except (OSError, ValueError):
                self.refresh = None

    @property
    def logged_in(self):
        return bool(self.refresh)

    # ---- login ----
    def login(self):
        verifier = base64.urlsafe_b64encode(os.urandom(64)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        url = AUTH + "?" + urllib.parse.urlencode({
            "client_id": self.client_id, "response_type": "code", "redirect_uri": REDIRECT,
            "scope": SCOPES, "state": state, "code_challenge_method": "S256", "code_challenge": challenge})
        print("Opening your browser to log in to Spotify...\nIf it doesn't open, paste this:\n" + url + "\n")
        webbrowser.open(url)
        result = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(h):
                q = urllib.parse.parse_qs(urllib.parse.urlparse(h.path).query)
                if q.get("state", [None])[0] == state and "code" in q:
                    result["code"] = q["code"][0]
                    msg = "Logged in. You can close this tab."
                else:
                    result["error"] = q.get("error", ["unknown"])[0]
                    msg = "Login failed: " + result["error"]
                h.send_response(200)
                h.send_header("Content-Type", "text/plain; charset=utf-8")
                h.end_headers()
                h.wfile.write(msg.encode())

            def log_message(h, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 8888), H)
        srv.timeout = 1
        end = time.time() + 180
        while not result and time.time() < end:
            srv.handle_request()
        srv.server_close()
        if "code" not in result:
            raise RuntimeError("Spotify login did not complete: " + result.get("error", "timed out"))
        r = requests.post(TOKEN, data={"grant_type": "authorization_code", "code": result["code"],
                                       "redirect_uri": REDIRECT, "client_id": self.client_id,
                                       "code_verifier": verifier}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Spotify token exchange failed ({r.status_code}): {r.text[:200]}")
        self._store(r.json())

    def _store(self, data):
        self.access = data["access_token"]
        self.expires = time.time() + data.get("expires_in", 3600) - 60
        if data.get("refresh_token"):
            self.refresh = data["refresh_token"]
            with open(self.token_file, "w") as f:
                json.dump({"refresh_token": self.refresh}, f)

    def _token(self):
        if self.access and time.time() < self.expires:
            return self.access
        if not self.refresh:
            raise RuntimeError("Not logged in to Spotify. Run: python spotify_player.py --login")
        r = requests.post(TOKEN, data={"grant_type": "refresh_token", "refresh_token": self.refresh,
                                       "client_id": self.client_id}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Spotify refresh failed ({r.status_code}). Run: python spotify_player.py --login")
        self._store(r.json())
        return self.access

    # ---- calls ----
    def _req(self, method, path, **kw):
        wait = self.blocked_until - time.time()
        if wait > 0:
            time.sleep(min(wait, 10))
        for attempt in (1, 2):
            r = requests.request(method, API + path, headers={"Authorization": f"Bearer {self._token()}"},
                                 timeout=15, **kw)
            if r.status_code == 401 and attempt == 1:
                self.access = None
                continue
            if r.status_code == 429:
                self.blocked_until = time.time() + float(r.headers.get("Retry-After", "5"))
            return r
        return r

    def pick_device(self, force=False):
        if self.device_id and not force:
            return self.device_id
        r = self._req("GET", "/me/player/devices")
        devices = r.json().get("devices", []) if r.status_code == 200 else []
        if not devices:
            self.device_id = None
            raise RuntimeError("No Spotify device found. Open the Spotify app on this computer and play a track once.")
        pick = next((d for d in devices if d["is_active"]), devices[0])
        self.device_id = pick["id"]
        return self.device_id

    def play(self, uri, position_ms=0):
        kind = uri.split(":")[1]
        body = {"uris": [uri]} if kind == "track" else {"context_uri": uri}
        if position_ms:
            body["position_ms"] = int(position_ms)
        for attempt in (1, 2):
            dev = self.pick_device(force=attempt == 2)
            r = self._req("PUT", "/me/player/play", params={"device_id": dev}, json=body)
            if r.status_code != 404:
                break
        if r.status_code >= 400:
            raise RuntimeError(f"Spotify play failed ({r.status_code}): {r.text[:200]}")
        self._req("PUT", "/me/player/repeat", params={"state": "track" if kind == "track" else "context",
                                                      "device_id": dev})

    def seek(self, position_ms):
        if self.device_id:
            self._req("PUT", "/me/player/seek", params={"position_ms": int(position_ms),
                                                        "device_id": self.device_id})

    def progress(self):
        """Current playback position in ms, or None."""
        r = self._req("GET", "/me/player")
        if r.status_code == 200:
            return r.json().get("progress_ms")
        return None

    def pause(self):
        if self.device_id:
            self._req("PUT", "/me/player/pause", params={"device_id": self.device_id})

    def volume(self, pct):
        if self.device_id:
            self._req("PUT", "/me/player/volume", params={"volume_percent": int(max(0, min(100, pct))),
                                                          "device_id": self.device_id})


class SpotifyMusic:
    """Follows the level: pauses at 0, plays the chosen link at a volume that rises with the level.
    Optional start/end (seconds) loop one part of the track. Commands run on a background
    thread so the audio loop never waits on the internet."""

    CALL_GAP = 0.4        # seconds between Spotify calls (rate limit)
    SEEK_LEAD = 0.25      # jump back this early to cover the time a seek takes

    def __init__(self, uri_text, client=None, start=0, end=0):
        self.uri, self.kind = to_uri(uri_text)
        self.sp = client or Spotify()
        if not self.sp.logged_in:
            raise RuntimeError("Not logged in to Spotify. Run: python spotify_player.py --login")
        self.start = max(0.0, to_seconds(start))
        self.end = to_seconds(end)
        if self.end and self.end <= self.start + 1:
            print("Spotify: the end time must be more than a second after the start; ignoring the end.")
            self.end = 0.0
        self.want = (0, 0)             # (level, volume percent)
        self.playing = False
        self.last_volume = None
        self.error = None
        self.ref_ms = 0.0              # playback position at ref_time (for predicting the loop point)
        self.ref_time = 0.0
        self.calibrate_at = None
        self._alive = True
        self._wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def set(self, level, volume_pct):
        if (level, int(volume_pct)) != self.want:
            self.want = (level, int(volume_pct))
            self._wake.set()

    def _mark(self, ms):
        now = time.monotonic()
        self.ref_ms, self.ref_time, self.calibrate_at = ms, now, now + 1.0

    def _looping(self):
        return self.playing and self.end > 0

    def _worker(self):
        while self._alive:
            self._wake.wait(timeout=0.25 if self._looping() else 5)
            self._wake.clear()
            level, vol = self.want
            called = False
            try:
                if level == 0:
                    if self.playing:
                        self.sp.pause()
                        self.playing, called = False, True
                else:
                    if not self.playing:
                        self.sp.play(self.uri, self.start * 1000)
                        self.playing, called = True, True
                        self.last_volume = None
                        self._mark(self.start * 1000)
                    if self.playing and vol != self.last_volume:
                        if called:
                            time.sleep(self.CALL_GAP)
                        self.sp.volume(vol)
                        self.last_volume, called = vol, True
                if self._looping():
                    now = time.monotonic()
                    if self.calibrate_at and now >= self.calibrate_at:
                        if called:
                            time.sleep(self.CALL_GAP)
                        pos = self.sp.progress()
                        called = True
                        self.calibrate_at = None
                        if pos is not None:
                            self.ref_ms, self.ref_time = float(pos), time.monotonic()
                    now = time.monotonic()
                    predicted = self.ref_ms + (now - self.ref_time) * 1000
                    if predicted >= self.end * 1000 - self.SEEK_LEAD * 1000:
                        if called:
                            time.sleep(self.CALL_GAP)
                        self.sp.seek(self.start * 1000)
                        called = True
                        self._mark(self.start * 1000)
                self.error = None
            except Exception as e:                  # keep going; the next change will try again
                self.error = str(e)
                print("Spotify:", e)
                self.playing = False
                time.sleep(2)
            if called:
                time.sleep(self.CALL_GAP)

    def stop(self):
        self._alive = False
        try:
            self.sp.pause()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true", help="log in once and save the token")
    ap.add_argument("--try", dest="try_link", help="play a link through the levels 1 to 4, then pause")
    ap.add_argument("--start", default="0", help="loop start, seconds or m:ss (with --try)")
    ap.add_argument("--end", default="0", help="loop end, seconds or m:ss (with --try); 0 = whole track")
    a = ap.parse_args()
    if a.login:
        sp = Spotify()
        sp.login()
        print("Logged in. Token saved to spotify_token.json")
    elif a.try_link:
        m = SpotifyMusic(a.try_link, start=a.start, end=a.end)
        hold = 4 if not m.end else max(4, (m.end - m.start) * 1.5)
        for level, vol in [(1, 25), (2, 45), (3, 70), (4, 100), (0, 0)]:
            print(f"level {level}: volume {vol}%")
            m.set(level, vol)
            time.sleep(hold)
        m.stop()
        if m.error:
            print("Last error:", m.error)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
