"""HTTP client for ufcstats.com.

ufcstats.com fronts every request with a small JavaScript proof-of-work
challenge (find n such that sha256(f"{nonce}:{n}") starts with a given
number of hex zeros, then POST it to /__c to receive a session cookie).
This client solves that challenge transparently, then serves pages from
an on-disk cache so repeated runs don't re-hit the network.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_cache"

_CHALLENGE_MARKER = "Checking your browser"
_NONCE_RE = re.compile(r'nonce\s*=\s*"([a-f0-9]+)"')
_DIFFICULTY_RE = re.compile(r"target\s*=\s*new Array\((\d+)\+1\)")


def _solve_pow(nonce: str, difficulty: int) -> int:
    target = "0" * difficulty
    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1
    return n


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / f"{digest}.html"


class UfcStatsClient:
    """Thread-safe-ish client: one shared session, a lock around the
    (rare) challenge-solving path, otherwise free concurrent GETs."""

    def __init__(self, min_delay: float = 0.05):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._lock = threading.Lock()
        self._min_delay = min_delay
        self._last_request = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _throttle(self):
        with self._lock:
            wait = self._min_delay - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()

    def _get_raw(self, url: str, timeout: int = 20) -> requests.Response:
        self._throttle()
        return self.session.get(url, timeout=timeout)

    def get(self, url: str, use_cache: bool = True, retries: int = 6) -> str:
        cache_file = _cache_path(url)
        if use_cache and cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        resp = None
        for attempt in range(retries):
            resp = self._get_raw(url)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(2.0 * (attempt + 1), 20.0)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                time.sleep(min(1.0 * (attempt + 1), 10.0))
                continue

            if _CHALLENGE_MARKER not in resp.text:
                break
            nonce_m = _NONCE_RE.search(resp.text)
            diff_m = _DIFFICULTY_RE.search(resp.text)
            if not (nonce_m and diff_m):
                break
            n = _solve_pow(nonce_m.group(1), int(diff_m.group(1)))
            base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            with self._lock:
                self.session.post(
                    base + "/__c",
                    data={"nonce": nonce_m.group(1), "n": n},
                    timeout=15,
                )

        if resp is None:
            raise RuntimeError(f"Failed to fetch {url}")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch {url}: HTTP {resp.status_code} after {retries} attempts")
        if _CHALLENGE_MARKER in resp.text:
            raise RuntimeError(f"Could not pass anti-bot challenge for {url}")

        text = resp.text
        if use_cache:
            cache_file.write_text(text, encoding="utf-8")
        return text
