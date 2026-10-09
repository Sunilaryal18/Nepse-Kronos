"""Polite, cached HTTP client for scraping public market websites.

- At least `min_interval` seconds between requests to the same site (thread-safe).
- Descriptive User-Agent.
- Retries with exponential back-off on connection errors, HTTP 429 and 5xx.
- Every raw response is stored gzip-compressed under the cache directory, so a re-run reads
  the file instead of the network. A 404 is remembered with a `<key>.404` marker.
"""
import gzip
import logging
import threading
import time
from pathlib import Path

import requests

USER_AGENT = ("KronosNepseResearch/0.1 (personal, non-commercial research; "
              "cached, max 1 request/second; python-requests)")
log = logging.getLogger(__name__)


class NotFound(Exception):
    """The server answered 404 (now or on an earlier, cached run)."""


class PoliteClient:
    def __init__(self, cache_dir, min_interval=1.0, session=None, max_retries=4, backoff=2.0,
                 timeout=30, sleep=time.sleep, clock=time.monotonic):
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self.session = session if session is not None else requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last = None
        self.network_calls = 0

    # -- cache ---------------------------------------------------------------------------
    def _path(self, key):
        return self.cache_dir / f"{key}.gz"

    def _marker(self, key):
        return self._path(key).with_suffix(".404")

    def cached(self, key):
        """Cached text for `key`, or None if not cached. Raises NotFound for a remembered 404."""
        if key is None:
            return None
        path = self._path(key)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return fh.read()
        if self._marker(key).exists():
            raise NotFound(key)
        return None

    def store(self, key, text):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write(text)
        tmp.replace(path)

    def forget(self, key):
        for path in (self._path(key), self._marker(key)):
            path.unlink(missing_ok=True)

    # -- network -------------------------------------------------------------------------
    def _throttle(self):
        with self._lock:
            if self._last is not None:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()

    def request(self, method, url, key=None, refresh=False, **kwargs):
        """Return the response text, from cache when possible.

        key=None disables caching (e.g. for fetching a CSRF token). Raises NotFound on 404 and
        requests.HTTPError for other 4xx (after retries for 429/5xx).
        """
        if key is not None and refresh:
            self.forget(key)
        text = self.cached(key)
        if text is not None:
            return text
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.network_calls += 1
            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                err = exc
            else:
                if resp.status_code == 404:
                    if key is not None:
                        marker = self._marker(key)
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.touch()
                    raise NotFound(url)
                if resp.status_code == 429 or resp.status_code >= 500:
                    err = requests.HTTPError(f"HTTP {resp.status_code} for {url}", response=resp)
                else:
                    resp.raise_for_status()
                    if key is not None:
                        self.store(key, resp.text)
                    return resp.text
            if attempt == self.max_retries:
                raise err
            delay = self.backoff * 2 ** attempt
            log.warning("%s %s failed (%s); retrying in %.0fs", method, url, err, delay)
            self._sleep(delay)

    def get(self, url, key=None, **kwargs):
        return self.request("GET", url, key=key, **kwargs)

    def post(self, url, key=None, **kwargs):
        return self.request("POST", url, key=key, **kwargs)
