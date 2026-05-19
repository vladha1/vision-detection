#!/usr/bin/env python3
"""
Shot event logger.
- Daily rotating JSONL files (shots_YYYY-MM-DD.jsonl)
- In-memory ring buffer for the dashboard
- SSE subscription queue per connected client
"""
import json
import os
import glob
import time
import threading
from collections import deque


class ShotLogger:
    def __init__(self, log_dir="dart_logs", max_mb=10, keep_days=30, maxlen=200):
        self.log_dir    = log_dir
        self.max_bytes  = max_mb * 1024 * 1024
        self.keep_days  = keep_days
        self._buf           = deque(maxlen=maxlen)
        self._lock          = threading.Lock()
        self._subscribers   = []
        self._fh            = None
        self._cur_path      = None
        self.total_shots    = 0
        self.running_score  = 0
        os.makedirs(log_dir, exist_ok=True)
        self._open_file()
        self._cleanup_old()

    # ── file management ────────────────────────────────────────────────────────

    def _today_base(self):
        return os.path.join(self.log_dir, f"shots_{time.strftime('%Y-%m-%d')}")

    def _open_file(self):
        if self._fh:
            self._fh.close()
        base = self._today_base()
        path = base + ".jsonl"
        n = 1
        while os.path.exists(path) and os.path.getsize(path) >= self.max_bytes:
            path = f"{base}_{n}.jsonl"
            n += 1
        self._cur_path = path
        self._fh = open(path, "a", buffering=1)

    def _rotate_if_needed(self):
        if (not self._cur_path.startswith(self._today_base())
                or os.path.getsize(self._cur_path) >= self.max_bytes):
            self._open_file()

    def _cleanup_old(self):
        cutoff = time.time() - self.keep_days * 86400
        for f in glob.glob(os.path.join(self.log_dir, "shots_*.jsonl")):
            if os.path.getmtime(f) < cutoff:
                os.remove(f)

    # ── public API ─────────────────────────────────────────────────────────────

    def log(self, event: dict):
        with self._lock:
            self._rotate_if_needed()
            self._fh.write(json.dumps(event) + "\n")
            self._buf.append(event)
            self.total_shots   += 1
            self.running_score += event.get("score", 0)
            for q in self._subscribers:
                q.append(event)

    def reset(self):
        with self._lock:
            self._buf.clear()
            self.total_shots   = 0
            self.running_score = 0
            for q in self._subscribers:
                q.clear()
                q.append({"_reset": True})

    def recent(self, n: int = 50) -> list:
        with self._lock:
            return list(self._buf)[-n:]

    def subscribe(self) -> deque:
        q = deque(maxlen=100)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: deque):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass
