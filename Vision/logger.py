#!/usr/bin/env python3
"""
Detection event logger.
- Writes to daily rotating JSONL files (detections_YYYY-MM-DD.jsonl)
- Caps each file at max_mb; rolls to a new numbered file if exceeded
- Keeps an in-memory ring buffer for the dashboard
- Deletes log files older than keep_days
"""
import json
import os
import glob
import time
import threading
from collections import deque

class DetectionLogger:
    def __init__(self, log_dir="logs", max_mb=50, keep_days=7, maxlen=500):
        self.log_dir  = log_dir
        self.max_bytes = max_mb * 1024 * 1024
        self.keep_days = keep_days
        self._buf  = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._subscribers = []
        self._fh   = None
        self._cur_path = None
        os.makedirs(log_dir, exist_ok=True)
        self._open_file()
        self._cleanup_old()

    # ── file management ────────────────────────────────────────────────────

    def _today_base(self):
        return os.path.join(self.log_dir,
                            f"detections_{time.strftime('%Y-%m-%d')}")

    def _open_file(self):
        if self._fh:
            self._fh.close()
        base = self._today_base()
        path = base + ".jsonl"
        # If today's file already exceeds limit, find next numbered slot
        n = 1
        while os.path.exists(path) and os.path.getsize(path) >= self.max_bytes:
            path = f"{base}_{n}.jsonl"
            n += 1
        self._cur_path = path
        self._fh = open(path, "a", buffering=1)   # line-buffered

    def _rotate_if_needed(self):
        # New day or file too big
        if (not self._cur_path.startswith(self._today_base())
                or os.path.getsize(self._cur_path) >= self.max_bytes):
            self._open_file()

    def _cleanup_old(self):
        cutoff = time.time() - self.keep_days * 86400
        for f in glob.glob(os.path.join(self.log_dir, "detections_*.jsonl")):
            if os.path.getmtime(f) < cutoff:
                os.remove(f)

    @property
    def log_file(self):
        return self._cur_path

    # ── public API ─────────────────────────────────────────────────────────

    def log(self, event):
        with self._lock:
            self._rotate_if_needed()
            self._fh.write(json.dumps(event) + "\n")
            self._buf.append(event)
            for q in self._subscribers:
                q.append(event)

    def recent(self, n=100):
        with self._lock:
            return list(self._buf)[-n:]

    def subscribe(self):
        q = deque(maxlen=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass
