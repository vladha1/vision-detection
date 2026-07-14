"""
fish_alerts.py - person-detection Telegram alerts for the fish-tank camera.

Runs in-process (started from server.py), taps the WallTracker's latest frame,
and sends a Telegram photo when a PERSON appears. Person detection (MediaPipe
EfficientDet) is used instead of raw motion because the camera also sees the
ceiling fan and the wall projection, which would spam a motion trigger.

Alerts are event-based: one photo when someone arrives, then at most a reminder
every `cooldown_sec` while they stay; re-arms for a fresh "new arrival" alert
after the scene is clear for `rearm_clear_sec`.

Diagnostics go to fish_alerts.log next to this file (so they're visible no matter
how the server was launched). Config: .fish_alerts.json.
"""
import os
import json
import time
import logging
import threading

import cv2
import requests
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, ".fish_alerts.json")
MODEL_PATH = os.path.join(HERE, "models", "efficientdet_lite0.tflite")
LOG_PATH = os.path.join(HERE, "fish_alerts.log")

log = logging.getLogger("fish_alerts")
if not log.handlers:
    log.setLevel(logging.INFO)
    _h = logging.FileHandler(LOG_PATH)
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(_h)

DEFAULTS = {
    "enabled": True,
    "bot_token": "",
    "chat_id": "",
    "person_conf": 0.45,
    "check_interval": 3.0,
    "rearm_clear_sec": 90,
    "cooldown_sec": 240,
}


def _load_cfg():
    cfg = dict(DEFAULTS)
    try:
        if os.path.exists(CFG_PATH):
            cfg.update(json.load(open(CFG_PATH)))
    except Exception as e:
        log.info("config load error: %r", e)
    return cfg


def _send_photo(cfg, jpg_bytes, caption):
    try:
        r = requests.post(
            "https://api.telegram.org/bot%s/sendPhoto" % cfg["bot_token"],
            data={"chat_id": cfg["chat_id"], "caption": caption},
            files={"photo": ("fishcam.jpg", jpg_bytes, "image/jpeg")},
            timeout=30,
        )
        if r.status_code == 200 and r.json().get("ok"):
            return True
        log.info("sendPhoto failed: %s %s", r.status_code, r.text[:150])
    except Exception as e:
        log.info("send error: %r", e)
    return False


class _AlertWorker(threading.Thread):
    def __init__(self, get_frame, cfg, detector):
        super().__init__(daemon=True)
        self.get_frame = get_frame
        self.cfg = cfg
        self.detector = detector

    def run(self):
        cfg = self.cfg
        interval = float(cfg["check_interval"])
        rearm = float(cfg["rearm_clear_sec"])
        cooldown = float(cfg["cooldown_sec"])
        last_send = 0.0
        last_active = 0.0
        armed = True
        checks = 0
        while True:
            time.sleep(interval)
            frame = self.get_frame()
            if frame is None:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            except Exception as e:
                log.info("detect error: %r", e)
                continue
            persons = result.detections
            checks += 1
            if checks % 20 == 0:
                log.info("alive: %d checks done, last saw %d person(s)", checks, len(persons))

            now = time.time()
            if not persons:
                if last_active and (now - last_active) >= rearm and not armed:
                    armed = True
                continue

            last_active = now
            new_event = armed
            if not (new_event or (now - last_send) >= cooldown):
                continue

            for d in persons:
                bb = d.bounding_box
                x1, y1 = bb.origin_x, bb.origin_y
                cv2.rectangle(frame, (x1, y1), (x1 + bb.width, y1 + bb.height), (0, 0, 255), 2)
                cv2.putText(frame, "person %.2f" % d.categories[0].score,
                            (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            tag = "NEW" if new_event else "still here"
            caption = "Fish-tank camera - %s (%s)\nPerson x%d" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), tag, len(persons))
            if _send_photo(cfg, buf.tobytes(), caption):
                last_send = now
                armed = False
                log.info("SENT [%s] persons=%d", tag, len(persons))


def start(get_frame):
    """Start the alert worker. Detector is created here (main import thread) to
    avoid MediaPipe GL-context issues when created inside a worker thread."""
    cfg = _load_cfg()
    if not cfg.get("enabled"):
        log.info("disabled in config"); return
    if not os.path.exists(MODEL_PATH):
        log.info("model missing: %s", MODEL_PATH); return
    if not cfg.get("bot_token") or not cfg.get("chat_id"):
        log.info("no bot_token/chat_id yet - not starting"); return
    try:
        detector = mp_vision.ObjectDetector.create_from_options(
            mp_vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                score_threshold=float(cfg["person_conf"]),
                category_allowlist=["person"],
            )
        )
    except Exception as e:
        log.info("detector init failed: %r", e); return
    _AlertWorker(get_frame, cfg, detector).start()
    log.info("started (conf %.2f, every %.0fs)", cfg["person_conf"], cfg["check_interval"])
