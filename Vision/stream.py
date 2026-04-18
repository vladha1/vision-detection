#!/usr/bin/env python3
"""
Optional MJPEG web stream server.
View detection feed in a browser at http://<mac-mini-ip>:5000 when headless.
"""
import threading
import cv2
from flask import Flask, Response, render_template_string

_app = Flask(__name__)
_frame_holder = [None]

HTML = """
<!DOCTYPE html><html><head>
<title>Vision Detection Stream</title>
<style>
  body { background:#111; display:flex; flex-direction:column;
         align-items:center; justify-content:center; min-height:100vh; margin:0; }
  img  { max-width:100%; border:2px solid #333; border-radius:4px; }
  h1   { color:#0f0; font-family:monospace; margin-bottom:12px; }
</style></head><body>
<h1>Vision Detection</h1>
<img src="/feed" />
</body></html>
"""

def _generate():
    while True:
        frame = _frame_holder[0]
        if frame is None:
            import time; time.sleep(0.05); continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

@_app.route("/")
def index():
    return render_template_string(HTML)

@_app.route("/feed")
def feed():
    return Response(_generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def start_stream_server(frame_holder, port=5000):
    global _frame_holder
    _frame_holder = frame_holder
    t = threading.Thread(target=lambda: _app.run(host="0.0.0.0", port=port,
                                                  threaded=True, use_reloader=False),
                         daemon=True)
    t.start()
