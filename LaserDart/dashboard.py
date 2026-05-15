#!/usr/bin/env python3
"""
Live dartboard score dashboard.
Served at http://0.0.0.0:<port> — streams shots via Server-Sent Events.
"""
import json
import threading
import time
from flask import Flask, Response, render_template_string

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Laser Dart</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: monospace; padding: 1rem; }
  h1 { font-size: 2rem; color: #f90; margin-bottom: 1rem; }
  #stats { display: flex; gap: 2rem; margin-bottom: 1.5rem; }
  .stat { background: #1e1e1e; border-radius: 8px; padding: 1rem 1.5rem; }
  .stat .val { font-size: 2.5rem; font-weight: bold; color: #0f9; }
  .stat .lbl { font-size: 0.75rem; color: #888; margin-top: 0.25rem; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 0.4rem 0.8rem; color: #888;
       border-bottom: 1px solid #333; font-size: 0.8rem; }
  td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #222; }
  tr.new td { animation: flash 0.8s ease-out; }
  tr.bullseye td:nth-child(3) { color: #f90; font-weight: bold; }
  tr.bull     td:nth-child(3) { color: #fa0; }
  tr.miss     td:nth-child(3) { color: #555; }
  @keyframes flash { from { background: #2a2a00; } to { background: transparent; } }
</style>
</head>
<body>
<h1>Laser Dart</h1>
<div id="stats">
  <div class="stat"><div class="val" id="s-shots">0</div><div class="lbl">SHOTS</div></div>
  <div class="stat"><div class="val" id="s-score">0</div><div class="lbl">TOTAL SCORE</div></div>
  <div class="stat"><div class="val" id="s-last">—</div><div class="lbl">LAST SHOT</div></div>
</div>
<table>
  <thead><tr><th>#</th><th>Time</th><th>Label</th><th>Score</th></tr></thead>
  <tbody id="shots"></tbody>
</table>
<script>
let shotCount = 0;
const tbody = document.getElementById("shots");

const es = new EventSource("/stream");
es.addEventListener("shot", e => {
  const d = JSON.parse(e.data);
  shotCount++;
  document.getElementById("s-shots").textContent = shotCount;
  document.getElementById("s-score").textContent = d.running_score;
  document.getElementById("s-last").textContent  = d.label + " (" + d.score + ")";

  const tr = document.createElement("tr");
  tr.className = "new " + (d.label === "Bullseye" ? "bullseye"
                         : d.label === "Bull"     ? "bull"
                         : d.score  === 0         ? "miss" : "");
  tr.innerHTML = `<td>${shotCount}</td><td>${d.timestamp}</td>
                  <td>${d.label}</td><td>${d.score}</td>`;
  tbody.prepend(tr);
  setTimeout(() => tr.classList.remove("new"), 900);
});
es.addEventListener("history", e => {
  const shots = JSON.parse(e.data);
  shots.forEach(d => {
    shotCount++;
    const tr = document.createElement("tr");
    tr.className = d.label === "Bullseye" ? "bullseye"
                 : d.label === "Bull"     ? "bull"
                 : d.score  === 0         ? "miss" : "";
    tr.innerHTML = `<td>${shotCount}</td><td>${d.timestamp}</td>
                    <td>${d.label}</td><td>${d.score}</td>`;
    tbody.appendChild(tr);
  });
  if (shots.length) {
    const last = shots[shots.length - 1];
    document.getElementById("s-shots").textContent = shotCount;
    document.getElementById("s-last").textContent  = last.label + " (" + last.score + ")";
  }
});
</script>
</body>
</html>"""


def start_dashboard(logger, port: int = 5001):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(_HTML)

    @app.route("/stream")
    def stream():
        def generate():
            history = logger.recent(50)
            if history:
                running = sum(e.get("score", 0) for e in history)
                for e in history:
                    e = dict(e)
                    e["running_score"] = running
                yield f"event: history\ndata: {json.dumps(history)}\n\n"

            q = logger.subscribe()
            try:
                while True:
                    if q:
                        event = q.popleft()
                        event = dict(event)
                        event["running_score"] = logger.running_score
                        yield f"event: shot\ndata: {json.dumps(event)}\n\n"
                    else:
                        yield ": keep-alive\n\n"
                        time.sleep(5)
            finally:
                logger.unsubscribe(q)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
        daemon=True,
    )
    t.start()
