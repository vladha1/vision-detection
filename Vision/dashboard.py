#!/usr/bin/env python3
"""Live detection dashboard — SSE-powered, no video feed."""
import json
import threading
import time
from flask import Flask, Response, render_template_string

_app = Flask(__name__)
_logger = None

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vision Detection Log</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: monospace; padding: 16px; }
  h1 { color: #00ff88; margin-bottom: 12px; font-size: 1.2rem; }

  #stats { display: flex; gap: 24px; margin-bottom: 16px; }
  .stat { background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
          padding: 10px 18px; text-align: center; }
  .stat .val { font-size: 2rem; font-weight: bold; color: #00ccff; }
  .stat .lbl { font-size: 0.7rem; color: #888; margin-top: 2px; }

  #log { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  #log th { background: #1a1a1a; color: #888; text-align: left;
             padding: 6px 10px; border-bottom: 1px solid #333; position: sticky; top: 0; }
  #log td { padding: 5px 10px; border-bottom: 1px solid #1e1e1e; vertical-align: top; }
  tr:hover td { background: #151515; }

  .tag { display: inline-block; border-radius: 3px; padding: 1px 6px;
         font-size: 0.75rem; margin: 1px; }
  .tag-obj  { background: #2a2200; color: #ffcc00; border: 1px solid #554400; }
  .tag-person { background: #002233; color: #00ccff; border: 1px solid #005566; }
  .tag-gesture { background: #002200; color: #00ff88; border: 1px solid #005500; }

  #status { position: fixed; top: 12px; right: 16px; font-size: 0.75rem;
             color: #555; }
  #status.live { color: #00ff88; }
  .wrap { max-height: calc(100vh - 130px); overflow-y: auto; }
</style>
</head>
<body>
<h1>Vision Detection Log</h1>
<div id="stats">
  <div class="stat"><div class="val" id="s-events">0</div><div class="lbl">EVENTS</div></div>
  <div class="stat"><div class="val" id="s-persons">0</div><div class="lbl">PERSONS SEEN</div></div>
  <div class="stat"><div class="val" id="s-objects">0</div><div class="lbl">OBJECTS SEEN</div></div>
  <div class="stat"><div class="val" id="s-gestures">0</div><div class="lbl">GESTURES</div></div>
</div>
<div class="wrap">
<table id="log">
  <thead><tr>
    <th style="width:160px">Time</th>
    <th style="width:60px">Persons</th>
    <th>Objects</th>
    <th>Gestures</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>
</div>
<div id="status">connecting…</div>

<script>
let totEvents = 0, totPersons = 0, totObjects = 0, totGestures = 0;
const MAX_ROWS = 200;

function addRow(e) {
  totEvents++;
  totPersons  += e.persons;
  totObjects  += e.objects.length;
  totGestures += e.gestures.length;

  document.getElementById('s-events').textContent   = totEvents;
  document.getElementById('s-persons').textContent  = totPersons;
  document.getElementById('s-objects').textContent  = totObjects;
  document.getElementById('s-gestures').textContent = totGestures;

  const tbody = document.getElementById('tbody');
  const tr = document.createElement('tr');

  const objTags = e.objects.map(o =>
    `<span class="tag tag-obj">${o.label} ${Math.round(o.conf*100)}%</span>`).join('');
  const gestureTags = e.gestures.map(g =>
    `<span class="tag tag-gesture">${g.hand}: ${g.gesture}</span>`).join('');
  const personTag = e.persons > 0
    ? `<span class="tag tag-person">${e.persons} person${e.persons>1?'s':''}</span>` : '—';

  tr.innerHTML = `
    <td>${e.timestamp}</td>
    <td>${personTag}</td>
    <td>${objTags || '—'}</td>
    <td>${gestureTags || '—'}</td>`;

  tbody.insertBefore(tr, tbody.firstChild);
  if (tbody.children.length > MAX_ROWS)
    tbody.removeChild(tbody.lastChild);
}

// Load recent history then connect SSE
fetch('/api/history').then(r => r.json()).then(events => {
  events.slice().reverse().forEach(addRow);
});

const es = new EventSource('/api/stream');
es.onopen = () => {
  document.getElementById('status').textContent = '● live';
  document.getElementById('status').className = 'live';
};
es.onmessage = e => { addRow(JSON.parse(e.data)); };
es.onerror   = () => {
  document.getElementById('status').textContent = '○ reconnecting…';
  document.getElementById('status').className = '';
};
</script>
</body>
</html>"""

@_app.route("/")
def index():
    return render_template_string(HTML)

@_app.route("/api/history")
def history():
    return Response(json.dumps(_logger.recent(200)),
                    mimetype="application/json")

@_app.route("/api/stream")
def stream():
    q = _logger.subscribe()
    def generate():
        try:
            # Send a heartbeat every 15s to keep connection alive
            last_ping = time.time()
            while True:
                if q:
                    event = q.popleft()
                    yield f"data: {json.dumps(event)}\n\n"
                else:
                    if time.time() - last_ping > 15:
                        yield ": ping\n\n"
                        last_ping = time.time()
                    time.sleep(0.1)
        finally:
            _logger.unsubscribe(q)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@_app.route("/api/download")
def download():
    """Download the full JSONL log file."""
    try:
        with open(_logger.log_file) as f:
            data = f.read()
    except FileNotFoundError:
        data = ""
    return Response(data, mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=detections.jsonl"})

def start_dashboard(logger, port=5000):
    global _logger
    _logger = logger
    t = threading.Thread(
        target=lambda: _app.run(host="0.0.0.0", port=port,
                                threaded=True, use_reloader=False),
        daemon=True)
    t.start()
