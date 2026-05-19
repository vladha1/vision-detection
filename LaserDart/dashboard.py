#!/usr/bin/env python3
"""
Laser target score dashboard.
Shows a live target board with pins at hit positions + shot history table.
"""
import json
import threading
import time
from flask import Flask, Response, render_template_string

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Laser Target</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #eee; font-family: monospace;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

header { display: flex; align-items: center; gap: 2rem; padding: 0.6rem 1.2rem;
         background: #1a1a1a; border-bottom: 1px solid #333; flex-shrink: 0; }
h1 { color: #f90; font-size: 1.4rem; }
.stat .val { font-size: 1.6rem; font-weight: bold; color: #0f9; line-height: 1; }
.stat .lbl { font-size: 0.65rem; color: #666; margin-top: 2px; }
#reset-btn { margin-left: auto; background: #5a0000; color: #eee; border: 1px solid #800;
             padding: 0.35rem 1.1rem; font-family: monospace; font-size: 0.85rem;
             border-radius: 4px; cursor: pointer; }
#reset-btn:hover { background: #aa0000; border-color: #c00; }

main { display: flex; flex: 1; overflow: hidden; }

#board-wrap { flex: 0 0 auto; display: flex; align-items: center;
              justify-content: center; padding: 1rem; }
canvas { display: block; }

#right { flex: 1; display: flex; flex-direction: column; overflow: hidden;
         border-left: 1px solid #222; }
#right-head { padding: 0.5rem 0.8rem; font-size: 0.7rem; color: #555;
              border-bottom: 1px solid #222; }
#shots-wrap { flex: 1; overflow-y: auto; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 0.3rem 0.6rem; color: #555;
     border-bottom: 1px solid #222; font-size: 0.7rem; position: sticky; top: 0;
     background: #111; }
td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #1a1a1a; font-size: 0.8rem; }
.pts { font-weight: bold; }
tr.new td { animation: flash 1s ease-out; }
tr.bull  .zone { color: #ffd700; }
tr.inner .zone { color: #ff8c00; }
tr.miss  .pts  { color: #444; }
@keyframes flash { from { background: #2a2200; } to { background: transparent; } }
</style>
</head>
<body>
<header>
  <h1>Laser Target</h1>
  <div class="stat"><div class="val" id="s-shots">0</div><div class="lbl">SHOTS</div></div>
  <div class="stat"><div class="val" id="s-score">0</div><div class="lbl">SCORE</div></div>
  <div class="stat"><div class="val" id="s-last" style="font-size:1rem">—</div>
                    <div class="lbl">LAST SHOT</div></div>
  <button id="reset-btn" onclick="doReset()">Reset</button>
</header>
<main>
  <div id="board-wrap"><canvas id="board" width="420" height="420"></canvas></div>
  <div id="right">
    <div id="right-head">SHOT HISTORY</div>
    <div id="shots-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>Time</th><th>Zone</th><th>Pts</th><th>Dist</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
const canvas  = document.getElementById('board');
const ctx     = canvas.getContext('2d');
const CX = 210, CY = 210, MAX_MULT = 10, SCALE = 200 / MAX_MULT;

// Ring definitions sent from server via config event
// Each: {mult, pts, label}  ordered inner→outer
let rings     = [];
let gameRadius = 30;
let shotCount = 0;
let allShots  = [];   // {seq, dx, dy, label, score}

// Ring fill colours, inner→outer (index 0 = bullseye)
const RING_COLORS = [
  '#ffd700',  // Bullseye - gold
  '#ff8c00',  // Inner    - orange
  '#cc0000',  // On Target- red
  '#8b0000',  // Close    - dark red
  '#00357a',  // Good     - navy
  '#1a4a7a',  // Okay     - blue
  '#1a3a4a',  // Near     - teal
  '#2a2a2a',  // Far      - dark grey
  '#1a1a1a',  // Out      - near black
];

function ringColor(idx) {
  return RING_COLORS[idx] || '#1a1a1a';
}

function drawBoard() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Outer background (Miss zone)
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw rings largest→smallest so inner covers outer
  const sorted = [...rings].reverse();  // outer first
  sorted.forEach((ring, i) => {
    const ri = rings.length - 1 - i;   // index in original inner→outer order
    ctx.beginPath();
    ctx.arc(CX, CY, ring.mult * SCALE, 0, 2 * Math.PI);
    ctx.fillStyle = ringColor(ri);
    ctx.fill();
  });

  // Thin ring borders
  rings.forEach(ring => {
    ctx.beginPath();
    ctx.arc(CX, CY, ring.mult * SCALE, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  // Crosshairs
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(CX, CY - MAX_MULT * SCALE);
  ctx.lineTo(CX, CY + MAX_MULT * SCALE);
  ctx.moveTo(CX - MAX_MULT * SCALE, CY);
  ctx.lineTo(CX + MAX_MULT * SCALE, CY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Ring point labels (outermost ring only for each zone)
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  rings.forEach((ring, i) => {
    if (ring.mult * SCALE > 14) {
      ctx.font = '9px monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.fillText(ring.pts, CX + ring.mult * SCALE - 22, CY - 5);
    }
  });

  // Centre dot
  ctx.beginPath();
  ctx.arc(CX, CY, 3, 0, 2 * Math.PI);
  ctx.fillStyle = '#fff';
  ctx.fill();

  // Draw all existing pins
  allShots.forEach(s => drawPin(s, false));
}

function pinPos(dx, dy) {
  return [
    CX + (dx / gameRadius) * SCALE,
    CY + (dy / gameRadius) * SCALE
  ];
}

function drawPin(shot, isNew) {
  if (shot.dx === undefined || shot.dy === undefined) return;
  const [px, py] = pinPos(shot.dx, shot.dy);

  if (isNew) {
    ctx.shadowColor = '#fff8';
    ctx.shadowBlur  = 18;
  }

  // Pin body
  ctx.beginPath();
  ctx.arc(px, py, 7, 0, 2 * Math.PI);
  ctx.fillStyle   = isNew ? '#ffffff' : 'rgba(255,255,255,0.85)';
  ctx.strokeStyle = '#000';
  ctx.lineWidth   = 1.5;
  ctx.fill();
  ctx.stroke();
  ctx.shadowBlur  = 0;

  // Shot number
  ctx.fillStyle    = '#000';
  ctx.font         = 'bold 7px monospace';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(shot.seq, px, py);

  // Score bubble to the right
  const label = String(shot.score);
  ctx.font         = '9px monospace';
  ctx.textAlign    = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle    = isNew ? '#ffd700' : 'rgba(255,220,0,0.7)';
  ctx.fillText(label, px + 9, py - 5);
}

function flashPin(shot) {
  drawPin(shot, true);
  setTimeout(() => {
    drawBoard();
  }, 700);
}

// ── SSE ────────────────────────────────────────────────────────────────────
const es = new EventSource('/stream');

es.addEventListener('config', e => {
  const cfg = JSON.parse(e.data);
  gameRadius = cfg.radius;
  rings      = cfg.rings;   // [{mult, pts, label}, ...]
  drawBoard();
});

es.addEventListener('history', e => {
  const history = JSON.parse(e.data);
  history.forEach(sh => {
    shotCount++;
    const s = mkShot(shotCount, sh);
    allShots.push(s);
    prependRow(sh, shotCount, false);
  });
  if (history.length) {
    const last = history[history.length - 1];
    document.getElementById('s-shots').textContent = shotCount;
    document.getElementById('s-score').textContent = last.running_score || '';
    document.getElementById('s-last').textContent  = last.label + ' (' + last.score + ')';
  }
  drawBoard();
});

es.addEventListener('shot', e => {
  const sh = JSON.parse(e.data);
  shotCount++;
  const s = mkShot(shotCount, sh);
  allShots.push(s);

  document.getElementById('s-shots').textContent = shotCount;
  document.getElementById('s-score').textContent = sh.running_score;
  document.getElementById('s-last').textContent  = sh.label + ' (' + sh.score + ')';

  drawBoard();
  flashPin(s);
  prependRow(sh, shotCount, true);
});

function mkShot(seq, sh) {
  const off = sh.offset_px || [null, null];
  return { seq, dx: off[0], dy: off[1], label: sh.label, score: sh.score };
}

function prependRow(sh, seq, animate) {
  const tbody = document.getElementById('tbody');
  const tr = document.createElement('tr');
  const zoneClass = sh.label === 'Bullseye' ? 'bull'
                  : sh.label === 'Inner'    ? 'inner'
                  : sh.score === 0          ? 'miss' : '';
  if (zoneClass) tr.classList.add(zoneClass);
  if (animate)   tr.classList.add('new');
  tr.innerHTML = `
    <td>${seq}</td>
    <td>${sh.timestamp ? sh.timestamp.slice(11,19) : ''}</td>
    <td class="zone">${sh.label}</td>
    <td class="pts">${sh.score}</td>
    <td>${sh.distance_px != null ? sh.distance_px + 'px' : ''}</td>`;
  tbody.prepend(tr);
  if (animate) setTimeout(() => tr.classList.remove('new'), 1000);
}

es.addEventListener('reset', () => {
  shotCount = 0;
  allShots  = [];
  document.getElementById('s-shots').textContent = '0';
  document.getElementById('s-score').textContent = '0';
  document.getElementById('s-last').textContent  = '—';
  document.getElementById('tbody').innerHTML = '';
  drawBoard();
});

function doReset() {
  if (!confirm('Clear the board and reset scores?')) return;
  fetch('/reset', { method: 'POST' });
}

// Draw empty board while waiting for config
drawBoard();
</script>
</body>
</html>"""


def start_dashboard(logger, port: int = 5001, game_radius: int = 30, rings=None):
    rings_serial = [
        {"mult": m, "pts": p, "label": l}
        for m, p, l in (rings or [])
    ]
    config_payload = json.dumps({"radius": game_radius, "rings": rings_serial})

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(_HTML)

    @app.route("/reset", methods=["POST"])
    def reset():
        logger.reset()
        return "", 204

    @app.route("/stream")
    def stream():
        def generate():
            # Config first so client can draw the board immediately
            yield f"event: config\ndata: {config_payload}\n\n"

            # Shot history
            history = logger.recent(50)
            if history:
                running = 0
                history_out = []
                for e in history:
                    running += e.get("score", 0)
                    out = dict(e)
                    out["running_score"] = running
                    history_out.append(out)
                yield f"event: history\ndata: {json.dumps(history_out)}\n\n"

            # Live shots
            q = logger.subscribe()
            try:
                while True:
                    if q:
                        event = dict(q.popleft())
                        if event.get("_reset"):
                            yield "event: reset\ndata: {}\n\n"
                        else:
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

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
        daemon=True,
    ).start()
