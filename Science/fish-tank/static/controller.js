// Mobile Pac-Man controller. Sends the last-pressed arrow to the server (it
// persists there, classic-arcade style, until another is pressed), pings the
// server ~1/s so the display knows a controller is active, and polls the live
// score to show it here.

const buttons = document.querySelectorAll('.dpad button');
const dpadEl = document.querySelector('.dpad');
const scoreboardEl = document.querySelector('.scoreboard');
const scoreEl = document.getElementById('score');
const levelEl = document.getElementById('level');
const livesEl = document.getElementById('lives');
const statusEl = document.getElementById('status');
const modeToggle = document.getElementById('modeToggle');

let currentDir = null;
let online = false;
let mode = 'controller'; // server default; corrected by the first poll

function renderMode() {
  const active = mode === 'controller';
  modeToggle.textContent = active ? 'Release to hand tracking' : 'Take control';
  modeToggle.classList.toggle('on', active);
  dpadEl.classList.toggle('live', active);
  scoreboardEl.classList.toggle('live', active);
}

async function setMode(next) {
  try {
    mode = (await (await fetch('/api/inputmode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: next }),
    })).json()).mode;
    setOnline(true);
  } catch (e) {
    setOnline(false);
  }
  renderMode();
}

modeToggle.addEventListener('click', () => setMode(mode === 'controller' ? 'hand' : 'controller'));

const resetBtn = document.getElementById('resetBtn');
resetBtn.addEventListener('click', async () => {
  try { await fetch('/api/pmreset', { method: 'POST' }); setOnline(true); }
  catch (e) { setOnline(false); }
});

async function sendDir(dir) {
  try {
    await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dir ? { dir } : {}),
    });
    setOnline(true);
  } catch (e) {
    setOnline(false);
  }
}

function setOnline(ok) {
  if (ok === online) return;
  online = ok;
  if (!ok) { statusEl.textContent = 'Reconnecting…'; statusEl.className = ''; return; }
  statusEl.textContent = mode === 'controller' ? 'You have control' : 'Hand tracking is active';
  statusEl.className = mode === 'controller' ? 'on' : '';
}

function press(dir) {
  if (mode !== 'controller') return; // must take control first
  currentDir = dir;
  for (const b of buttons) b.classList.toggle('held', b.dataset.dir === dir);
  sendDir(dir);
}

for (const btn of buttons) {
  // pointerdown fires immediately on touch, before any scroll/zoom gesture.
  btn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    press(btn.dataset.dir);
  });
}

// Keyboard arrows too, so it's usable from a laptop.
window.addEventListener('keydown', (e) => {
  const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
  if (map[e.key]) { e.preventDefault(); press(map[e.key]); }
});

async function poll() {
  try {
    const [s, c] = await Promise.all([
      (await fetch('/api/pmstate')).json(),
      (await fetch('/api/control')).json(),
    ]);
    scoreEl.textContent = s.score;
    levelEl.textContent = s.level;
    livesEl.textContent = Math.max(0, s.lives);
    if (c.mode !== mode) { mode = c.mode; renderMode(); } // reflect admin-side changes
    setOnline(true);
  } catch (e) {
    setOnline(false);
  }
}
renderMode();
poll();
setInterval(poll, 600);
