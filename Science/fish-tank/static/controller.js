// Mobile Pac-Man controller. Sends the last-pressed arrow to the server (it
// persists there, classic-arcade style, until another is pressed), pings the
// server ~1/s so the display knows a controller is active, and polls the live
// score to show it here.

const buttons = document.querySelectorAll('.dpad button');
const scoreEl = document.getElementById('score');
const livesEl = document.getElementById('lives');
const statusEl = document.getElementById('status');

let currentDir = null;
let online = false;

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
  statusEl.textContent = ok ? 'Connected — you have control' : 'Reconnecting…';
  statusEl.className = ok ? 'on' : '';
}

function press(dir) {
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

// Heartbeat: keep the controller "active" even when idle so the display keeps
// ignoring the hand and staying in arrow mode.
setInterval(() => sendDir(currentDir), 1000);
sendDir(null);

async function pollScore() {
  try {
    const s = await (await fetch('/api/pmstate')).json();
    scoreEl.textContent = s.score;
    livesEl.textContent = Math.max(0, s.lives);
    setOnline(true);
  } catch (e) {
    setOnline(false);
  }
}
pollScore();
setInterval(pollScore, 600);
