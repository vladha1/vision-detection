const canvas = document.getElementById('tank');
const ctx = canvas.getContext('2d');

const WANDER_SPEED = 90;
const SEEK_SPEED = 200;
const SEEK_RADIUS = 320;
const REACT_HOLD_SECONDS = 1.0;
const SEEK_ARRIVE_RADIUS = 90;
const FLEE_SPEED = 260;
const FLEE_FORCE = 900;
const EDGE_MARGIN = 80;
const FISH_LENGTH = 90;
const WATER_COLOR = '#0a2846';

let W = window.innerWidth;
let H = window.innerHeight;
let PLAYABLE_H = H * 0.8;
let config = null; // calibration space the /api/hand coordinates are reported in

function resize() {
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = W;
  canvas.height = H;
  if (config) PLAYABLE_H = H * (config.playable_height / config.height);
}
window.addEventListener('resize', resize);
resize();

fetch('/api/config').then(r => r.json()).then(cfg => {
  config = cfg;
  resize();
});

// /api/hand reports positions in the calibrated projector space (e.g.
// 1920x1080), which may not match this window's actual pixel size.
function toCanvasSpace(point) {
  if (!point || !config) return point;
  return { x: point.x * (W / config.width), y: point.y * (H / config.height) };
}

const imageCache = {};
function getImage(src) {
  if (!imageCache[src]) {
    const img = new Image();
    img.src = src;
    imageCache[src] = img;
  }
  return imageCache[src];
}

function proceduralSprite(color) {
  const w = FISH_LENGTH, h = Math.round(FISH_LENGTH * 0.6);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.fillStyle = color;
  g.beginPath();
  g.moveTo(w * 0.15, h * 0.5); g.lineTo(w * 0.55, h * 0.1); g.lineTo(w * 0.92, h * 0.3);
  g.lineTo(w * 0.92, h * 0.7); g.lineTo(w * 0.55, h * 0.9); g.closePath(); g.fill();
  g.beginPath();
  g.moveTo(w * 0.15, h * 0.5); g.lineTo(0, h * 0.1); g.lineTo(0, h * 0.9); g.closePath(); g.fill();
  g.fillStyle = '#141414';
  g.beginPath(); g.arc(w * 0.75, h * 0.35, 4, 0, Math.PI * 2); g.fill();
  return c;
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function len(v) { return Math.hypot(v.x, v.y); }
function norm(v) { const l = len(v); return l > 1e-3 ? { x: v.x / l, y: v.y / l } : { x: 1, y: 0 }; }
function scale(v, s) { return { x: v.x * s, y: v.y * s }; }
function add(a, b) { return { x: a.x + b.x, y: a.y + b.y }; }
function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function rotateVec(v, rad) {
  const c = Math.cos(rad), s = Math.sin(rad);
  return { x: v.x * c - v.y * s, y: v.x * s + v.y * c };
}

class Fish {
  constructor(entry) {
    this.id = entry.id;
    this.temperament = entry.temperament;
    this.image = entry.kind === 'image' ? getImage('/sprites/' + entry.filename) : proceduralSprite(entry.color);
    this.pos = {
      x: EDGE_MARGIN + Math.random() * Math.max(1, W - 2 * EDGE_MARGIN),
      y: EDGE_MARGIN + Math.random() * Math.max(1, PLAYABLE_H - 2 * EDGE_MARGIN),
    };
    const a = Math.random() * Math.PI * 2;
    this.vel = { x: Math.cos(a), y: Math.sin(a) };
    this.wanderAngle = 0;
    this.state = 'wander';
    this.reactUntil = 0;
  }

  steerToward(target, maxSpeed, arriveRadius) {
    const toTarget = sub(target, this.pos);
    const d = len(toTarget);
    const desired = d > 1e-3 ? scale(norm(toTarget), maxSpeed * Math.min(1, d / arriveRadius)) : { x: 0, y: 0 };
    return scale(sub(desired, this.vel), 6);
  }

  fleeFrom(threat) {
    const away = sub(this.pos, threat);
    const d = len(away);
    const dir = d > 1e-3 ? scale(away, 1 / d) : rotateVec({ x: 1, y: 0 }, Math.random() * Math.PI * 2);
    return scale(dir, FLEE_FORCE * (1 - Math.min(d, SEEK_RADIUS) / SEEK_RADIUS));
  }

  wanderForce() {
    this.wanderAngle += (Math.random() - 0.5);
    const heading = len(this.vel) > 0 ? norm(this.vel) : { x: 1, y: 0 };
    return scale(rotateVec(heading, this.wanderAngle), 60);
  }

  update(hand, now, dt) {
    let steer = { x: 0, y: 0 };
    let maxSpeed = WANDER_SPEED;

    if (hand && dist(this.pos, hand) < SEEK_RADIUS) {
      this.state = this.temperament;
      this.reactUntil = now + REACT_HOLD_SECONDS;
    }

    if (this.state === 'seek' && now < this.reactUntil && hand) {
      maxSpeed = SEEK_SPEED;
      steer = add(steer, this.steerToward(hand, maxSpeed, SEEK_ARRIVE_RADIUS));
    } else if (this.state === 'flee' && now < this.reactUntil && hand) {
      maxSpeed = FLEE_SPEED;
      steer = add(steer, this.fleeFrom(hand));
    } else {
      this.state = 'wander';
      steer = add(steer, this.wanderForce());
    }

    if (this.pos.x < EDGE_MARGIN) steer.x += (EDGE_MARGIN - this.pos.x) * 4;
    else if (this.pos.x > W - EDGE_MARGIN) steer.x -= (this.pos.x - (W - EDGE_MARGIN)) * 4;
    if (this.pos.y < EDGE_MARGIN) steer.y += (EDGE_MARGIN - this.pos.y) * 4;
    else if (this.pos.y > PLAYABLE_H - EDGE_MARGIN) steer.y -= (this.pos.y - (PLAYABLE_H - EDGE_MARGIN)) * 4;

    this.vel = add(this.vel, scale(steer, dt));
    const speed = len(this.vel);
    if (speed > maxSpeed) this.vel = scale(this.vel, maxSpeed / speed);
    this.pos = add(this.pos, scale(this.vel, dt));
  }

  draw() {
    const angle = Math.atan2(this.vel.y, this.vel.x);
    const img = this.image;
    const iw = img.width || FISH_LENGTH;
    const ih = img.height || FISH_LENGTH * 0.6;
    ctx.save();
    ctx.translate(this.pos.x, this.pos.y);
    ctx.rotate(angle);
    ctx.drawImage(img, -iw / 2, -ih / 2, iw, ih);
    ctx.restore();
  }
}

let fishes = [];
let fishById = {};

async function syncFishList() {
  try {
    const res = await fetch('/api/fish');
    const roster = await res.json();
    const currentIds = new Set(roster.map(e => e.id));
    fishes = fishes.filter(f => currentIds.has(f.id));
    fishById = Object.fromEntries(fishes.map(f => [f.id, f]));
    for (const entry of roster) {
      const existing = fishById[entry.id];
      if (existing) {
        existing.temperament = entry.temperament;
      } else {
        const f = new Fish(entry);
        fishes.push(f);
        fishById[f.id] = f;
      }
    }
  } catch (e) {
    console.error('fish sync failed', e);
  }
}

let hand = null;
async function syncHand() {
  try {
    const res = await fetch('/api/hand');
    hand = toCanvasSpace(await res.json());
  } catch (e) {
    hand = null;
  }
}

syncFishList();
setInterval(syncFishList, 2000);
syncHand();
setInterval(syncHand, 100);

const DEBUG = new URLSearchParams(location.search).has('debug');

let lastTime = performance.now();
function frame(t) {
  const dt = Math.min(0.05, (t - lastTime) / 1000);
  lastTime = t;
  const now = t / 1000;

  ctx.fillStyle = WATER_COLOR;
  ctx.fillRect(0, 0, W, H);

  for (const f of fishes) {
    f.update(hand, now, dt);
    f.draw();
  }

  if (DEBUG) {
    ctx.strokeStyle = '#787878';
    ctx.beginPath();
    ctx.moveTo(0, PLAYABLE_H);
    ctx.lineTo(W, PLAYABLE_H);
    ctx.stroke();
    if (hand) {
      ctx.fillStyle = '#ff00ff';
      ctx.beginPath();
      ctx.arc(hand.x, hand.y, 10, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
