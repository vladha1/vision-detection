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
const WANDER_SWAY_FREQ = 0.6;   // radians/sec - speed of the side-to-side sway
const WANDER_SWAY_AMOUNT = 0.9; // radians - how wide the sway is
const MAX_TURN_RATE = Math.PI * 1.4; // radians/sec - caps how fast the sprite can visually turn

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
    this.heading = a;
    this.wanderPhase = Math.random() * Math.PI * 2;
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

  wanderForce(now) {
    const sway = Math.sin(now * WANDER_SWAY_FREQ + this.wanderPhase) * WANDER_SWAY_AMOUNT;
    const heading = len(this.vel) > 0 ? norm(this.vel) : { x: 1, y: 0 };
    return scale(rotateVec(heading, sway), 60);
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
      steer = add(steer, this.wanderForce(now));
    }

    if (this.pos.x < EDGE_MARGIN) steer.x += (EDGE_MARGIN - this.pos.x) * 4;
    else if (this.pos.x > W - EDGE_MARGIN) steer.x -= (this.pos.x - (W - EDGE_MARGIN)) * 4;
    if (this.pos.y < EDGE_MARGIN) steer.y += (EDGE_MARGIN - this.pos.y) * 4;
    else if (this.pos.y > PLAYABLE_H - EDGE_MARGIN) steer.y -= (this.pos.y - (PLAYABLE_H - EDGE_MARGIN)) * 4;

    this.vel = add(this.vel, scale(steer, dt));
    const speed = len(this.vel);
    if (speed > maxSpeed) this.vel = scale(this.vel, maxSpeed / speed);
    this.pos = add(this.pos, scale(this.vel, dt));

    if (speed > 1) {
      const targetAngle = Math.atan2(this.vel.y, this.vel.x);
      let diff = Math.atan2(Math.sin(targetAngle - this.heading), Math.cos(targetAngle - this.heading));
      const maxStep = MAX_TURN_RATE * dt;
      diff = Math.max(-maxStep, Math.min(maxStep, diff));
      this.heading += diff;
    }
  }

  draw() {
    const angle = this.heading;
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
    if (!res.ok) {
      console.error('hand fetch bad status', res.status);
      hand = null;
      return;
    }
    const raw = await res.json();
    hand = toCanvasSpace(raw);
    if (DEBUG) console.log('hand raw', raw, 'canvas-space', hand, 'config', config);
  } catch (e) {
    console.error('hand sync failed', e);
    hand = null;
  }
}

syncFishList();
setInterval(syncFishList, 2000);
syncHand();
setInterval(syncHand, 100);

const DEBUG = new URLSearchParams(location.search).has('debug');

// --- entrance bloom: a field of flowers opens up when the page loads, then
// fades away into the ordinary tank scene (teamLab "Flower Forest"-style).
const FLOWER_COLORS = ['#ffb3c6', '#ffd166', '#ef476f', '#8bd3ff', '#c77dff', '#9be8b8'];
const INTRO_BLOOM_SECONDS = 1.6;
const INTRO_HOLD_SECONDS = 0.9;
const INTRO_FADE_SECONDS = 1.4;
const INTRO_MAX_DELAY = 1.1;

function makeIntroFlowers(count) {
  const flowers = [];
  for (let i = 0; i < count; i++) {
    flowers.push({
      x: Math.random() * W,
      y: Math.random() * H,
      maxRadius: 26 + Math.random() * 42,
      petals: 5 + Math.floor(Math.random() * 3),
      color: FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)],
      delay: Math.random() * INTRO_MAX_DELAY,
      rotation: Math.random() * Math.PI * 2,
    });
  }
  return flowers;
}

function drawFlower(f, growth, alpha = 1) {
  const r = f.maxRadius * growth;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(f.x, f.y);
  ctx.rotate(f.rotation);
  for (let p = 0; p < f.petals; p++) {
    ctx.save();
    ctx.rotate((p / f.petals) * Math.PI * 2);
    ctx.fillStyle = f.color;
    ctx.beginPath();
    ctx.ellipse(r * 0.55, 0, r * 0.5, r * 0.28, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  ctx.fillStyle = '#fff8e7';
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.28, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

const introFlowers = makeIntroFlowers(24);
const introStart = performance.now() / 1000;
let introActive = true;

function drawIntro(now) {
  const t = now - introStart;
  const fadeStart = INTRO_MAX_DELAY + INTRO_BLOOM_SECONDS + INTRO_HOLD_SECONDS;
  const fadeEnd = fadeStart + INTRO_FADE_SECONDS;
  if (t > fadeEnd) return false;

  const bgAlpha = t > fadeStart ? Math.max(0, 1 - (t - fadeStart) / INTRO_FADE_SECONDS) : 1;

  ctx.save();
  ctx.globalAlpha = bgAlpha;
  ctx.fillStyle = '#03121f';
  ctx.fillRect(0, 0, W, H);
  ctx.restore();

  for (const f of introFlowers) {
    const localT = t - f.delay;
    if (localT <= 0) continue;
    let growth = Math.min(1, localT / INTRO_BLOOM_SECONDS);
    growth = 1 - Math.pow(1 - growth, 3); // ease-out
    drawFlower(f, growth, bgAlpha);
  }
  return true;
}

// --- persistent "flowers" scene: a continuously blooming/fading garden,
// selectable from the admin page instead of the fish tank.
const GARDEN_TARGET_COUNT = 16;
const GARDEN_BLOOM_SECONDS = 1.4;
const GARDEN_SPAWN_CHECK_SECONDS = 0.2;
let gardenFlowers = [];
let lastGardenSpawnCheck = 0;

function spawnGardenFlower(now) {
  gardenFlowers.push({
    x: Math.random() * W,
    y: Math.random() * H,
    maxRadius: 30 + Math.random() * 50,
    petals: 5 + Math.floor(Math.random() * 3),
    color: FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)],
    rotation: Math.random() * Math.PI * 2,
    bornAt: now,
    hold: 2.5 + Math.random() * 3,
    fade: 1.2 + Math.random() * 0.8,
  });
}

function drawFlowerScene(now) {
  ctx.fillStyle = '#03121f';
  ctx.fillRect(0, 0, W, H);

  if (gardenFlowers.length < GARDEN_TARGET_COUNT && now - lastGardenSpawnCheck > GARDEN_SPAWN_CHECK_SECONDS) {
    lastGardenSpawnCheck = now;
    if (Math.random() < 0.6) spawnGardenFlower(now);
  }

  gardenFlowers = gardenFlowers.filter((f) => {
    const age = now - f.bornAt;
    const total = GARDEN_BLOOM_SECONDS + f.hold + f.fade;
    if (age > total) return false;

    let growth, alpha;
    if (age < GARDEN_BLOOM_SECONDS) {
      growth = 1 - Math.pow(1 - age / GARDEN_BLOOM_SECONDS, 3);
      alpha = growth;
    } else if (age < GARDEN_BLOOM_SECONDS + f.hold) {
      growth = 1;
      alpha = 1;
    } else {
      growth = 1;
      alpha = Math.max(0, 1 - (age - GARDEN_BLOOM_SECONDS - f.hold) / f.fade);
    }
    drawFlower(f, growth, alpha);
    return true;
  });
}

let currentScene = 'fish';
async function syncScene() {
  try {
    const res = await fetch('/api/scene');
    currentScene = (await res.json()).scene;
  } catch (e) {
    // keep last known scene
  }
}
syncScene();
setInterval(syncScene, 2000);

let lastTime = performance.now();
function frame(t) {
  const dt = Math.min(0.05, (t - lastTime) / 1000);
  lastTime = t;
  const now = t / 1000;

  if (currentScene === 'flowers') {
    drawFlowerScene(now);
  } else {
    ctx.fillStyle = WATER_COLOR;
    ctx.fillRect(0, 0, W, H);
    for (const f of fishes) {
      f.update(hand, now, dt);
      f.draw();
    }
  }

  if (introActive) {
    introActive = drawIntro(now);
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
