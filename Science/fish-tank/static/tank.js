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
let handWasSeen = false;
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
    if (hand && !handWasSeen) console.log('hand detected', hand);
    else if (!hand && handWasSeen) console.log('hand lost');
    handWasSeen = !!hand;
  } catch (e) {
    console.error('hand sync failed', e);
    hand = null;
    handWasSeen = false;
  }
}

syncFishList();
setInterval(syncFishList, 2000);
syncHand();
setInterval(syncHand, 100);

// --- entrance bloom: a field of flowers opens up when the page loads, then
// fades away into the ordinary tank scene (teamLab "Flower Forest"-style).
const FLOWER_COLORS = ['#ffb3c6', '#ffd166', '#ef476f', '#8bd3ff', '#c77dff', '#9be8b8'];
const FLOWER_KINDS = ['daisy', 'star', 'cluster'];
const INTRO_BLOOM_SECONDS = 1.6;
const INTRO_HOLD_SECONDS = 0.9;
const INTRO_FADE_SECONDS = 1.4;
const INTRO_MAX_DELAY = 1.1;

function randomKind() {
  return FLOWER_KINDS[Math.floor(Math.random() * FLOWER_KINDS.length)];
}

function makeIntroFlowers(count) {
  const flowers = [];
  for (let i = 0; i < count; i++) {
    flowers.push({
      x: Math.random() * W,
      y: Math.random() * H,
      maxRadius: 18 + Math.random() * 58,
      petals: 5 + Math.floor(Math.random() * 3),
      color: FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)],
      kind: randomKind(),
      delay: Math.random() * INTRO_MAX_DELAY,
      rotation: Math.random() * Math.PI * 2,
    });
  }
  return flowers;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function lerpColor(hexA, hexB, t) {
  const a = hexToRgb(hexA), b = hexToRgb(hexB);
  return `rgb(${Math.round(a.r + (b.r - a.r) * t)},${Math.round(a.g + (b.g - a.g) * t)},${Math.round(a.b + (b.b - a.b) * t)})`;
}

function drawFlower(f, growth, alpha = 1, wobble = 0) {
  const r = f.maxRadius * growth;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(f.x, f.y);
  ctx.rotate(f.rotation + wobble);

  if (f.kind === 'star') {
    for (let p = 0; p < f.petals; p++) {
      ctx.save();
      ctx.rotate((p / f.petals) * Math.PI * 2);
      ctx.fillStyle = f.color;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(r * 0.25, r * 0.14);
      ctx.lineTo(r, 0);
      ctx.lineTo(r * 0.25, -r * 0.14);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  } else if (f.kind === 'cluster') {
    for (let p = 0; p < f.petals; p++) {
      const ang = (p / f.petals) * Math.PI * 2;
      ctx.fillStyle = f.color;
      ctx.beginPath();
      ctx.arc(Math.cos(ang) * r * 0.5, Math.sin(ang) * r * 0.5, r * 0.32, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    for (let p = 0; p < f.petals; p++) {
      ctx.save();
      ctx.rotate((p / f.petals) * Math.PI * 2);
      ctx.fillStyle = f.color;
      ctx.beginPath();
      ctx.ellipse(r * 0.55, 0, r * 0.5, r * 0.28, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
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
const GARDEN_TARGET_COUNT = 30;
const GARDEN_MAX_COUNT = 60; // safety cap so a lingering hand can't spawn forever
const GARDEN_BLOOM_SECONDS = 1.4;
const GARDEN_SPAWN_CHECK_SECONDS = 0.15;
const WOBBLE_RADIUS = 380;    // how far a hand's presence reaches into the garden
const WOBBLE_FREQ = 9;        // radians/sec - how fast flowers shiver
const WOBBLE_AMOUNT = 0.16;   // radians - how far they shiver, kept subtle
const REACTION_CHECK_SECONDS = 0.5; // how often we "roll the dice" while a hand is present
const REACTION_NEARBY_RADIUS = WOBBLE_RADIUS;
const REACTION_PATH_RADIUS = 140; // "in the way" distance for the dodge reaction
const REACTION_MOVE_DURATION = 1.0;
const REACTION_COLOR_DURATION = 0.8;
let gardenFlowers = [];
let lastGardenSpawnCheck = 0;
let lastReactionCheck = 0;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function makeGardenFlower(now, x, y) {
  return {
    x, y,
    maxRadius: 16 + Math.random() * 70, // wide size range - tiny buds to big blooms
    petals: 5 + Math.floor(Math.random() * 3),
    color: FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)],
    kind: randomKind(),
    rotation: Math.random() * Math.PI * 2,
    wobbleSeed: Math.random() * Math.PI * 2,
    bornAt: now,
    hold: 2.5 + Math.random() * 3,
    fade: 1.2 + Math.random() * 0.8,
  };
}

function spawnGardenFlower(now, x, y) {
  gardenFlowers.push(makeGardenFlower(now, x !== undefined ? x : Math.random() * W, y !== undefined ? y : Math.random() * H));
}

function startMove(f, targetX, targetY, now, duration = REACTION_MOVE_DURATION) {
  f.moveFrom = { x: f.x, y: f.y };
  f.moveTo = { x: clamp(targetX, EDGE_MARGIN, W - EDGE_MARGIN), y: clamp(targetY, EDGE_MARGIN, H - EDGE_MARGIN) };
  f.moveStart = now;
  f.moveDuration = duration;
}

// While a hand lingers, occasionally: plant a new flower at the fingertip,
// relocate a nearby one, nudge one out of the way, recolor one - or do
// nothing at all, so the garden feels alive rather than mechanically
// responsive.
function triggerHandReaction(now) {
  if (gardenFlowers.length >= GARDEN_MAX_COUNT) return;
  const roll = Math.random();
  if (roll < 0.25) {
    const jitter = 40;
    spawnGardenFlower(now, hand.x + (Math.random() - 0.5) * jitter * 2, hand.y + (Math.random() - 0.5) * jitter * 2);
  } else if (roll < 0.45) {
    const nearby = gardenFlowers.filter(f => f.moveStart === undefined && dist(f, hand) < REACTION_NEARBY_RADIUS);
    if (nearby.length) {
      const f = nearby[Math.floor(Math.random() * nearby.length)];
      startMove(f, Math.random() * W, Math.random() * H, now);
    }
  } else if (roll < 0.65) {
    const inPath = gardenFlowers.filter(f => f.moveStart === undefined && dist(f, hand) < REACTION_PATH_RADIUS);
    if (inPath.length) {
      const f = inPath[0];
      const away = norm(sub(f, hand));
      startMove(f, f.x + away.x * 180, f.y + away.y * 180, now, 0.6);
    }
  } else if (roll < 0.9) {
    const nearby = gardenFlowers.filter(f => f.colorChangeStart === undefined && dist(f, hand) < REACTION_NEARBY_RADIUS);
    if (nearby.length) {
      const f = nearby[Math.floor(Math.random() * nearby.length)];
      let nextColor = f.color;
      while (nextColor === f.color) nextColor = FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)];
      f.colorFrom = f.color;
      f.colorTo = nextColor;
      f.colorChangeStart = now;
    }
  }
  // else: no reaction this round
}

// --- ground filler: grass tufts (static positions, gentle sway) so the
// garden reads as full even between flower blooms.
const GRASS_COLOR = '#2e7d4f';
let grassBlades = [];

function initGrass() {
  grassBlades = [];
  const count = Math.max(20, Math.floor(W / 16));
  for (let i = 0; i < count; i++) {
    grassBlades.push({
      x: (i / count) * W + (Math.random() - 0.5) * 16,
      height: 20 + Math.random() * 34,
      width: 3 + Math.random() * 3,
      lean: (Math.random() - 0.5) * 0.4,
      swaySeed: Math.random() * Math.PI * 2,
      swaySpeed: 0.5 + Math.random() * 0.5,
    });
  }
}

function drawGrass(now) {
  const baseline = H - 6;
  ctx.strokeStyle = GRASS_COLOR;
  ctx.lineCap = 'round';
  for (const b of grassBlades) {
    let gust = 0;
    if (hand) {
      const d = Math.hypot(b.x - hand.x, baseline - b.height * 0.5 - hand.y);
      if (d < WOBBLE_RADIUS) {
        const intensity = 1 - d / WOBBLE_RADIUS;
        gust = Math.sin(now * WOBBLE_FREQ + b.swaySeed) * 0.6 * intensity;
      }
    }
    const sway = Math.sin(now * b.swaySpeed + b.swaySeed) * 0.3 + gust;
    ctx.lineWidth = b.width;
    ctx.beginPath();
    ctx.moveTo(b.x, baseline);
    const midX = b.x + (sway + b.lean) * b.height * 0.4;
    const tipX = b.x + (sway * 1.6 + b.lean) * b.height * 0.7;
    ctx.quadraticCurveTo(midX, baseline - b.height * 0.55, tipX, baseline - b.height);
    ctx.stroke();
  }
}

// --- butterflies: a small ambient population that flutters around
// continuously, independent of the flower lifecycle.
const BUTTERFLY_COLORS = ['#fff176', '#ff8a65', '#ba68c8', '#4fc3f7', '#f06292'];
const BUTTERFLY_COUNT = 6;
let butterflies = [];

function initButterflies() {
  butterflies = [];
  for (let i = 0; i < BUTTERFLY_COUNT; i++) {
    butterflies.push({
      x: Math.random() * W,
      y: Math.random() * H * 0.7,
      angle: Math.random() * Math.PI * 2,
      speed: 26 + Math.random() * 22,
      color: BUTTERFLY_COLORS[Math.floor(Math.random() * BUTTERFLY_COLORS.length)],
      wanderSeed: Math.random() * Math.PI * 2,
      flapSeed: Math.random() * Math.PI * 2,
    });
  }
}

function updateAndDrawButterflies(now, dt) {
  const margin = 24;
  for (const b of butterflies) {
    b.angle += Math.sin(now * 0.8 + b.wanderSeed) * dt * 1.6;
    b.x += Math.cos(b.angle) * b.speed * dt;
    b.y += Math.sin(b.angle) * b.speed * dt + Math.sin(now * 2 + b.wanderSeed) * 8 * dt;

    if (b.x < margin) { b.x = margin; b.angle = Math.PI - b.angle; }
    if (b.x > W - margin) { b.x = W - margin; b.angle = Math.PI - b.angle; }
    if (b.y < margin) { b.y = margin; b.angle = -b.angle; }
    if (b.y > H - margin) { b.y = H - margin; b.angle = -b.angle; }

    const flap = Math.abs(Math.sin(now * 14 + b.flapSeed));
    const wingSpan = 10 + flap * 6;
    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(b.angle);
    ctx.fillStyle = b.color;
    ctx.beginPath();
    ctx.ellipse(-4, 0, wingSpan * 0.5, wingSpan * 0.32, 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(4, 0, wingSpan * 0.5, wingSpan * 0.32, -0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#3a2a1a';
    ctx.beginPath();
    ctx.ellipse(0, 0, 2, 5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

let gardenInitialized = false;

function drawFlowerScene(now, dt) {
  if (!gardenInitialized) {
    gardenInitialized = true;
    initGrass();
    initButterflies();
  }

  ctx.fillStyle = '#03121f';
  ctx.fillRect(0, 0, W, H);
  drawGrass(now);

  if (gardenFlowers.length < GARDEN_TARGET_COUNT && now - lastGardenSpawnCheck > GARDEN_SPAWN_CHECK_SECONDS) {
    lastGardenSpawnCheck = now;
    if (Math.random() < 0.6) spawnGardenFlower(now);
  }

  if (hand && now - lastReactionCheck > REACTION_CHECK_SECONDS) {
    lastReactionCheck = now;
    triggerHandReaction(now);
  }

  gardenFlowers = gardenFlowers.filter((f) => {
    const age = now - f.bornAt;
    const total = GARDEN_BLOOM_SECONDS + f.hold + f.fade;
    if (age > total) return false;

    if (f.moveStart !== undefined) {
      const mt = Math.min(1, (now - f.moveStart) / f.moveDuration);
      const eased = 1 - Math.pow(1 - mt, 3);
      f.x = f.moveFrom.x + (f.moveTo.x - f.moveFrom.x) * eased;
      f.y = f.moveFrom.y + (f.moveTo.y - f.moveFrom.y) * eased;
      if (mt >= 1) f.moveStart = undefined;
    }

    if (f.colorChangeStart !== undefined) {
      const ct = Math.min(1, (now - f.colorChangeStart) / REACTION_COLOR_DURATION);
      f.color = lerpColor(f.colorFrom, f.colorTo, ct);
      if (ct >= 1) f.colorChangeStart = undefined;
    }

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

    let wobble = 0;
    if (hand) {
      const d = dist(f, hand);
      if (d < WOBBLE_RADIUS) {
        const intensity = 1 - d / WOBBLE_RADIUS;
        wobble = Math.sin(now * WOBBLE_FREQ + f.wobbleSeed) * WOBBLE_AMOUNT * intensity;
        // pulse size too - a rotation shiver alone is easy to miss among many flowers
        growth *= 1 + Math.sin(now * WOBBLE_FREQ + f.wobbleSeed) * 0.22 * intensity;
      }
    }
    drawFlower(f, growth, alpha, wobble);
    return true;
  });

  updateAndDrawButterflies(now, dt);
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

// --- always-on hand marker: shows where the tracked hand is, styled to
// match whichever scene is active (a small fish, or a small flower).
const HAND_MARKER_COLOR = '#ffffff';
let handMarkerFishSprite = null;
function getHandMarkerFish() {
  if (!handMarkerFishSprite) handMarkerFishSprite = proceduralSprite(HAND_MARKER_COLOR);
  return handMarkerFishSprite;
}

function drawGlow(x, y, radius, color) {
  const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
  grad.addColorStop(0, color);
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.save();
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

let handMarkerHeading = 0;
let handMarkerLastPos = null;

function drawHandMarker(now) {
  if (!hand) {
    handMarkerLastPos = null;
    return;
  }
  if (handMarkerLastPos) {
    const dx = hand.x - handMarkerLastPos.x;
    const dy = hand.y - handMarkerLastPos.y;
    if (Math.hypot(dx, dy) > 2) {
      handMarkerHeading = Math.atan2(dy, dx);
    }
  }
  handMarkerLastPos = { x: hand.x, y: hand.y };

  const pulse = 1 + Math.sin(now * 3) * 0.12;
  drawGlow(hand.x, hand.y, 70 * pulse, 'rgba(255,255,180,0.4)');
  if (currentScene === 'flowers') {
    const marker = { x: hand.x, y: hand.y, maxRadius: 24 * pulse, petals: 6, color: HAND_MARKER_COLOR, kind: 'daisy', rotation: now * 0.6 };
    drawFlower(marker, 1, 0.85, 0);
  } else {
    const img = getHandMarkerFish();
    const s = 0.45 * pulse;
    const iw = (img.width || FISH_LENGTH) * s;
    const ih = (img.height || FISH_LENGTH * 0.6) * s;
    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.translate(hand.x, hand.y);
    ctx.rotate(handMarkerHeading);
    ctx.drawImage(img, -iw / 2, -ih / 2, iw, ih);
    ctx.restore();
  }
}

let lastTime = performance.now();
function frame(t) {
  const dt = Math.min(0.05, (t - lastTime) / 1000);
  lastTime = t;
  const now = t / 1000;

  if (currentScene === 'flowers') {
    drawFlowerScene(now, dt);
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

  drawHandMarker(now);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
