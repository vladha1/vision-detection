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

function drawFlowerFace(r, face) {
  const eyeX = r * 0.11;
  const eyeY = -r * 0.03;
  if (face === 'scared') {
    ctx.fillStyle = '#2a2a2a';
    ctx.beginPath(); ctx.arc(-eyeX, eyeY, r * 0.065, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(eyeX, eyeY, r * 0.065, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = Math.max(1, r * 0.025);
    ctx.beginPath(); ctx.arc(0, r * 0.16, r * 0.05, 0, Math.PI * 2); ctx.stroke();
  } else if (face === 'happy') {
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = Math.max(1, r * 0.035);
    ctx.beginPath(); ctx.arc(-eyeX, eyeY + r * 0.05, r * 0.05, Math.PI, 0); ctx.stroke();
    ctx.beginPath(); ctx.arc(eyeX, eyeY + r * 0.05, r * 0.05, Math.PI, 0); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, r * 0.02, r * 0.16, 0.15 * Math.PI, 0.85 * Math.PI); ctx.stroke();
  }
}

function drawFlower(f, growth, alpha = 1, wobble = 0, face = null) {
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

  if (face) drawFlowerFace(r, face);

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
const GARDEN_TARGET_COUNT = 110;
const GARDEN_MAX_COUNT = 180; // safety cap so a lingering hand can't spawn forever
const GARDEN_BLOOM_SECONDS = 2.0;
const GARDEN_SPAWN_CHECK_SECONDS = 0.15;
const WOBBLE_RADIUS = 380;    // how far a hand's presence reaches into the garden
const WOBBLE_FREQ = 5;        // radians/sec - slow, gentle shimmer rather than a shake
const WOBBLE_AMOUNT = 0.11;   // radians
const WOBBLE_PULSE_AMOUNT = 0.12; // size-pulse on top of the rotation shimmer
const TOUCH_LEAN_RADIUS = 160;  // tighter than WOBBLE_RADIUS - a clear "it noticed you" cue
const TOUCH_LEAN_AMOUNT = 20;   // px, guaranteed lean-away displacement at zero distance
const REACTION_CHECK_SECONDS = 0.7; // how often we "roll the dice" while a hand is present
const REACTION_NEARBY_RADIUS = WOBBLE_RADIUS;
const REACTION_PATH_RADIUS = 140; // "in the way" distance for the dodge reaction
const REACTION_MOVE_DURATION = 1.8;   // slow, soothing glide rather than a snap
const REACTION_RELOCATE_DISTANCE = 220; // move nearby, not clear across the screen
const REACTION_DODGE_DISTANCE = 70;     // a small step aside, not a shove
const REACTION_DODGE_DURATION = 1.4;
const REACTION_COLOR_DURATION = 1.4;
const REACTION_FACE_SECONDS = 2.5;
const GRASS_ZONE_HEIGHT = 110;      // how far up from the bottom counts as "touching the grass"
const GRASS_TOUCH_CHECK_SECONDS = 0.6;
const GRASS_TOUCH_SPAWN_CHANCE = 0.4; // good frequency, but not guaranteed every check
let gardenFlowers = [];
let lastGrassTouchCheck = 0;
let lastGardenSpawnCheck = 0;
let lastReactionCheck = 0;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function smoothstep(t) { return t * t * (3 - 2 * t); } // gentle ease-in/ease-out

function makeGardenFlower(now, x, y) {
  return {
    x, y,
    maxRadius: 14 + Math.random() * 62, // wide size range - tiny buds to big blooms
    petals: 5 + Math.floor(Math.random() * 3),
    color: FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)],
    kind: randomKind(),
    rotation: Math.random() * Math.PI * 2,
    wobbleSeed: Math.random() * Math.PI * 2,
    bornAt: now,
    hold: 2.5 + Math.random() * 3,
    fade: 1.6 + Math.random() * 1.0,
  };
}

function spawnGardenFlower(now, x, y) {
  gardenFlowers.push(makeGardenFlower(now, x !== undefined ? x : Math.random() * W, y !== undefined ? y : Math.random() * H));
}

const SPROUT_RISE_MIN = 90;  // px risen from the grass line
const SPROUT_RISE_MAX = 260;

function spawnSproutingFlower(now, x) {
  if (gardenFlowers.length >= GARDEN_MAX_COUNT) return;
  const groundY = H - 6; // matches the grass baseline
  const restY = clamp(groundY - (SPROUT_RISE_MIN + Math.random() * (SPROUT_RISE_MAX - SPROUT_RISE_MIN)), EDGE_MARGIN, H - EDGE_MARGIN);
  const flower = makeGardenFlower(now, clamp(x, EDGE_MARGIN, W - EDGE_MARGIN), restY);
  flower.groundY = groundY;
  gardenFlowers.push(flower);
}

function startMove(f, targetX, targetY, now, duration = REACTION_MOVE_DURATION) {
  f.moveFrom = { x: f.x, y: f.y };
  f.moveTo = { x: clamp(targetX, EDGE_MARGIN, W - EDGE_MARGIN), y: clamp(targetY, EDGE_MARGIN, H - EDGE_MARGIN) };
  f.moveStart = now;
  f.moveDuration = duration;
}

function nearbyFlowers(radius) {
  return gardenFlowers.filter(f => dist(f, hand) < radius);
}

function reactSpawn(now) {
  if (gardenFlowers.length >= GARDEN_MAX_COUNT) return;
  const jitter = 30;
  spawnGardenFlower(now, hand.x + (Math.random() - 0.5) * jitter * 2, hand.y + (Math.random() - 0.5) * jitter * 2);
}

function reactRelocate(now) {
  const candidates = nearbyFlowers(REACTION_NEARBY_RADIUS).filter(f => f.moveStart === undefined);
  if (!candidates.length) return;
  const f = candidates[Math.floor(Math.random() * candidates.length)];
  const targetX = f.x + (Math.random() - 0.5) * 2 * REACTION_RELOCATE_DISTANCE;
  const targetY = f.y + (Math.random() - 0.5) * 2 * REACTION_RELOCATE_DISTANCE;
  startMove(f, targetX, targetY, now, REACTION_MOVE_DURATION);
}

function reactDodge(now) {
  const candidates = nearbyFlowers(REACTION_PATH_RADIUS).filter(f => f.moveStart === undefined);
  if (!candidates.length) return;
  const f = candidates[0];
  const away = norm(sub(f, hand));
  startMove(f, f.x + away.x * REACTION_DODGE_DISTANCE, f.y + away.y * REACTION_DODGE_DISTANCE, now, REACTION_DODGE_DURATION);
}

function reactRecolor(now) {
  const candidates = nearbyFlowers(REACTION_NEARBY_RADIUS).filter(f => f.colorChangeStart === undefined);
  if (!candidates.length) return;
  const f = candidates[Math.floor(Math.random() * candidates.length)];
  let nextColor = f.color;
  while (nextColor === f.color) nextColor = FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)];
  f.colorFrom = f.color;
  f.colorTo = nextColor;
  f.colorChangeStart = now;
}

function reactBecomeButterfly(now) {
  const candidates = nearbyFlowers(REACTION_NEARBY_RADIUS);
  if (!candidates.length) return;
  const f = candidates[Math.floor(Math.random() * candidates.length)];
  gardenFlowers = gardenFlowers.filter(x => x !== f);
  spawnButterfly(f.x, f.y, f.color);
}

function reactFace(now, face) {
  const candidates = nearbyFlowers(REACTION_NEARBY_RADIUS);
  if (!candidates.length) return;
  const f = candidates[Math.floor(Math.random() * candidates.length)];
  f.faceState = face;
  f.faceUntil = now + REACTION_FACE_SECONDS;
}

// While a hand lingers, occasionally: plant a new flower at the fingertip,
// relocate a nearby one a little, nudge one gently out of the way, recolor
// one, turn one into a butterfly, give one an expression - or do nothing at
// all, so the garden feels alive rather than mechanically responsive.
const REACTIONS = [
  { weight: 2.5, action: reactSpawn },
  { weight: 2.5, action: reactRelocate },
  { weight: 2, action: reactDodge },
  { weight: 2, action: reactRecolor },
  { weight: 1, action: reactBecomeButterfly },
  { weight: 1.2, action: (now) => reactFace(now, 'scared') },
  { weight: 1.2, action: (now) => reactFace(now, 'happy') },
  { weight: 4, action: null }, // no reaction - keeps it calm rather than constantly busy
];

function triggerHandReaction(now) {
  const total = REACTIONS.reduce((sum, r) => sum + r.weight, 0);
  let roll = Math.random() * total;
  for (const r of REACTIONS) {
    if (roll < r.weight) {
      if (r.action) r.action(now);
      return;
    }
    roll -= r.weight;
  }
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

function spawnButterfly(x, y, color) {
  butterflies.push({
    x, y,
    angle: Math.random() * Math.PI * 2,
    speed: 26 + Math.random() * 22,
    color: color || BUTTERFLY_COLORS[Math.floor(Math.random() * BUTTERFLY_COLORS.length)],
    wanderSeed: Math.random() * Math.PI * 2,
    flapSeed: Math.random() * Math.PI * 2,
  });
}

function initButterflies() {
  butterflies = [];
  for (let i = 0; i < BUTTERFLY_COUNT; i++) {
    spawnButterfly(Math.random() * W, Math.random() * H * 0.7);
  }
}

const BUTTERFLY_LAND_CHANCE = 0.3;   // odds a wandering butterfly heads for a flower at each decision point
const BUTTERFLY_FLEE_RADIUS = TOUCH_LEAN_RADIUS + 40; // hand distance from the flower that startles it off

function updateAndDrawButterflies(now, dt) {
  const margin = 24;
  for (const b of butterflies) {
    if (b.nextDecisionAt === undefined) b.nextDecisionAt = now + 2 + Math.random() * 4;

    if (b.state === 'landed') {
      const stillThere = b.target && gardenFlowers.includes(b.target);
      const satTooLong = now - b.landedAt > b.landedDuration;
      const handTooClose = hand && b.target && dist(b.target, hand) < BUTTERFLY_FLEE_RADIUS;
      if (!stillThere || satTooLong || handTooClose) {
        b.state = 'wander';
        b.nextDecisionAt = now + 3 + Math.random() * 4;
        b.angle = handTooClose ? Math.atan2(b.y - hand.y, b.x - hand.x) : Math.random() * Math.PI * 2;
        b.speed = handTooClose ? 90 + Math.random() * 30 : 26 + Math.random() * 22;
        b.target = null;
      } else {
        b.x = b.target.x + b.sitOffset.x;
        b.y = b.target.y + b.sitOffset.y;
      }
    } else if (b.state === 'seeking') {
      if (!b.target || !gardenFlowers.includes(b.target)) {
        b.state = 'wander';
        b.target = null;
      } else {
        const dx = b.target.x - b.x, dy = b.target.y - b.y;
        if (Math.hypot(dx, dy) < 16) {
          b.state = 'landed';
          b.landedAt = now;
          b.landedDuration = 2.5 + Math.random() * 3;
          b.sitOffset = { x: (Math.random() - 0.5) * 10, y: (Math.random() - 0.5) * 6 };
        } else {
          b.angle = Math.atan2(dy, dx);
          b.speed = 34 + Math.random() * 10;
        }
      }
    } else {
      b.state = 'wander';
      b.angle += Math.sin(now * 0.8 + b.wanderSeed) * dt * 1.6;
      if (now > b.nextDecisionAt) {
        b.nextDecisionAt = now + 3 + Math.random() * 4;
        if (Math.random() < BUTTERFLY_LAND_CHANCE && gardenFlowers.length) {
          b.target = gardenFlowers[Math.floor(Math.random() * gardenFlowers.length)];
          b.state = 'seeking';
        }
      }
    }

    if (b.state !== 'landed') {
      b.x += Math.cos(b.angle) * b.speed * dt;
      b.y += Math.sin(b.angle) * b.speed * dt + (b.state === 'wander' ? Math.sin(now * 2 + b.wanderSeed) * 8 * dt : 0);

      if (b.x < margin) { b.x = margin; b.angle = Math.PI - b.angle; }
      if (b.x > W - margin) { b.x = W - margin; b.angle = Math.PI - b.angle; }
      if (b.y < margin) { b.y = margin; b.angle = -b.angle; }
      if (b.y > H - margin) { b.y = H - margin; b.angle = -b.angle; }
    }

    const flapSpeed = b.state === 'landed' ? 4 : 14;
    const flap = Math.abs(Math.sin(now * flapSpeed + b.flapSeed));
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

  if (hand && hand.y > H - GRASS_ZONE_HEIGHT && now - lastGrassTouchCheck > GRASS_TOUCH_CHECK_SECONDS) {
    lastGrassTouchCheck = now;
    if (Math.random() < GRASS_TOUCH_SPAWN_CHANCE) {
      spawnSproutingFlower(now, hand.x + (Math.random() - 0.5) * 40);
    }
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
      const eased = smoothstep(mt);
      f.x = f.moveFrom.x + (f.moveTo.x - f.moveFrom.x) * eased;
      f.y = f.moveFrom.y + (f.moveTo.y - f.moveFrom.y) * eased;
      if (mt >= 1) f.moveStart = undefined;
    }

    if (f.colorChangeStart !== undefined) {
      const ct = Math.min(1, (now - f.colorChangeStart) / REACTION_COLOR_DURATION);
      f.color = lerpColor(f.colorFrom, f.colorTo, smoothstep(ct));
      if (ct >= 1) f.colorChangeStart = undefined;
    }

    let growth, alpha;
    if (age < GARDEN_BLOOM_SECONDS) {
      growth = smoothstep(age / GARDEN_BLOOM_SECONDS);
      alpha = growth;
    } else if (age < GARDEN_BLOOM_SECONDS + f.hold) {
      growth = 1;
      alpha = 1;
    } else {
      growth = 1;
      alpha = Math.max(0, 1 - (age - GARDEN_BLOOM_SECONDS - f.hold) / f.fade);
    }

    let wobble = 0;
    let leanX = 0, leanY = 0;
    if (hand) {
      const d = dist(f, hand);
      if (d < WOBBLE_RADIUS) {
        const intensity = 1 - d / WOBBLE_RADIUS;
        wobble = Math.sin(now * WOBBLE_FREQ + f.wobbleSeed) * WOBBLE_AMOUNT * intensity;
        growth *= 1 + Math.sin(now * WOBBLE_FREQ + f.wobbleSeed) * WOBBLE_PULSE_AMOUNT * intensity;
      }
      // guaranteed, deterministic lean-away from a close hand - independent of
      // the random reaction roll, so it's always clear the flower noticed you
      if (d < TOUCH_LEAN_RADIUS) {
        const leanIntensity = smoothstep(1 - d / TOUCH_LEAN_RADIUS);
        const away = norm(sub(f, hand));
        leanX = away.x * TOUCH_LEAN_AMOUNT * leanIntensity;
        leanY = away.y * TOUCH_LEAN_AMOUNT * leanIntensity;
      }
    }

    let riseY = 0;
    if (f.groundY !== undefined && age < GARDEN_BLOOM_SECONDS) {
      riseY = (f.groundY - f.y) * (1 - smoothstep(age / GARDEN_BLOOM_SECONDS));
    }

    const face = (f.faceState && now < f.faceUntil) ? f.faceState : null;
    const origX = f.x, origY = f.y;
    f.x += leanX;
    f.y += leanY + riseY;
    drawFlower(f, growth, alpha, wobble, face);
    f.x = origX;
    f.y = origY;
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
  // On the Pac-Man scene, a lost hand falls back to a marker anchored ahead
  // of Pac-Man (see pmCursorPx) instead of just vanishing - other scenes
  // still hide the marker outright when untracked.
  const fallback = !hand && currentScene === 'pacman' ? pmCursorPx : null;
  const source = hand || fallback;
  if (!source) {
    handMarkerLastPos = null;
    return;
  }
  if (handMarkerLastPos) {
    const dx = source.x - handMarkerLastPos.x;
    const dy = source.y - handMarkerLastPos.y;
    if (Math.hypot(dx, dy) > 2) {
      handMarkerHeading = Math.atan2(dy, dx);
    }
  }
  handMarkerLastPos = { x: source.x, y: source.y };

  const pulse = 1 + Math.sin(now * 3) * 0.12;
  // On the Pac-Man scene, keep the cursor within the board's actual pixel
  // rectangle instead of letting it drift into the letterboxed margins.
  const confine = currentScene === 'pacman' && pmBoardBounds;
  const mx = confine ? Math.max(pmBoardBounds.left, Math.min(pmBoardBounds.right, source.x)) : source.x;
  const my = confine ? Math.max(pmBoardBounds.top, Math.min(pmBoardBounds.bottom, source.y)) : source.y;

  drawGlow(mx, my, 70 * pulse, 'rgba(255,255,180,0.4)');
  if (currentScene === 'flowers') {
    const marker = { x: mx, y: my, maxRadius: 24 * pulse, petals: 6, color: HAND_MARKER_COLOR, kind: 'daisy', rotation: now * 0.6 };
    drawFlower(marker, 1, 0.85, 0);
  } else if (currentScene === 'fish') {
    const img = getHandMarkerFish();
    const s = 0.45 * pulse;
    const iw = (img.width || FISH_LENGTH) * s;
    const ih = (img.height || FISH_LENGTH * 0.6) * s;
    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.translate(mx, my);
    ctx.rotate(handMarkerHeading);
    ctx.drawImage(img, -iw / 2, -ih / 2, iw, ih);
    ctx.restore();
  } else {
    // pacman / driving: a solid, bright filled dot with a black outline ring
    // so it stays clearly visible against any part of the scene.
    const r = 16 * pulse;
    const crossColor = '#39ff9d';
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(mx, my, r + 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = crossColor;
    ctx.beginPath();
    ctx.arc(mx, my, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    if (currentScene === 'pacman' && pmPac && (pmPac.dir.dr || pmPac.dir.dc)) {
      const arrowAngle = Math.atan2(pmPac.dir.dr, pmPac.dir.dc);
      ctx.save();
      ctx.translate(mx, my);
      ctx.rotate(arrowAngle);
      ctx.fillStyle = '#ffd23f';
      ctx.beginPath();
      ctx.moveTo(r + 30, 0);
      ctx.lineTo(r + 14, -8);
      ctx.lineTo(r + 14, 8);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }
}

// ============================================================================
// Pac-Man scene: swipe your hand up/down/left/right to change direction,
// like nudging a joystick, rather than pointing at an absolute position.
// ============================================================================
// Classic maze layout (adapted from Dale Harvey's well-known open-source
// Pac-Man clone). 0=wall, 1=dot, 2=empty floor (no dot), 3=ghost-house
// block (treated as wall here), 4=power pellet.
const PM_RAW_MAP = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
  [0, 4, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 4, 0],
  [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
  [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
  [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
  [2, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 2, 2],
  [0, 0, 0, 0, 1, 0, 1, 0, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0],
  [2, 2, 2, 2, 1, 1, 1, 0, 3, 3, 3, 0, 1, 1, 1, 2, 2, 2, 2],
  [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
  [2, 2, 2, 0, 1, 0, 1, 1, 1, 2, 1, 1, 1, 0, 1, 0, 2, 2, 2],
  [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
  [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
  [0, 4, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 0],
  [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
  [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
  [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

const PM_COLS = 19;
const PM_ROWS = PM_RAW_MAP.length;
const PM_SPEED = 3.0; // cells/sec - slower than before so there's a real reaction window
const PM_GHOST_SPEED = 2.4;
const PM_TURN_TOLERANCE = 0.32; // how close to a cell center counts as "at an intersection" - wide enough that the ~100ms hand-tracking update interval reliably lands inside it
// Steering: a free cursor follows the raw hand position (no confinement, no
// gesture detection), and Pac-Man autonomously paths toward it - shortest
// route via BFS, recalculated at every intersection - rather than the
// player needing to time individual turns. This sidesteps the timing/noise
// problems that direction-gesture-based control kept running into: the
// exact cursor position barely matters since it just picks a rough
// destination, not a precise instant-by-instant direction.
const PM_TUNNEL_ROW = 10; // left/right wraparound passage, classic Pac-Man style
const PM_POWER_DURATION = 7;
const PM_POWER_COLOR = '#2233dd';
const PM_WALL_COLOR = '#1a3fbf';
const PM_GHOST_COLORS = ['#ff4d4d', '#ffb3f0', '#66e0ff'];

let pmGrid = null;
let pmDots = null;
let pmPowerDots = null;
let pmPowerUntil = 0;
let pmPac = null;
let pmGhosts = null;
let pmTargetRow = null; // nearest open cell to the raw cursor position - persists across hand dropouts
let pmTargetCol = null;
let pmCursorPx = null; // fallback marker position (ahead of Pac-Man) while the hand is untracked
let pmBoardBounds = null; // board's pixel rectangle, for confining the cursor marker
let pmCaughtUntil = 0;
let pmStarted = false;
let pmScore = 0;
let pmLives = 3;
let pmGameOverUntil = 0;
let pmInitialized = false;

function pmBuildMaze() {
  return PM_RAW_MAP.map(row => row.map(v => (v === 0 || v === 3) ? 1 : 0));
}

function pmResetDots() {
  pmDots = new Set();
  pmPowerDots = new Set();
  for (let r = 0; r < PM_ROWS; r++) {
    for (let c = 0; c < PM_COLS; c++) {
      if (PM_RAW_MAP[r][c] === 1) pmDots.add(`${r},${c}`);
      else if (PM_RAW_MAP[r][c] === 4) pmPowerDots.add(`${r},${c}`);
    }
  }
}

function pmIsWall(row, col) {
  const r = Math.round(row);
  const c = Math.round(col);
  if (r === PM_TUNNEL_ROW && (c < 0 || c >= PM_COLS)) return false; // open tunnel mouth
  if (r < 0 || r >= PM_ROWS || c < 0 || c >= PM_COLS) return true;
  return pmGrid[r][c] === 1;
}

function pmWrapTunnel(entity) {
  if (Math.round(entity.row) !== PM_TUNNEL_ROW) return;
  if (entity.col < -0.5) entity.col = PM_COLS - 0.5;
  else if (entity.col > PM_COLS - 0.5) entity.col = -0.5;
}

// Confines a raw (row, col) to the maze bounds and, if it lands on a wall,
// to the nearest open cell - so the cursor always maps to somewhere
// Pac-Man could plausibly path to.
function pmNearestOpenCell(row, col) {
  const r0 = Math.max(0, Math.min(PM_ROWS - 1, Math.round(row)));
  const c0 = Math.max(0, Math.min(PM_COLS - 1, Math.round(col)));
  if (!pmIsWall(r0, c0)) return { row: r0, col: c0 };
  const maxRadius = Math.max(PM_ROWS, PM_COLS);
  for (let radius = 1; radius <= maxRadius; radius++) {
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        if (Math.max(Math.abs(dr), Math.abs(dc)) !== radius) continue;
        const rr = r0 + dr, cc = c0 + dc;
        if (rr < 0 || rr >= PM_ROWS || cc < 0 || cc >= PM_COLS) continue;
        if (!pmIsWall(rr, cc)) return { row: rr, col: cc };
      }
    }
  }
  return { row: r0, col: c0 };
}

// BFS distance-to-target for every reachable cell, computed once per frame
// from the target outward, so looking up any cell's distance afterward is
// instant - cheap enough to redo every frame for a board this size.
function pmBFSDistances(targetRow, targetCol) {
  const dist = Array.from({ length: PM_ROWS }, () => new Array(PM_COLS).fill(Infinity));
  if (pmIsWall(targetRow, targetCol)) return dist;
  dist[targetRow][targetCol] = 0;
  const queue = [[targetRow, targetCol]];
  let qi = 0;
  while (qi < queue.length) {
    const [r, c] = queue[qi++];
    const d = dist[r][c];
    const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]];
    if (r === PM_TUNNEL_ROW) {
      if (c === 0) neighbors.push([r, PM_COLS - 1]);
      if (c === PM_COLS - 1) neighbors.push([r, 0]);
    }
    for (const [nr, nc] of neighbors) {
      if (nr < 0 || nr >= PM_ROWS || nc < 0 || nc >= PM_COLS) continue;
      if (pmIsWall(nr, nc)) continue;
      if (dist[nr][nc] > d + 1) {
        dist[nr][nc] = d + 1;
        queue.push([nr, nc]);
      }
    }
  }
  return dist;
}

// Picks the best next step from (r,c) toward whatever dist[][] measures
// distance-to-target from - prefers a real turn over reversing (only
// reverses if that's the only option), and among equally-good options,
// prefers continuing in the current direction to avoid needless zigzag.
function pmBestStepToward(r, c, currentDir, dist) {
  const reverseDir = { dr: -currentDir.dr, dc: -currentDir.dc };
  const options = [{ dr: -1, dc: 0 }, { dr: 1, dc: 0 }, { dr: 0, dc: -1 }, { dr: 0, dc: 1 }]
    .filter(d => !pmIsWall(r + d.dr, c + d.dc))
    .map(d => {
      let nc = c + d.dc;
      if (nc < 0) nc = PM_COLS - 1;
      else if (nc >= PM_COLS) nc = 0;
      const nr = r + d.dr;
      return { dr: d.dr, dc: d.dc, dist: dist ? dist[nr][nc] : 0 };
    });
  if (!options.length) return currentDir;
  const finite = options.filter(o => o.dist < Infinity);
  const pool = finite.length ? finite : options;

  // Distance decides first, among ALL options including reversing - if
  // turning around really is the shortest path (target is behind), it has
  // to win outright instead of being excluded upfront. Only among options
  // that are TIED for best does it prefer continuing straight, then any
  // other turn, and reverse last.
  const bestDist = Math.min(...pool.map(o => o.dist));
  const tied = pool.filter(o => o.dist === bestDist);
  const keepCurrent = tied.find(o => o.dr === currentDir.dr && o.dc === currentDir.dc);
  if (keepCurrent) return keepCurrent;
  const nonReverseTied = tied.filter(o => !(o.dr === reverseDir.dr && o.dc === reverseDir.dc));
  return nonReverseTied.length ? nonReverseTied[0] : tied[0];
}

function pmResetPositions() {
  pmPac = { row: 12, col: 9, dir: { dr: 0, dc: 0 } }; // the "door" cell just below the ghost house
  pmGhosts = PM_GHOST_COLORS.map((color, i) => {
    const homeRow = 8, homeCol = 8 + i; // spread across the open row just above the ghost house
    return { row: homeRow, col: homeCol, dir: { dr: 0, dc: -1 }, color, homeRow, homeCol };
  });
  pmStarted = false; // ghosts stay still until Pac-Man's first real move
  pmPowerUntil = 0;
}

function pmFullReset() {
  pmScore = 0;
  pmLives = 3;
  pmResetDots();
  pmResetPositions();
  pmCaughtUntil = 0;
  pmGameOverUntil = 0;
}

function pmInit() {
  pmInitialized = true;
  pmGrid = pmBuildMaze();
  pmFullReset();
}

function pmMoveGhost(g, dt, frightened) {
  const atCenter = Math.abs(g.row - Math.round(g.row)) < 0.1 && Math.abs(g.col - Math.round(g.col)) < 0.1;
  const r = Math.round(g.row), c = Math.round(g.col);
  const cellKey = `${r},${c}`;
  // Only make ONE direction decision per intersection visit - re-rolling a
  // fresh random direction every single frame while lingering near center
  // (this ran every frame atCenter was true) could flip-flop between
  // opposite directions frame to frame and net out to barely moving at all.
  if (atCenter && g.lastDecisionCell !== cellKey) {
    g.lastDecisionCell = cellKey;
    const options = [{ dr: -1, dc: 0 }, { dr: 1, dc: 0 }, { dr: 0, dc: -1 }, { dr: 0, dc: 1 }]
      .filter(d => !pmIsWall(r + d.dr, c + d.dc) && !(d.dr === -g.dir.dr && d.dc === -g.dir.dc));
    if (options.length) {
      g.row = r; g.col = c;
      if (Math.random() < 0.6) {
        options.sort((a, b) => {
          const da = Math.hypot((r + a.dr) - pmPac.row, (c + a.dc) - pmPac.col);
          const db = Math.hypot((r + b.dr) - pmPac.row, (c + b.dc) - pmPac.col);
          return frightened ? db - da : da - db; // frightened: prefer farthest, not closest
        });
        g.dir = options[0];
      } else {
        g.dir = options[Math.floor(Math.random() * options.length)];
      }
    }
  } else if (!atCenter) {
    g.lastDecisionCell = null; // clear once clearly away from center, ready for the next intersection
  }
  const speed = frightened ? PM_GHOST_SPEED * 0.6 : PM_GHOST_SPEED;
  g.row += g.dir.dr * speed * dt;
  g.col += g.dir.dc * speed * dt;
  pmWrapTunnel(g);
}

function drawPacmanScene(now, dt) {
  if (!pmInitialized) pmInit();

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);

  const tile = Math.min(W / PM_COLS, H / PM_ROWS);
  const offX = (W - tile * PM_COLS) / 2;
  const offY = (H - tile * PM_ROWS) / 2;
  const toPx = (row, col) => ({ x: offX + col * tile, y: offY + row * tile });

  // Shared with drawHandMarker so the cursor dot stays within the board's
  // pixel rectangle instead of drifting into the letterboxed margins.
  pmBoardBounds = { left: offX, top: offY, right: offX + tile * PM_COLS, bottom: offY + tile * PM_ROWS };

  const caught = now < pmCaughtUntil;
  const gameOver = now < pmGameOverUntil;
  if (pmGameOverUntil && !gameOver) {
    pmFullReset(); // game-over display finished - start a fresh game so it loops unattended
  }

  // Free cursor: just the raw hand position, confined only enough to map to
  // a real board cell. While the hand is untracked, rather than freezing at
  // the last raw position (which stalls Pac-Man once he reaches it), keep
  // projecting the target a few cells ahead along his current heading so he
  // keeps walking - and re-anchor to the real hand the moment it's sensed.
  if (hand) {
    const rawRow = (hand.y - offY) / tile;
    const rawCol = (hand.x - offX) / tile;
    const t = pmNearestOpenCell(rawRow, rawCol);
    pmTargetRow = t.row;
    pmTargetCol = t.col;
    pmCursorPx = null;
  } else {
    const dir = (pmPac.dir.dr || pmPac.dir.dc) ? pmPac.dir : { dr: 0, dc: -1 };
    const t = pmNearestOpenCell(pmPac.row + dir.dr * 3, pmPac.col + dir.dc * 3);
    pmTargetRow = t.row;
    pmTargetCol = t.col;
    pmCursorPx = { x: offX + (t.col + 0.5) * tile, y: offY + (t.row + 0.5) * tile };
  }
  const pmDist = pmTargetRow !== null ? pmBFSDistances(pmTargetRow, pmTargetCol) : null;

  if (!caught && !gameOver) {
    // Wide intersection window (not a narrow instant) so it reliably
    // overlaps with the hand tracker's ~100ms update interval.
    const atRow = Math.abs(pmPac.row - Math.round(pmPac.row)) < PM_TURN_TOLERANCE;
    const atCol = Math.abs(pmPac.col - Math.round(pmPac.col)) < PM_TURN_TOLERANCE;
    if (atRow && atCol && pmDist) {
      const r = Math.round(pmPac.row), c = Math.round(pmPac.col);
      const step = pmBestStepToward(r, c, pmPac.dir, pmDist);
      const changing = step.dr !== pmPac.dir.dr || step.dc !== pmPac.dir.dc;
      if (changing && !pmIsWall(r + step.dr, c + step.dc)) {
        pmPac.dir = step;
        pmPac.row = r; pmPac.col = c;
        pmStarted = true; // only counts as "started" once actually being directed somewhere
      }
    }
    if (pmPac.dir.dr || pmPac.dir.dc) {
      const aheadRow = pmPac.row + pmPac.dir.dr * 0.55;
      const aheadCol = pmPac.col + pmPac.dir.dc * 0.55;
      if (!pmIsWall(aheadRow, aheadCol)) {
        pmPac.row += pmPac.dir.dr * PM_SPEED * dt;
        pmPac.col += pmPac.dir.dc * PM_SPEED * dt;
      } else {
        // hit a wall - auto-turn onto whatever's open instead of stopping dead
        pmPac.row = Math.round(pmPac.row);
        pmPac.col = Math.round(pmPac.col);
        pmPac.dir = pmBestStepToward(pmPac.row, pmPac.col, pmPac.dir, pmDist);
      }
      pmWrapTunnel(pmPac);
    }

    const key = `${Math.round(pmPac.row)},${Math.round(pmPac.col)}`;
    if (pmDots.has(key)) { pmDots.delete(key); pmScore++; }
    if (pmPowerDots.has(key)) { pmPowerDots.delete(key); pmPowerUntil = now + PM_POWER_DURATION; pmScore += 10; }
    if (pmDots.size === 0 && pmPowerDots.size === 0) { pmResetDots(); }

    if (pmStarted) {
      const frightened = now < pmPowerUntil;
      for (const g of pmGhosts) pmMoveGhost(g, dt, frightened);
      for (const g of pmGhosts) {
        if (Math.hypot(g.row - pmPac.row, g.col - pmPac.col) < 0.6) {
          if (frightened) {
            g.row = g.homeRow; g.col = g.homeCol; g.dir = { dr: 0, dc: -1 };
            pmScore += 50;
          } else {
            pmLives -= 1;
            if (pmLives <= 0) {
              pmGameOverUntil = now + 3.5;
            } else {
              pmCaughtUntil = now + 1.4;
            }
            pmResetPositions();
            break;
          }
        }
      }
    }
  }

  // walls
  ctx.fillStyle = PM_WALL_COLOR;
  for (let r = 0; r < PM_ROWS; r++) {
    for (let c = 0; c < PM_COLS; c++) {
      if (pmGrid[r][c] === 1) {
        const p = toPx(r, c);
        ctx.fillRect(p.x + 1, p.y + 1, tile - 2, tile - 2);
      }
    }
  }

  // dots
  ctx.fillStyle = '#ffe9a8';
  for (const key of pmDots) {
    const [r, c] = key.split(',').map(Number);
    const p = toPx(r, c);
    ctx.beginPath();
    ctx.arc(p.x + tile / 2, p.y + tile / 2, tile * 0.08, 0, Math.PI * 2);
    ctx.fill();
  }

  // power pellets ("killer buttons") - eating one lets Pac-Man eat ghosts
  ctx.fillStyle = '#fff';
  for (const key of pmPowerDots) {
    const [r, c] = key.split(',').map(Number);
    const p = toPx(r, c);
    const pulse = 0.7 + 0.3 * Math.abs(Math.sin(now * 4));
    ctx.beginPath();
    ctx.arc(p.x + tile / 2, p.y + tile / 2, tile * 0.18 * pulse, 0, Math.PI * 2);
    ctx.fill();
  }

  // ghosts
  const frightenedNow = now < pmPowerUntil;
  const flashingNow = frightenedNow && (pmPowerUntil - now) < 2 && Math.floor(now * 6) % 2 === 0;
  for (const g of pmGhosts) {
    const p = toPx(g.row, g.col);
    const cx = p.x + tile / 2, cy = p.y + tile / 2;
    const gr = tile * 0.42;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.fillStyle = flashingNow ? '#ffffff' : (frightenedNow ? PM_POWER_COLOR : g.color);
    ctx.beginPath();
    ctx.arc(0, 0, gr, Math.PI, 0);
    ctx.lineTo(gr, gr * 0.7);
    for (let i = 0; i < 4; i++) {
      ctx.lineTo(gr - (i + 0.5) * (gr / 2), i % 2 === 0 ? gr * 0.35 : gr * 0.7);
    }
    ctx.lineTo(-gr, gr * 0.7);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(-gr * 0.35, -gr * 0.1, gr * 0.22, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(gr * 0.35, -gr * 0.1, gr * 0.22, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#1a1a3a';
    const lookX = Math.sign(g.dir.dc) * gr * 0.08, lookY = Math.sign(g.dir.dr) * gr * 0.08;
    ctx.beginPath(); ctx.arc(-gr * 0.35 + lookX, -gr * 0.1 + lookY, gr * 0.1, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(gr * 0.35 + lookX, -gr * 0.1 + lookY, gr * 0.1, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  // pac-man
  {
    const p = toPx(pmPac.row, pmPac.col);
    const cx = p.x + tile / 2, cy = p.y + tile / 2;
    const pr = tile * 0.42;
    const angle = pmPac.dir.dc || pmPac.dir.dr
      ? Math.atan2(pmPac.dir.dr, pmPac.dir.dc)
      : 0;
    const mouth = caught ? 0.5 : (Math.abs(Math.sin(now * 9)) * 0.28);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.fillStyle = '#ffd23f';
    ctx.beginPath();
    ctx.arc(0, 0, pr, mouth * Math.PI, (2 - mouth) * Math.PI);
    ctx.lineTo(0, 0);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // Highlight the target cell (where the free cursor currently maps to on
  // the board) so it's clear where Pac-Man is currently pathing toward.
  if (pmTargetRow !== null) {
    const p = toPx(pmTargetRow, pmTargetCol);
    ctx.save();
    ctx.globalAlpha = 0.5 + 0.3 * Math.abs(Math.sin(now * 4));
    ctx.strokeStyle = '#39ff9d';
    ctx.lineWidth = 3;
    ctx.strokeRect(p.x + 3, p.y + 3, tile - 6, tile - 6);
    ctx.restore();
  }

  ctx.fillStyle = '#fff';
  ctx.font = `${Math.round(tile * 0.5)}px sans-serif`;
  ctx.fillText(`Score: ${pmScore}`, offX, offY - tile * 0.25);
  ctx.textAlign = 'right';
  ctx.fillText(`Lives: ${Math.max(pmLives, 0)}`, offX + tile * PM_COLS, offY - tile * 0.25);
  ctx.textAlign = 'left';

  if (gameOver) {
    ctx.fillStyle = 'rgba(255,80,80,0.95)';
    ctx.font = `${Math.round(tile * 0.8)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('GAME OVER', W / 2, H / 2 - tile * 0.5);
    ctx.font = `${Math.round(tile * 0.5)}px sans-serif`;
    ctx.fillText(`Final Score: ${pmScore}`, W / 2, H / 2 + tile * 0.3);
    ctx.textAlign = 'left';
  } else if (caught) {
    ctx.fillStyle = 'rgba(255,80,80,0.9)';
    ctx.font = `${Math.round(tile * 0.8)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('Caught!', W / 2, H / 2);
    ctx.textAlign = 'left';
  }
}

// ============================================================================
// Driving scene: GTA-style free-roam - the car steers and drives toward
// wherever the hand currently is, slowing for sharp turns.
// ============================================================================
const CAR_MAX_SPEED = 240;
const CAR_TURN_RATE = Math.PI * 1.5;
const CAR_ARRIVE_RADIUS = 140;
const BUILDING_COLORS = ['#5c4a6e', '#4a5c6e', '#6e5c4a', '#4a6e5a', '#6e4a52'];
const GROUND_COLOR = '#2b2b2e';

let drivingInitialized = false;
let buildings = [];
let car = null;

function initDriving() {
  drivingInitialized = true;
  car = { x: W / 2, y: H / 2, heading: 0, speed: 0 };
  buildings = [];
  const count = 9;
  for (let i = 0; i < count; i++) {
    const w = 70 + Math.random() * 140;
    const h = 70 + Math.random() * 140;
    buildings.push({
      x: 40 + Math.random() * Math.max(1, W - 80 - w),
      y: 40 + Math.random() * Math.max(1, H - 80 - h),
      w, h,
      color: BUILDING_COLORS[Math.floor(Math.random() * BUILDING_COLORS.length)],
    });
  }
}

function carCollides(x, y) {
  for (const b of buildings) {
    if (x > b.x - 16 && x < b.x + b.w + 16 && y > b.y - 16 && y < b.y + b.h + 16) return true;
  }
  return false;
}

function updateCar(now, dt) {
  if (hand) {
    const dx = hand.x - car.x, dy = hand.y - car.y;
    const d = Math.hypot(dx, dy);
    const desiredHeading = Math.atan2(dy, dx);
    let diff = Math.atan2(Math.sin(desiredHeading - car.heading), Math.cos(desiredHeading - car.heading));
    const maxStep = CAR_TURN_RATE * dt;
    diff = Math.max(-maxStep, Math.min(maxStep, diff));
    car.heading += diff;
    const alignment = Math.max(0.15, Math.cos(diff));
    const targetSpeed = CAR_MAX_SPEED * Math.min(1, d / CAR_ARRIVE_RADIUS) * alignment;
    car.speed += (targetSpeed - car.speed) * Math.min(1, dt * 3);
  } else {
    car.speed *= 0.92;
  }

  const nx = car.x + Math.cos(car.heading) * car.speed * dt;
  const ny = car.y + Math.sin(car.heading) * car.speed * dt;
  if (!carCollides(nx, ny)) {
    car.x = clamp(nx, 20, W - 20);
    car.y = clamp(ny, 20, H - 20);
  } else {
    car.speed *= 0.25;
  }
}

function drawDrivingScene(now, dt) {
  if (!drivingInitialized) initDriving();

  updateCar(now, dt);

  ctx.fillStyle = GROUND_COLOR;
  ctx.fillRect(0, 0, W, H);

  for (const b of buildings) {
    ctx.fillStyle = b.color;
    ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 3;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
  }

  ctx.save();
  ctx.translate(car.x, car.y);
  ctx.rotate(car.heading);
  ctx.fillStyle = '#e63946';
  ctx.fillRect(-16, -9, 32, 18);
  ctx.fillStyle = '#a8dadc';
  ctx.fillRect(2, -7, 10, 14);
  ctx.fillStyle = '#1d1d1d';
  ctx.fillRect(-14, -10, 6, 3);
  ctx.fillRect(-14, 7, 6, 3);
  ctx.restore();
}

let lastTime = performance.now();
function frame(t) {
  const dt = Math.min(0.05, (t - lastTime) / 1000);
  lastTime = t;
  const now = t / 1000;

  if (currentScene === 'flowers') {
    drawFlowerScene(now, dt);
  } else if (currentScene === 'pacman') {
    drawPacmanScene(now, dt);
  } else if (currentScene === 'driving') {
    drawDrivingScene(now, dt);
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
