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
const FISH_MOODS = ['dance', 'dart', 'wiggle', 'hop']; // spontaneous joyful behaviours

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
    this.mood = null;      // current spontaneous mood (dance/dart/wiggle/hop)
    this.moodUntil = 0;
    this.nextMoodAt = 0;
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

    const reacting = (this.state === 'seek' || this.state === 'flee') && now < this.reactUntil && hand;

    // Spontaneous joyful "moods" while calm - a fish will occasionally dance
    // (swim in a tight circle), dart, wiggle or hop, just to bring the tank
    // to life even when nobody's interacting.
    if (!reacting) {
      if (!this.nextMoodAt) this.nextMoodAt = now + 2 + Math.random() * 5;
      if (now >= this.nextMoodAt && now >= this.moodUntil) {
        this.mood = FISH_MOODS[Math.floor(Math.random() * FISH_MOODS.length)];
        this.moodUntil = now + (this.mood === 'dance' ? 2.6 : this.mood === 'hop' ? 1.0 : 1.6);
        this.nextMoodAt = now + 4 + Math.random() * 6;
      }
    }
    const moodActive = !reacting && now < this.moodUntil;

    if (this.state === 'seek' && now < this.reactUntil && hand) {
      maxSpeed = SEEK_SPEED;
      steer = add(steer, this.steerToward(hand, maxSpeed, SEEK_ARRIVE_RADIUS));
    } else if (this.state === 'flee' && now < this.reactUntil && hand) {
      maxSpeed = FLEE_SPEED;
      steer = add(steer, this.fleeFrom(hand));
    } else {
      this.state = 'wander';
      steer = add(steer, this.wanderForce(now));
      if (moodActive) {
        if (this.mood === 'dance') {
          const perp = rotateVec(norm(this.vel), Math.PI / 2);
          steer = add(steer, scale(perp, 260)); maxSpeed = 160;
        } else if (this.mood === 'dart') {
          steer = add(steer, scale(norm(this.vel), 320)); maxSpeed = 280;
          if (bubbles.length < 70 && Math.random() < 0.4) bubbles.push(makeBubbleAt(this.pos.x, this.pos.y));
        } else if (this.mood === 'hop') {
          steer.y -= 320; maxSpeed = 200;
        }
        // 'wiggle' is purely visual (see draw)
      }
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

  draw(now = 0) {
    const angle = this.heading;
    const img = this.image;
    const iw = img.width || FISH_LENGTH;
    const ih = img.height || FISH_LENGTH * 0.6;
    const bob = Math.sin(now * 3 + this.wanderPhase) * 2; // subtle life even at rest
    let sx = 1, sy = 1;
    if (this.mood === 'wiggle' && now < this.moodUntil) {
      sy = 1 + Math.sin(now * 20) * 0.2;   // squash-and-stretch giggle
      sx = 1 - Math.sin(now * 20) * 0.1;
    } else if (this.mood === 'dance' && now < this.moodUntil) {
      const s = 1 + Math.sin(now * 8) * 0.08;
      sx = s; sy = s;
    }
    ctx.save();
    ctx.translate(this.pos.x, this.pos.y + bob);
    ctx.rotate(angle);
    ctx.scale(sx, sy);
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
const GARDEN_TARGET_COUNT = 150;
const GARDEN_MAX_COUNT = 260; // safety cap so a lingering hand can't spawn forever
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

// A burst of a few flowers blooming together right at the fingertip.
function reactBurst(now) {
  const n = 4 + Math.floor(Math.random() * 3);
  for (let i = 0; i < n; i++) {
    if (gardenFlowers.length >= GARDEN_MAX_COUNT) break;
    const a = Math.random() * Math.PI * 2, r = 10 + Math.random() * 60;
    spawnGardenFlower(now, hand.x + Math.cos(a) * r, hand.y + Math.sin(a) * r);
  }
}

// A ring of flowers blooming in a circle around the hand.
function reactRing(now) {
  const n = 6, R = 95;
  for (let i = 0; i < n; i++) {
    if (gardenFlowers.length >= GARDEN_MAX_COUNT) break;
    const a = (i / n) * Math.PI * 2;
    spawnGardenFlower(now, hand.x + Math.cos(a) * R, hand.y + Math.sin(a) * R);
  }
}

function reactSpin(now) {
  const c = nearbyFlowers(REACTION_NEARBY_RADIUS).filter(f => f.spinStart === undefined);
  if (!c.length) return;
  const f = c[Math.floor(Math.random() * c.length)];
  f.spinStart = now;
  f.spinDir = Math.random() < 0.5 ? 1 : -1;
}

function reactPulse(now) {
  const c = nearbyFlowers(REACTION_NEARBY_RADIUS).filter(f => f.pulseStart === undefined);
  if (!c.length) return;
  c[Math.floor(Math.random() * c.length)].pulseStart = now;
}

function reactRainbow(now) {
  const c = nearbyFlowers(REACTION_NEARBY_RADIUS);
  if (!c.length) return;
  c[Math.floor(Math.random() * c.length)].rainbowUntil = now + 3.5;
}

function reactSparkle(now) {
  const c = nearbyFlowers(REACTION_NEARBY_RADIUS);
  if (!c.length) return;
  const f = c[Math.floor(Math.random() * c.length)];
  emitSparkles(f.x, f.y, f.color);
}

// A staggered bounce that ripples left-to-right through nearby flowers.
function reactWave(now) {
  const c = nearbyFlowers(REACTION_NEARBY_RADIUS).filter(f => f.bounceStart === undefined);
  c.sort((a, b) => a.x - b.x);
  c.slice(0, 9).forEach((f, i) => { f.bounceStart = now + i * 0.08; });
}

// While a hand lingers, the garden picks one of many playful reactions -
// planting, bursting or ringing new blooms, relocating/dodging, recoloring,
// spinning, pulsing, rainbow-cycling, sparkling, a rippling wave, turning a
// flower into a butterfly, or a facial expression - or, less often now, doing
// nothing, so it feels lively and generous rather than idle.
const REACTIONS = [
  { weight: 2.0, action: reactSpawn },
  { weight: 1.6, action: reactRelocate },
  { weight: 1.4, action: reactDodge },
  { weight: 1.6, action: reactRecolor },
  { weight: 1.8, action: reactBurst },
  { weight: 1.0, action: reactRing },
  { weight: 1.6, action: reactSpin },
  { weight: 1.6, action: reactPulse },
  { weight: 1.2, action: reactRainbow },
  { weight: 1.8, action: reactSparkle },
  { weight: 1.2, action: reactWave },
  { weight: 0.8, action: reactBecomeButterfly },
  { weight: 1.0, action: (now) => reactFace(now, 'scared') },
  { weight: 1.0, action: (now) => reactFace(now, 'happy') },
  { weight: 2.0, action: null }, // occasional pause - keeps it from feeling frantic
];

// --- sparkles: short-lived glints thrown off by a sparkling flower ---
let sparkles = [];
function emitSparkles(x, y, color) {
  for (let i = 0; i < 12; i++) {
    const a = Math.random() * Math.PI * 2, sp = 40 + Math.random() * 120;
    sparkles.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 40, life: 0.7 + Math.random() * 0.7, age: 0, color, r: 2 + Math.random() * 3 });
  }
}
function updateAndDrawSparkles(dt) {
  sparkles = sparkles.filter(s => {
    s.age += dt; s.vy += 180 * dt; s.x += s.vx * dt; s.y += s.vy * dt;
    return s.age < s.life;
  });
  for (const s of sparkles) {
    ctx.globalAlpha = Math.max(0, 1 - s.age / s.life);
    ctx.fillStyle = s.color;
    ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// --- drifting petals: gentle ambient motion so the whole frame feels alive ---
let petals = [];
function makePetal(y) {
  return {
    x: Math.random() * W,
    y: y !== undefined ? y : -10,
    vy: 8 + Math.random() * 20,
    drift: (Math.random() - 0.5) * 26,
    seed: Math.random() * Math.PI * 2,
    r: 2.5 + Math.random() * 3.5,
    color: FLOWER_COLORS[Math.floor(Math.random() * FLOWER_COLORS.length)],
  };
}
function initPetals() {
  petals = [];
  for (let i = 0; i < 30; i++) petals.push(makePetal(Math.random() * H));
}
function drawPetals(now, dt) {
  ctx.globalAlpha = 0.5;
  for (const p of petals) {
    p.y += p.vy * dt;
    p.x += Math.sin(now * 0.8 + p.seed) * p.drift * dt;
    if (p.y > H + 10) Object.assign(p, makePetal(-10));
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.ellipse(p.x, p.y, p.r, p.r * 0.55, now * 0.5 + p.seed, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

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
  const count = Math.max(30, Math.floor(W / 11));
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
const BUTTERFLY_COUNT = 10;
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
    initPetals();
  }

  ctx.fillStyle = '#03121f';
  ctx.fillRect(0, 0, W, H);
  drawPetals(now, dt);
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

    if (f.rainbowUntil !== undefined) {
      if (now < f.rainbowUntil) f.color = `hsl(${Math.round(now * 130) % 360}, 80%, 66%)`;
      else f.rainbowUntil = undefined;
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
    // Spin: two eased full turns (4π returns to the original angle, so no
    // permanent offset). Pulse: balloon up to ~1.8x and back.
    if (f.spinStart !== undefined) {
      const st = Math.min(1, (now - f.spinStart) / 1.6);
      wobble += f.spinDir * smoothstep(st) * Math.PI * 4;
      if (st >= 1) f.spinStart = undefined;
    }
    if (f.pulseStart !== undefined) {
      const pt = Math.min(1, (now - f.pulseStart) / 1.2);
      growth *= 1 + Math.sin(pt * Math.PI) * 0.8;
      if (pt >= 1) f.pulseStart = undefined;
    }
    let bounceY = 0;
    if (f.bounceStart !== undefined && now >= f.bounceStart) {
      const bt = (now - f.bounceStart) / 0.5;
      if (bt >= 1) f.bounceStart = undefined;
      else bounceY = -Math.sin(bt * Math.PI) * 26;
    }

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
    f.y += leanY + riseY + bounceY;
    drawFlower(f, growth, alpha, wobble, face);
    f.x = origX;
    f.y = origY;
    return true;
  });

  updateAndDrawSparkles(dt);
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

// --- mobile controller relay: a phone posts arrow presses to /api/control
// and reads the live score from /api/pmstate. While a controller is active
// the display drives Pac-Man from those arrows and ignores the hand (the
// server also stops reporting a hand for the Pac-Man scene). ---
const PM_ARROW_VECTORS = {
  up: { dr: -1, dc: 0 }, down: { dr: 1, dc: 0 },
  left: { dr: 0, dc: -1 }, right: { dr: 0, dc: 1 },
};
let pmControllerActive = false;
let pmArrowDir = null;
let pmLastResetSeq = null; // watches the server reset counter; a change restarts the game
async function syncControl() {
  try {
    const d = await (await fetch('/api/control')).json();
    pmControllerActive = d.mode === 'controller';
    pmArrowDir = (d.dir && PM_ARROW_VECTORS[d.dir]) ? PM_ARROW_VECTORS[d.dir] : null;
    if (pmLastResetSeq === null) {
      pmLastResetSeq = d.reset; // first sync: adopt current value, don't reset on load
    } else if (d.reset !== pmLastResetSeq) {
      pmLastResetSeq = d.reset;
      if (pmInitialized) pmFullReset();
    }
  } catch (e) {
    pmControllerActive = false;
    pmArrowDir = null;
  }
}
syncControl();
setInterval(syncControl, 100);

async function pushPmState() {
  if (currentScene !== 'pacman') return;
  try {
    await fetch('/api/pmstate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ score: pmScore, lives: pmLives, level: pmLevel }),
    });
  } catch (e) { /* display keeps running even if the relay is down */ }
}
setInterval(pushPmState, 500);

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
  // The paint scene draws its own always-visible cursor (with dwell ring and
  // pen state), so skip the generic marker there entirely.
  if (currentScene === 'paint') return;
  // On the Pac-Man scene the marker follows the smoothed cursor (pmCursorPx),
  // which eases toward the hand when present and leads Pac-Man when it isn't -
  // and stays hidden until the first hand appears. Other scenes track the raw
  // hand and hide the marker outright when untracked.
  const source = currentScene === 'pacman' ? pmCursorPx : hand;
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

  if (currentScene !== 'paint' && currentScene !== 'constellation') {
    drawGlow(mx, my, 70 * pulse, 'rgba(255,255,180,0.4)');
  }
  if (currentScene === 'flowers') {
    const marker = { x: mx, y: my, maxRadius: 24 * pulse, petals: 6, color: HAND_MARKER_COLOR, kind: 'daisy', rotation: now * 0.6 };
    drawFlower(marker, 1, 0.85, 0);
  } else if (currentScene === 'constellation') {
    // a cool glowing star-cursor that hints at the next star to connect
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    drawGlow(mx, my, 26 * pulse, 'rgba(150,190,255,0.7)');
    ctx.fillStyle = 'rgba(215,232,255,0.95)';
    ctx.beginPath(); ctx.arc(mx, my, 4 * pulse, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
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
const PM_SPEED = 2.9; // cells/sec - clearly quicker than the ghosts so it's easy to escape
const PM_GHOST_SPEED = 1.5; // kept slower than Pac-Man to make the game more forgiving
// Error-adaptive cursor smoothing: the cursor eases toward the hand with a
// time constant that STRETCHES when the hand is near it (steady, accurate,
// jitter-proof fine control) and SHRINKS when the hand makes a big deliberate
// move (snappy, responsive) - so it's slow by default yet quick when needed.
const PM_CURSOR_TAU_SLOW = 0.60; // s: heavy smoothing when the hand sits near the cursor
const PM_CURSOR_TAU_FAST = 0.16; // s: light smoothing when the hand is far from the cursor
const PM_CURSOR_SNAP_TILES = 6;  // hand-to-cursor error (tiles) at which smoothing is fully fast
const PM_TURN_TOLERANCE = 0.32; // how close to a cell center counts as "at an intersection" - wide enough that the ~100ms hand-tracking update interval reliably lands inside it
// Steering: a free cursor follows the raw hand position (no confinement, no
// gesture detection), and Pac-Man autonomously paths toward it - shortest
// route via BFS, recalculated at every intersection - rather than the
// player needing to time individual turns. This sidesteps the timing/noise
// problems that direction-gesture-based control kept running into: the
// exact cursor position barely matters since it just picks a rough
// destination, not a precise instant-by-instant direction.
const PM_TUNNEL_ROW = 10; // left/right wraparound passage, classic Pac-Man style
const PM_POWER_DURATION = 15;
const PM_POWER_COLOR = '#2233dd';
const PM_WALL_COLOR = '#1a3fbf';
const PM_GHOST_COLORS = ['#ff4d4d', '#ffb3f0', '#66e0ff'];

let pmGrid = null;
let pmDots = null;
let pmPowerDots = null;
let pmPowerUntil = 0;
let pmPac = null;
let pmGhosts = null;
let pmTargetRow = null; // nearest open cell to the smoothed cursor - drives the BFS pathing
let pmTargetCol = null;
let pmCursorPx = null; // smoothed cursor marker pixel position {x,y}, eased toward its goal each frame
let pmHandSeen = false; // gates all motion until a hand first appears, so the game waits for a player
let pmLastHeading = { dr: 0, dc: -1 }; // remembered heading, so the default cursor can lead Pac-Man
let pmBoardBounds = null; // board's pixel rectangle, for confining the cursor marker
let pmCaughtUntil = 0;
let pmStarted = false;
let pmScore = 0;
let pmLives = 3;
let pmLevel = 1;
let pmGameOverUntil = 0;
let pmBoardClearedUntil = 0; // brief "Board Cleared!" pause before the next board loads
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

// Walks forward along `dir` from (row, col), up to maxLead cells, stopping
// before the first wall/edge - so the returned cell is always a valid open
// cell genuinely AHEAD of the entity (used to lead the default cursor).
function pmLeadCell(row, col, dir, maxLead) {
  const r = Math.round(row), c = Math.round(col);
  let lr = r, lc = c;
  for (let i = 1; i <= maxLead; i++) {
    const nr = r + dir.dr * i, nc = c + dir.dc * i;
    if (nr < 0 || nr >= PM_ROWS || nc < 0 || nc >= PM_COLS) break;
    if (pmIsWall(nr, nc)) break;
    lr = nr; lc = nc;
  }
  return { row: lr, col: lc };
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
  pmLevel = 1;
  pmResetDots();
  pmResetPositions();
  pmCaughtUntil = 0;
  pmGameOverUntil = 0;
  pmBoardClearedUntil = 0;
  pmHandSeen = false; // wait for a hand again before the fresh game starts
  pmCursorPx = null;
  pmLastHeading = { dr: 0, dc: -1 };
}

// Ghosts speed up each level, but never enough to out-run Pac-Man.
function pmGhostSpeed() {
  return Math.min(PM_GHOST_SPEED + (pmLevel - 1) * 0.2, PM_SPEED - 0.3);
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
  const base = pmGhostSpeed();
  const speed = frightened ? base * 0.6 : base;
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

  const boardCleared = now < pmBoardClearedUntil;
  if (pmBoardClearedUntil && !boardCleared) {
    // "Board Cleared!" pause finished - load the next board.
    pmResetDots();
    pmResetPositions();
    pmBoardClearedUntil = 0;
  }

  // A mobile controller (if one is active) drives Pac-Man with arrows and
  // suppresses the hand cursor entirely.
  const controllerMode = pmControllerActive;

  // --- Smoothed cursor & target (hand mode only) ------------------------
  // The cursor is a persistent pixel position that eases toward a goal every
  // frame, so it never snaps abruptly (and it filters the raw ~10Hz hand
  // jitter). With a hand it eases toward the hand; with no hand it eases
  // toward a point a few cells AHEAD of Pac-Man, so the default cursor leads
  // him rather than freezing behind. All hand motion waits for a first hand.
  let pmDist = null;
  if (controllerMode) {
    pmCursorPx = null;
    pmTargetRow = null;
    pmTargetCol = null;
  } else {
    if (hand) pmHandSeen = true;
    if (pmPac.dir.dr || pmPac.dir.dc) pmLastHeading = pmPac.dir;

    let goalPx = null;
    let adaptive = false;
    if (hand) {
      goalPx = {
        x: Math.max(offX, Math.min(offX + tile * PM_COLS, hand.x)),
        y: Math.max(offY, Math.min(offY + tile * PM_ROWS, hand.y)),
      };
      adaptive = true; // hand steering uses error-adaptive smoothing
    } else if (pmHandSeen) {
      const dir = (pmPac.dir.dr || pmPac.dir.dc) ? pmPac.dir : pmLastHeading;
      const lead = pmLeadCell(pmPac.row, pmPac.col, dir, 3);
      goalPx = { x: offX + (lead.col + 0.5) * tile, y: offY + (lead.row + 0.5) * tile };
    }

    if (goalPx) {
      if (!pmCursorPx) {
        pmCursorPx = { x: goalPx.x, y: goalPx.y };
      } else {
        let tau;
        if (adaptive) {
          // Stretch the time constant when the hand is near the cursor (fine,
          // accurate) and shrink it when far (quick) - the error itself picks
          // the speed, so jitter stays calm while big moves snap through.
          const errTiles = Math.hypot(goalPx.x - pmCursorPx.x, goalPx.y - pmCursorPx.y) / tile;
          const f = Math.min(1, errTiles / PM_CURSOR_SNAP_TILES);
          tau = PM_CURSOR_TAU_SLOW - (PM_CURSOR_TAU_SLOW - PM_CURSOR_TAU_FAST) * f;
        } else {
          tau = 0.18; // no-hand lead point moves cell-to-cell; steady easing is fine
        }
        const k = 1 - Math.exp(-dt / tau); // framerate-independent easing
        pmCursorPx.x += (goalPx.x - pmCursorPx.x) * k;
        pmCursorPx.y += (goalPx.y - pmCursorPx.y) * k;
      }
      const t = pmNearestOpenCell((pmCursorPx.y - offY) / tile, (pmCursorPx.x - offX) / tile);
      pmTargetRow = t.row;
      pmTargetCol = t.col;
    } else {
      pmCursorPx = null;
      pmTargetRow = null;
      pmTargetCol = null;
    }
    pmDist = pmTargetRow !== null ? pmBFSDistances(pmTargetRow, pmTargetCol) : null;
  }

  const ready = controllerMode ? true : pmHandSeen;
  if (!caught && !gameOver && !boardCleared && ready) {
   if (controllerMode) {
    // --- Classic arrow control (mobile controller) ----------------------
    // Buffered turn: adopt the pressed arrow at the next cell centre if it's
    // open, otherwise keep the current heading; stop dead when it hits a wall.
    const atR = Math.abs(pmPac.row - Math.round(pmPac.row)) < PM_TURN_TOLERANCE;
    const atC = Math.abs(pmPac.col - Math.round(pmPac.col)) < PM_TURN_TOLERANCE;
    if (atR && atC && pmArrowDir) {
      const r = Math.round(pmPac.row), c = Math.round(pmPac.col);
      if (!pmIsWall(r + pmArrowDir.dr, c + pmArrowDir.dc)) {
        if (pmArrowDir.dr !== pmPac.dir.dr || pmArrowDir.dc !== pmPac.dir.dc) {
          pmPac.row = r; pmPac.col = c;
        }
        pmPac.dir = pmArrowDir;
        pmStarted = true;
      }
    }
    if (pmPac.dir.dr || pmPac.dir.dc) {
      const aheadRow = pmPac.row + pmPac.dir.dr * 0.55;
      const aheadCol = pmPac.col + pmPac.dir.dc * 0.55;
      if (!pmIsWall(aheadRow, aheadCol)) {
        pmPac.row += pmPac.dir.dr * PM_SPEED * dt;
        pmPac.col += pmPac.dir.dc * PM_SPEED * dt;
      } else {
        pmPac.row = Math.round(pmPac.row);
        pmPac.col = Math.round(pmPac.col);
        pmPac.dir = { dr: 0, dc: 0 };
      }
      pmWrapTunnel(pmPac);
    }
   } else {
    // Wide intersection window (not a narrow instant) so it reliably
    // overlaps with the hand tracker's ~100ms update interval.
    const atRow = Math.abs(pmPac.row - Math.round(pmPac.row)) < PM_TURN_TOLERANCE;
    const atCol = Math.abs(pmPac.col - Math.round(pmPac.col)) < PM_TURN_TOLERANCE;
    if (atRow && atCol && pmDist) {
      const r = Math.round(pmPac.row), c = Math.round(pmPac.col);
      if (r === pmTargetRow && c === pmTargetCol) {
        // Arrived at the target cell: park exactly on it and wait. Without
        // this he'd step onto a neighbour (all one step from the target) and
        // the BFS would immediately send him back - an endless oscillation
        // whenever the hand is held still on one spot.
        pmPac.row = r; pmPac.col = c;
        pmPac.dir = { dr: 0, dc: 0 };
      } else {
        const step = pmBestStepToward(r, c, pmPac.dir, pmDist);
        const changing = step.dr !== pmPac.dir.dr || step.dc !== pmPac.dir.dc;
        if (changing && !pmIsWall(r + step.dr, c + step.dc)) {
          pmPac.dir = step;
          pmPac.row = r; pmPac.col = c;
          pmStarted = true; // only counts as "started" once actually being directed somewhere
        }
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
   }

    const key = `${Math.round(pmPac.row)},${Math.round(pmPac.col)}`;
    if (pmDots.has(key)) { pmDots.delete(key); pmScore++; }
    if (pmPowerDots.has(key)) { pmPowerDots.delete(key); pmPowerUntil = now + PM_POWER_DURATION; pmScore += 10; }
    if (pmDots.size === 0 && pmPowerDots.size === 0) {
      // Board cleared! Bump the level and pause briefly before the next board.
      pmLevel += 1;
      pmScore += 100; // clear bonus
      pmPowerUntil = 0;
      pmBoardClearedUntil = now + 2.5;
    }

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
  // Blink the whole time the power pellet is active (white<->blue), and blink
  // faster in the last 2s as a "running out" warning.
  const blinkRate = (pmPowerUntil - now) < 2 ? 8 : 3.5;
  const flashingNow = frightenedNow && Math.floor(now * blinkRate) % 2 === 0;
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
  ctx.textAlign = 'center';
  ctx.fillText(`Level ${pmLevel}`, offX + tile * PM_COLS / 2, offY - tile * 0.25);
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
  } else if (boardCleared) {
    ctx.fillStyle = `rgba(90,255,150,${0.7 + 0.3 * Math.abs(Math.sin(now * 6))})`;
    ctx.font = `${Math.round(tile * 0.8)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('Board Cleared!', W / 2, H / 2 - tile * 0.5);
    ctx.font = `${Math.round(tile * 0.5)}px sans-serif`;
    ctx.fillText(`Level ${pmLevel}`, W / 2, H / 2 + tile * 0.3);
    ctx.textAlign = 'left';
  } else if (controllerMode && !pmStarted) {
    ctx.fillStyle = `rgba(255,255,255,${0.6 + 0.35 * Math.abs(Math.sin(now * 2.5))})`;
    ctx.font = `${Math.round(tile * 0.7)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('Press an arrow to start', W / 2, H / 2);
    ctx.textAlign = 'left';
  } else if (!controllerMode && !pmHandSeen) {
    ctx.fillStyle = `rgba(255,255,255,${0.6 + 0.35 * Math.abs(Math.sin(now * 2.5))})`;
    ctx.font = `${Math.round(tile * 0.7)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('Raise your hand to start', W / 2, H / 2);
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

// ============================================================================
// Fish scene ambience: gradient water, light rays, swaying kelp, a sandy
// floor and rising bubbles - so the tank feels like a full, living reef
// around the fish rather than a bare blue rectangle.
// ============================================================================
let fishSceneInit = false;
let bubbles = [];
let kelp = [];

function makeBubble(y) {
  return {
    x: Math.random() * W,
    y: y !== undefined ? y : H + 10,
    r: 2 + Math.random() * 6,
    speed: 20 + Math.random() * 55,
    wobbleSeed: Math.random() * Math.PI * 2,
  };
}

function makeBubbleAt(x, y) {
  return { x, y, r: 2 + Math.random() * 3, speed: 30 + Math.random() * 45, wobbleSeed: Math.random() * Math.PI * 2 };
}

function initFishScene() {
  fishSceneInit = true;
  bubbles = [];
  for (let i = 0; i < 40; i++) bubbles.push(makeBubble(Math.random() * H));
  kelp = [];
  const strands = 8 + Math.floor(W / 200);
  for (let i = 0; i < strands; i++) {
    kelp.push({
      x: (i + 0.5) / strands * W + (Math.random() - 0.5) * 70,
      height: 130 + Math.random() * 230,
      phase: Math.random() * Math.PI * 2,
      width: 9 + Math.random() * 13,
      hue: 120 + Math.random() * 45,
    });
  }
}

function drawFishScene(now, dt) {
  if (!fishSceneInit) initFishScene();

  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#0d3f68');
  g.addColorStop(0.55, WATER_COLOR);
  g.addColorStop(1, '#05131f');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  // slow, angled shafts of light
  ctx.save();
  ctx.globalAlpha = 0.05;
  ctx.fillStyle = '#cdeaff';
  for (let i = 0; i < 4; i++) {
    const x = ((i + 0.5) / 4) * W + Math.sin(now * 0.2 + i) * 40;
    ctx.beginPath();
    ctx.moveTo(x - 35, 0); ctx.lineTo(x + 35, 0); ctx.lineTo(x + 130, H); ctx.lineTo(x - 30, H);
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();

  // swaying kelp rooted to the floor
  ctx.lineCap = 'round';
  for (const k of kelp) {
    ctx.strokeStyle = `hsl(${k.hue}, 55%, 32%)`;
    ctx.lineWidth = k.width;
    ctx.beginPath();
    const segs = 8;
    for (let s = 0; s <= segs; s++) {
      const t = s / segs;
      const y = H - t * k.height;
      const x = k.x + Math.sin(now * 1.1 + k.phase + t * 2.2) * 24 * t;
      if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // rising bubbles
  ctx.strokeStyle = 'rgba(200,230,255,0.45)';
  ctx.lineWidth = 1.5;
  for (const b of bubbles) {
    b.y -= b.speed * dt;
    b.x += Math.sin(now * 2 + b.wobbleSeed) * 0.5;
    if (b.y < -10) Object.assign(b, makeBubble());
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // sandy floor
  ctx.fillStyle = '#c9ad7a';
  ctx.beginPath();
  ctx.moveTo(0, H);
  ctx.lineTo(0, H - 22);
  for (let x = 0; x <= W; x += 40) ctx.lineTo(x, H - 22 + Math.sin(x * 0.05) * 6);
  ctx.lineTo(W, H);
  ctx.closePath();
  ctx.fill();

  for (const f of fishes) {
    f.update(hand, now, dt);
    f.draw(now);
  }
}

// ============================================================================
// Paint wall: a real, PERSISTENT finger-painting tool. Because a tracked hand
// has no physical click, "clicking" is done by DWELL - hold the cursor still
// for a moment and a ring fills up to confirm. A dwell over a palette item
// selects it; a dwell on the canvas toggles the PEN up/down, so you choose
// when you're drawing vs just moving. The cursor is always on screen (it holds
// its last position if tracking briefly drops). After a long idle the wall
// auto-wipes for the next person.
// ============================================================================
let paintLayer = null, paintLayerCtx = null, paintLayerW = 0, paintLayerH = 0;
let paintLast = null;
let paintColor = '#1e88e5';
let paintErase = false;
let paintBrush = 28;
let paintPenDown = false;
let paintCursor = null;             // last known cursor position (kept visible on dropout)
let dwellAnchor = null, dwellStart = 0, dwellArmed = false;
let paintClickFlash = 0;            // time of last click, for a brief confirm pulse
let paintLastHand = 0;
let paintSpeed = 0;                 // smoothed hand speed (px/s)
let paintPrevPos = null;
const PAINT_WALL_COLOR = '#ece7dd';       // an off-white wall so colours read like paint
const PAINT_COLORS = ['#e53935', '#fb8c00', '#fdd835', '#43a047', '#00acc1', '#1e88e5', '#5e35b1', '#d81b60', '#6d4c41', '#111111'];
const PAINT_DWELL_TIME = 0.6;             // seconds to hold still to "click"
const PAINT_DWELL_RADIUS = 30;            // px; moving beyond this re-arms the dwell
const PAINT_TRAVEL_SPEED = 1700;          // px/s; a flick faster than this = "moving away" -> lifts the pen
const PAINT_IDLE_CLEAR = 90;              // seconds with no hand -> auto-wipe

function ensurePaintLayer() {
  const w = Math.max(1, W | 0), h = Math.max(1, H | 0);
  if (!paintLayer || paintLayerW !== w || paintLayerH !== h) {
    paintLayer = document.createElement('canvas');
    paintLayer.width = w; paintLayer.height = h;
    paintLayerCtx = paintLayer.getContext('2d');
    paintLayerW = w; paintLayerH = h;
  }
}

function paintPaletteItems() {
  const items = PAINT_COLORS.map(c => ({ type: 'color', color: c }));
  items.push({ type: 'erase' });
  items.push({ type: 'size', size: 14 });
  items.push({ type: 'size', size: 28 });
  items.push({ type: 'size', size: 54 });
  items.push({ type: 'clear' });
  return items;
}

function paintDab(p, x, y, r, color, erase) {
  if (erase) {
    p.save();
    p.globalCompositeOperation = 'destination-out';
    p.fillStyle = '#000';
    p.beginPath(); p.arc(x, y, r, 0, Math.PI * 2); p.fill();
    p.restore();
    return;
  }
  const g = p.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, color);
  g.addColorStop(0.65, color);
  g.addColorStop(1, color + '00'); // soft edge (7-char hex -> transparent)
  p.fillStyle = g;
  p.beginPath(); p.arc(x, y, r, 0, Math.PI * 2); p.fill();
}

// A dwell "click" landed at (x,y): pick a palette item, or toggle the pen.
function paintClick(x, y, barH, items, iw) {
  if (y < barH) {
    const idx = Math.floor(x / iw);
    if (idx < 0 || idx >= items.length) return;
    const it = items[idx];
    if (it.type === 'color') { paintColor = it.color; paintErase = false; }
    else if (it.type === 'erase') { paintErase = true; }
    else if (it.type === 'size') { paintBrush = it.size; }
    else if (it.type === 'clear') { paintLayerCtx.clearRect(0, 0, W, H); }
  } else {
    paintPenDown = !paintPenDown; // start / stop drawing
    paintLast = null;
  }
}

function drawPaintPalette(now, items, barH, iw, hovered) {
  ctx.save();
  ctx.fillStyle = 'rgba(20,22,28,0.82)';
  ctx.fillRect(0, 0, W, barH);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < items.length; i++) {
    const it = items[i], x = i * iw, cx = x + iw / 2, cy = barH / 2;
    const selected = (it.type === 'color' && !paintErase && it.color === paintColor)
      || (it.type === 'erase' && paintErase)
      || (it.type === 'size' && it.size === paintBrush);
    if (selected) { ctx.fillStyle = 'rgba(255,255,255,0.16)'; ctx.fillRect(x + 2, 2, iw - 4, barH - 4); }
    if (hovered === i) { ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 2; ctx.strokeRect(x + 2, 2, iw - 4, barH - 4); }
    if (it.type === 'color') {
      ctx.fillStyle = it.color;
      ctx.beginPath(); ctx.arc(cx, cy, Math.min(barH * 0.32, iw * 0.34), 0, Math.PI * 2); ctx.fill();
    } else if (it.type === 'erase') {
      ctx.fillStyle = '#eaeaea'; ctx.font = `${Math.round(barH * 0.26)}px sans-serif`;
      ctx.fillText('Erase', cx, cy);
    } else if (it.type === 'size') {
      ctx.fillStyle = '#eaeaea';
      ctx.beginPath(); ctx.arc(cx, cy, Math.min(it.size * 0.42, barH * 0.34), 0, Math.PI * 2); ctx.fill();
    } else if (it.type === 'clear') {
      ctx.fillStyle = '#ff6b6b'; ctx.font = `${Math.round(barH * 0.26)}px sans-serif`;
      ctx.fillText('Clear', cx, cy);
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, barH); ctx.lineTo(W, barH); ctx.stroke();
  ctx.restore();
}

// Always-visible cursor: a brush ring in the current colour, filled when the
// pen is down, with a dwell-progress ring that fills as you hold still.
function drawPaintCursor(now, dwellProgress, live) {
  if (!paintCursor) return;
  const { x, y } = paintCursor;
  ctx.save();
  ctx.globalAlpha = live ? 1 : 0.4; // dim but still visible if tracking dropped
  const col = paintErase ? '#444' : paintColor;
  if (paintPenDown && !paintErase) {
    ctx.globalAlpha = (live ? 1 : 0.4) * 0.35;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, paintBrush, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = live ? 1 : 0.4;
  }
  ctx.lineWidth = 3;
  ctx.strokeStyle = col;
  ctx.beginPath(); ctx.arc(x, y, paintBrush, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
  if (dwellProgress > 0) {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.arc(x, y, paintBrush + 10, -Math.PI / 2, -Math.PI / 2 + dwellProgress * Math.PI * 2);
    ctx.stroke();
  }
  if (now - paintClickFlash < 0.25) { // confirm pulse
    ctx.globalAlpha = 1 - (now - paintClickFlash) / 0.25;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, paintBrush + 18, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.restore();
}

function drawPaintScene(now, dt) {
  ensurePaintLayer();
  const p = paintLayerCtx;
  const items = paintPaletteItems();
  const barH = Math.max(58, Math.min(100, H * 0.09));
  const iw = W / items.length;

  // auto-wipe after a long idle so it's fresh for the next person
  if (hand) paintLastHand = now;
  if (paintLastHand && now - paintLastHand > PAINT_IDLE_CLEAR) { p.clearRect(0, 0, W, H); paintLastHand = now; paintPenDown = false; }

  // --- hand speed: a fast flick means "moving away" (travelling), which lifts
  // the pen automatically so you don't have to dwell to stop drawing ---
  if (hand) {
    if (paintPrevPos) {
      const inst = Math.hypot(hand.x - paintPrevPos.x, hand.y - paintPrevPos.y) / Math.max(dt, 1e-3);
      paintSpeed += (inst - paintSpeed) * 0.5;
    } else paintSpeed = 0;
    paintPrevPos = { x: hand.x, y: hand.y };
    if (paintPenDown && paintSpeed > PAINT_TRAVEL_SPEED) { paintPenDown = false; paintLast = null; }
  } else {
    paintPrevPos = null; paintSpeed = 0;
  }
  const travelling = paintSpeed > PAINT_TRAVEL_SPEED;

  // --- dwell "click": hold the hand still to trigger ---
  let dwellProgress = 0;
  if (hand) {
    paintCursor = { x: hand.x, y: hand.y };
    if (!dwellAnchor || Math.hypot(hand.x - dwellAnchor.x, hand.y - dwellAnchor.y) > PAINT_DWELL_RADIUS) {
      dwellAnchor = { x: hand.x, y: hand.y }; dwellStart = now; dwellArmed = true;
    } else if (dwellArmed) {
      dwellProgress = Math.min(1, (now - dwellStart) / PAINT_DWELL_TIME);
      if (dwellProgress >= 1) {
        dwellArmed = false; // fire once; re-arms only after moving out of the radius
        paintClickFlash = now;
        paintClick(dwellAnchor.x, dwellAnchor.y, barH, items, iw);
      }
    }
  } else {
    dwellAnchor = null; dwellArmed = false;
  }

  const hovered = (hand && hand.y < barH) ? Math.floor(hand.x / iw) : null;

  // paint only while the pen is down, below the bar, moving, and not flicking away
  if (paintPenDown && !travelling && hand && hand.y >= barH) {
    const cur = { x: hand.x, y: hand.y };
    if (paintLast) {
      const d = Math.hypot(cur.x - paintLast.x, cur.y - paintLast.y);
      if (d > 0.5) {
        const spacing = Math.max(2, paintBrush * 0.3);
        const steps = Math.min(80, Math.max(1, Math.floor(d / spacing)));
        for (let i = 1; i <= steps; i++) {
          const t = i / steps;
          paintDab(p, paintLast.x + (cur.x - paintLast.x) * t, paintLast.y + (cur.y - paintLast.y) * t, paintBrush, paintColor, paintErase);
        }
      }
    } else {
      paintDab(p, cur.x, cur.y, paintBrush, paintColor, paintErase); // dot at pen-down
    }
    paintLast = cur;
  } else {
    paintLast = null;
  }

  ctx.fillStyle = PAINT_WALL_COLOR;
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(paintLayer, 0, 0);
  drawPaintPalette(now, items, barH, iw, (hovered !== null && hovered >= 0 && hovered < items.length) ? hovered : null);

  // status hint
  ctx.save();
  ctx.textAlign = 'center';
  ctx.font = `${Math.round(barH * 0.28)}px sans-serif`;
  ctx.fillStyle = paintPenDown ? 'rgba(30,120,40,0.9)' : 'rgba(60,60,60,0.75)';
  const hint = !paintCursor ? 'Move your hand to begin'
    : paintPenDown ? 'DRAWING - flick away fast (or hold still) to lift the pen'
      : 'Hold still to start drawing, or on a palette item to pick it';
  ctx.fillText(hint, W / 2, H - Math.max(16, barH * 0.35));
  ctx.restore();

  drawPaintCursor(now, dwellProgress, !!hand);
}

// ============================================================================
// Constellation drawer: a night sky of stars. Move your hand from star to
// star to connect them into glowing constellations; pause or lift your hand
// and the finished shape is committed and slowly fades among the stars.
// ============================================================================
let constInit = false;
let stars = [];
let activePath = [];      // star indices being connected right now
let doneConstellations = [];
let lastConnectAt = 0;
let constHue = 200;
const STAR_CONNECT_RADIUS = 60;
const CONST_IDLE_FINALIZE = 1.6;  // seconds hovering with no new star -> commit
const CONST_FADE_SECONDS = 16;

function initConstellation() {
  constInit = true;
  stars = [];
  const count = Math.max(50, Math.floor((W * H) / 17000));
  for (let i = 0; i < count; i++) {
    stars.push({ x: Math.random() * W, y: Math.random() * H, r: 0.8 + Math.random() * 2.2, tw: Math.random() * Math.PI * 2, twSpeed: 0.5 + Math.random() * 1.5 });
  }
}

function drawConstPath(points, hue, alpha) {
  if (points.length < 1) return;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.strokeStyle = `hsla(${hue},90%,66%,${0.85 * alpha})`;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.shadowColor = `hsl(${hue},90%,60%)`;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.stroke();
  ctx.shadowBlur = 0;
  for (const pt of points) {
    ctx.fillStyle = `hsla(${hue},90%,78%,${alpha})`;
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

function finalizeConstellation(now) {
  if (activePath.length >= 2) {
    doneConstellations.push({ points: activePath.map(i => ({ x: stars[i].x, y: stars[i].y })), born: now, hue: constHue });
  }
  activePath = [];
}

function drawConstellationScene(now, dt) {
  if (!constInit) initConstellation();

  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#05030f'); g.addColorStop(1, '#0b0922');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  for (const s of stars) {
    const tw = 0.55 + 0.45 * Math.sin(now * s.twSpeed + s.tw);
    ctx.fillStyle = `rgba(255,255,240,${tw})`;
    ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
  }

  doneConstellations = doneConstellations.filter(c => {
    const age = now - c.born;
    if (age > CONST_FADE_SECONDS) return false;
    const a = Math.min(1, 2 - 2 * age / CONST_FADE_SECONDS); // hold, then fade
    drawConstPath(c.points, c.hue, Math.max(0, a));
    return true;
  });

  if (hand) {
    let nearest = -1, nd = STAR_CONNECT_RADIUS;
    for (let i = 0; i < stars.length; i++) {
      const d = Math.hypot(stars[i].x - hand.x, stars[i].y - hand.y);
      if (d < nd) { nd = d; nearest = i; }
    }
    if (nearest >= 0) {
      const last = activePath[activePath.length - 1];
      if (last !== nearest && !activePath.includes(nearest)) {
        activePath.push(nearest);
        lastConnectAt = now;
        constHue = (constHue + 24) % 360;
      }
    }
    if (activePath.length >= 2 && now - lastConnectAt > CONST_IDLE_FINALIZE) {
      finalizeConstellation(now);
    }
  } else {
    finalizeConstellation(now); // hand lost - commit whatever's there
  }

  if (activePath.length) {
    const pts = activePath.map(i => ({ x: stars[i].x, y: stars[i].y }));
    drawConstPath(pts, constHue, 1);
    if (hand) {
      const s = stars[activePath[activePath.length - 1]];
      ctx.strokeStyle = `hsla(${constHue},90%,72%,0.4)`;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(hand.x, hand.y); ctx.stroke();
    }
  }
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
  } else if (currentScene === 'paint') {
    drawPaintScene(now, dt);
  } else if (currentScene === 'constellation') {
    drawConstellationScene(now, dt);
  } else {
    drawFishScene(now, dt);
  }

  if (introActive) {
    introActive = drawIntro(now);
  }

  drawHandMarker(now);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
