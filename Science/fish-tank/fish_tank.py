import argparse
import json
import math
import os
import random
import threading

import cv2
import mediapipe as mp
import numpy as np
import pygame
from screeninfo import get_monitors

WATER_COLOR = (10, 40, 70)
FISH_LENGTH = 90
NUM_FISH = 10
FISH_COLORS = [
    (255, 140, 40),   # orange
    (255, 70, 70),    # red
    (90, 170, 255),   # blue
    (255, 215, 0),    # gold
    (190, 90, 255),   # purple
    (90, 255, 150),   # mint
    (255, 110, 180),  # pink
    (120, 220, 255),  # light blue
    (245, 245, 245),  # white
    (170, 255, 60),   # lime
]

WANDER_SPEED = 90
SEEK_SPEED = 200
SEEK_RADIUS = 320
REACT_HOLD_SECONDS = 1.0
SEEK_ARRIVE_RADIUS = 90
FLEE_SPEED = 260
FLEE_FORCE = 900

EDGE_MARGIN = 80
PLAYABLE_HEIGHT_FRAC = 0.8

CROP_PADDING_FRAC = 0.2
CROP_TARGET_SIZE = 640

HAND_CONFIRM_FRAMES = 4
HAND_JUMP_THRESHOLD = 120


class WallTracker(threading.Thread):
    """Runs camera capture on a background thread and exposes the latest
    hand-fingertip position in projector coordinates (mapped through the
    calibration homography)."""

    def __init__(self, camera_index, homography, camera_points=None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self.homography = homography
        self.crop_box = self._compute_crop_box(camera_points) if camera_points else None
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4
        )
        self.lock = threading.Lock()
        self.hand_point = None
        self.running = True

    @staticmethod
    def _compute_crop_box(camera_points):
        pts = np.array(camera_points, dtype=np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_x = (x1 - x0) * CROP_PADDING_FRAC
        pad_y = (y1 - y0) * CROP_PADDING_FRAC
        return (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)

    def _crop_and_scale(self, frame):
        if self.crop_box is None:
            return frame, 0, 0, 1.0
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.crop_box
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(w, int(x1))
        y1 = min(h, int(y1))
        crop = frame[y0:y1, x0:x1]
        scale = CROP_TARGET_SIZE / max(crop.shape[:2])
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
        return crop, x0, y0, scale

    def _to_projector(self, x, y, off_x, off_y, scale):
        full_x = x / scale + off_x
        full_y = y / scale + off_y
        px = np.array([[[full_x, full_y]]], dtype=np.float32)
        proj = cv2.perspectiveTransform(px, self.homography)
        return (float(proj[0, 0, 0]), float(proj[0, 0, 1]))

    def run(self):
        hand_seen = False
        last_raw = None
        consecutive = 0
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    continue
                crop, off_x, off_y, scale = self._crop_and_scale(frame)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)
                raw_point = None
                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0].landmark[8]  # index fingertip
                    ch, cw = crop.shape[:2]
                    raw_point = self._to_projector(lm.x * cw, lm.y * ch, off_x, off_y, scale)

                if raw_point is None:
                    consecutive = 0
                    last_raw = None
                else:
                    if last_raw is not None and math.hypot(
                        raw_point[0] - last_raw[0], raw_point[1] - last_raw[1]
                    ) <= HAND_JUMP_THRESHOLD:
                        consecutive += 1
                    else:
                        consecutive = 1
                    last_raw = raw_point

                # a real hand stays roughly in place across several frames; a
                # one-off false positive (e.g. briefly mistaking a fish sprite
                # for a hand) won't reach this streak and gets filtered out.
                hand_point = raw_point if consecutive >= HAND_CONFIRM_FRAMES else None

                if hand_point is not None and not hand_seen:
                    print(f"[hand] detected -> projector {hand_point}")
                elif hand_point is None and hand_seen:
                    print("[hand] lost")
                hand_seen = hand_point is not None

                with self.lock:
                    self.hand_point = hand_point
            except Exception as exc:
                print(f"[tracker] error: {exc}")

    def get_hand(self):
        with self.lock:
            return self.hand_point

    def stop(self):
        self.running = False
        self.cap.release()


def build_fish_surface(color):
    w, h = FISH_LENGTH, int(FISH_LENGTH * 0.6)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    body = [(w * 0.15, h * 0.5), (w * 0.55, h * 0.1), (w * 0.92, h * 0.3),
            (w * 0.92, h * 0.7), (w * 0.55, h * 0.9)]
    tail = [(w * 0.15, h * 0.5), (0, h * 0.1), (0, h * 0.9)]
    pygame.draw.polygon(surf, color, body)
    pygame.draw.polygon(surf, color, tail)
    pygame.draw.circle(surf, (20, 20, 20), (int(w * 0.75), int(h * 0.35)), 4)
    return surf


class Fish:
    def __init__(self, bounds, color, temperament):
        self.bounds = bounds
        self.temperament = temperament  # "seek" (curious) or "flee" (scared)
        self.surface = build_fish_surface(color)
        margin = EDGE_MARGIN
        self.pos = pygame.Vector2(
            random.uniform(margin, bounds[0] - margin), random.uniform(margin, bounds[1] - margin)
        )
        self.vel = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        self.wander_angle = 0.0
        self.state = "wander"
        self.react_until = 0.0

    def _steer_toward(self, target, max_speed, arrive_radius):
        to_target = target - self.pos
        dist = to_target.length()
        if dist > 1e-3:
            desired = to_target.normalize() * max_speed * min(1.0, dist / arrive_radius)
        else:
            desired = pygame.Vector2()
        return (desired - self.vel) * 6

    def _flee_from(self, threat):
        away = self.pos - threat
        dist = away.length()
        if dist > 1e-3:
            away_dir = away / dist
        else:
            away_dir = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        return away_dir * FLEE_FORCE * (1 - min(dist, SEEK_RADIUS) / SEEK_RADIUS)

    def _wander_force(self):
        self.wander_angle += random.uniform(-0.5, 0.5)
        heading = self.vel.normalize() if self.vel.length() > 0 else pygame.Vector2(1, 0)
        return heading.rotate(math.degrees(self.wander_angle)) * 60

    def update(self, dt, hand, now):
        steer = pygame.Vector2()
        max_speed = WANDER_SPEED

        if hand is not None and (self.pos - pygame.Vector2(hand)).length() < SEEK_RADIUS:
            self.state = self.temperament
            self.react_until = now + REACT_HOLD_SECONDS

        if self.state == "seek" and now < self.react_until and hand is not None:
            max_speed = SEEK_SPEED
            steer += self._steer_toward(pygame.Vector2(hand), max_speed, SEEK_ARRIVE_RADIUS)
        elif self.state == "flee" and now < self.react_until and hand is not None:
            max_speed = FLEE_SPEED
            steer += self._flee_from(pygame.Vector2(hand))
        else:
            self.state = "wander"
            steer += self._wander_force()

        w, h = self.bounds
        for axis, size in ((0, w), (1, h)):
            if self.pos[axis] < EDGE_MARGIN:
                steer[axis] += (EDGE_MARGIN - self.pos[axis]) * 4
            elif self.pos[axis] > size - EDGE_MARGIN:
                steer[axis] -= (self.pos[axis] - (size - EDGE_MARGIN)) * 4

        self.vel += steer * dt
        if self.vel.length() > max_speed:
            self.vel.scale_to_length(max_speed)

        self.pos += self.vel * dt

    def draw(self, screen):
        angle = -math.degrees(math.atan2(self.vel.y, self.vel.x))
        rotated = pygame.transform.rotate(self.surface, angle)
        rect = rotated.get_rect(center=self.pos)
        screen.blit(rotated, rect)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--debug", action="store_true", help="draw tracked hand position and playable boundary")
    args = parser.parse_args()

    with open(args.calibration) as f:
        calib = json.load(f)

    monitors = get_monitors()
    mon = monitors[calib["projector_monitor"]]
    homography = np.array(calib["homography"], dtype=np.float32)

    camera_points = calib.get("camera_points")
    if camera_points is None:
        print("[warn] calibration.json has no camera_points - re-run calibrate.py to enable cropped/zoomed detection")
    tracker = WallTracker(calib["camera_index"], homography, camera_points)
    tracker.start()

    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon.x},{mon.y}"
    pygame.init()
    screen = pygame.display.set_mode((mon.width, mon.height), pygame.NOFRAME)
    pygame.display.set_caption("fish-tank")
    clock = pygame.time.Clock()

    playable_height = int(mon.height * PLAYABLE_HEIGHT_FRAC)
    bounds = (mon.width, playable_height)

    temperaments = ["seek"] * (NUM_FISH // 2) + ["flee"] * (NUM_FISH - NUM_FISH // 2)
    random.shuffle(temperaments)
    fishes = [
        Fish(bounds, FISH_COLORS[i % len(FISH_COLORS)], temperaments[i])
        for i in range(NUM_FISH)
    ]

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        hand = tracker.get_hand()
        if hand is not None and hand[1] > playable_height:
            hand = None

        for fish in fishes:
            fish.update(dt, hand, now)

        screen.fill(WATER_COLOR)
        for fish in fishes:
            fish.draw(screen)
        if args.debug:
            pygame.draw.line(screen, (120, 120, 120), (0, playable_height), (mon.width, playable_height), 1)
            if hand is not None:
                pygame.draw.circle(screen, (255, 0, 255), (int(hand[0]), int(hand[1])), 10)
        pygame.display.flip()

    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
