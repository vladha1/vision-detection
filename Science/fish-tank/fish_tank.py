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
FISH_COLOR = (255, 140, 40)
FISH_LENGTH = 90

WANDER_SPEED = 90
FLEE_SPEED = 280
FLEE_RADIUS = 260
FLEE_HOLD_SECONDS = 1.5
EDGE_MARGIN = 80


CROP_PADDING_FRAC = 0.2
CROP_TARGET_SIZE = 640


class HandTracker(threading.Thread):
    """Runs camera capture + hand detection on a background thread and exposes
    the latest fingertip position in projector coordinates."""

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
        self.point = None
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

    def run(self):
        was_seen = False
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    continue
                crop, off_x, off_y, scale = self._crop_and_scale(frame)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)
                point = None
                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0].landmark[8]  # index fingertip
                    ch, cw = crop.shape[:2]
                    full_x = (lm.x * cw) / scale + off_x
                    full_y = (lm.y * ch) / scale + off_y
                    px = np.array([[[full_x, full_y]]], dtype=np.float32)
                    proj = cv2.perspectiveTransform(px, self.homography)
                    point = (float(proj[0, 0, 0]), float(proj[0, 0, 1]))
                if point is not None and not was_seen:
                    print(f"[hand] detected -> projector {point}")
                elif point is None and was_seen:
                    print("[hand] lost")
                was_seen = point is not None
                with self.lock:
                    self.point = point
            except Exception as exc:
                print(f"[hand] tracker error: {exc}")

    def get_point(self):
        with self.lock:
            return self.point

    def stop(self):
        self.running = False
        self.cap.release()


def build_fish_surface():
    w, h = FISH_LENGTH, int(FISH_LENGTH * 0.6)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    body = [(w * 0.15, h * 0.5), (w * 0.55, h * 0.1), (w * 0.92, h * 0.3),
            (w * 0.92, h * 0.7), (w * 0.55, h * 0.9)]
    tail = [(w * 0.15, h * 0.5), (0, h * 0.1), (0, h * 0.9)]
    pygame.draw.polygon(surf, FISH_COLOR, body)
    pygame.draw.polygon(surf, FISH_COLOR, tail)
    pygame.draw.circle(surf, (20, 20, 20), (int(w * 0.75), int(h * 0.35)), 4)
    return surf


class Fish:
    def __init__(self, bounds):
        self.bounds = bounds
        self.pos = pygame.Vector2(bounds[0] / 2, bounds[1] / 2)
        self.vel = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        self.wander_angle = 0.0
        self.state = "wander"
        self.flee_until = 0.0

    def update(self, dt, threat, now):
        steer = pygame.Vector2()

        if threat is not None:
            threat_v = pygame.Vector2(threat)
            away = self.pos - threat_v
            dist = away.length()
            if dist < FLEE_RADIUS:
                if dist > 1e-3:
                    away.scale_to_length(1)
                else:
                    away = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
                steer += away * (1 - dist / FLEE_RADIUS) * 900
                self.state = "flee"
                self.flee_until = now + FLEE_HOLD_SECONDS

        if self.state == "flee" and now > self.flee_until:
            self.state = "wander"

        if self.state == "wander":
            self.wander_angle += random.uniform(-0.5, 0.5)
            heading = self.vel.normalize() if self.vel.length() > 0 else pygame.Vector2(1, 0)
            wander_dir = heading.rotate(math.degrees(self.wander_angle))
            steer += wander_dir * 60

        w, h = self.bounds
        for axis, size in ((0, w), (1, h)):
            if self.pos[axis] < EDGE_MARGIN:
                steer[axis] += (EDGE_MARGIN - self.pos[axis]) * 4
            elif self.pos[axis] > size - EDGE_MARGIN:
                steer[axis] -= (self.pos[axis] - (size - EDGE_MARGIN)) * 4

        max_speed = FLEE_SPEED if self.state == "flee" else WANDER_SPEED
        self.vel += steer * dt
        if self.vel.length() > max_speed:
            self.vel.scale_to_length(max_speed)

        self.pos += self.vel * dt

    def draw(self, screen, fish_surface):
        angle = -math.degrees(math.atan2(self.vel.y, self.vel.x))
        rotated = pygame.transform.rotate(fish_surface, angle)
        rect = rotated.get_rect(center=self.pos)
        screen.blit(rotated, rect)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--debug", action="store_true", help="draw tracked hand position and flee radius")
    args = parser.parse_args()

    with open(args.calibration) as f:
        calib = json.load(f)

    monitors = get_monitors()
    mon = monitors[calib["projector_monitor"]]
    homography = np.array(calib["homography"], dtype=np.float32)

    camera_points = calib.get("camera_points")
    if camera_points is None:
        print("[warn] calibration.json has no camera_points - re-run calibrate.py to enable cropped/zoomed hand detection")
    tracker = HandTracker(calib["camera_index"], homography, camera_points)
    tracker.start()

    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon.x},{mon.y}"
    pygame.init()
    screen = pygame.display.set_mode((mon.width, mon.height), pygame.NOFRAME)
    pygame.display.set_caption("fish-tank")
    clock = pygame.time.Clock()

    fish_surface = build_fish_surface()
    fish = Fish((mon.width, mon.height))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        threat = tracker.get_point()
        fish.update(dt, threat, now)

        screen.fill(WATER_COLOR)
        fish.draw(screen, fish_surface)
        if args.debug:
            pygame.draw.circle(screen, (0, 200, 0), fish.pos, FLEE_RADIUS, 1)
            if threat is not None:
                pygame.draw.circle(screen, (255, 0, 255), (int(threat[0]), int(threat[1])), 10)
                pygame.draw.line(screen, (255, 0, 255), fish.pos, threat, 1)
        pygame.display.flip()

    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
