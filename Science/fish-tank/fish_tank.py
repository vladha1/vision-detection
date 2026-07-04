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
FOOD_COLOR = (255, 220, 60)
FISH_LENGTH = 90

WANDER_SPEED = 90
SEEK_SPEED = 200
SEEK_RADIUS = 320
SEEK_HOLD_SECONDS = 1.0
SEEK_ARRIVE_RADIUS = 90

FOOD_SPEED = 260
FOOD_ARRIVE_RADIUS = 40
FOOD_EAT_RADIUS = 26

EDGE_MARGIN = 80

CROP_PADDING_FRAC = 0.2
CROP_TARGET_SIZE = 640

LASER_HUE_RANGE = (35, 95)
LASER_SAT_MIN = 60
LASER_VAL_MIN = 210
LASER_MIN_AREA = 2
LASER_MAX_AREA = 500


class WallTracker(threading.Thread):
    """Runs camera capture on a background thread and exposes the latest
    hand-fingertip and green-laser-dot positions, both in projector
    coordinates (mapped through the calibration homography)."""

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
        self.laser_point = None
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

    def _detect_laser(self, crop_bgr):
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([LASER_HUE_RANGE[0], LASER_SAT_MIN, LASER_VAL_MIN])
        upper = np.array([LASER_HUE_RANGE[1], 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < LASER_MIN_AREA or area > LASER_MAX_AREA:
            return None
        m = cv2.moments(c)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    def run(self):
        hand_seen = False
        laser_seen = False
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    continue
                crop, off_x, off_y, scale = self._crop_and_scale(frame)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)
                hand_point = None
                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0].landmark[8]  # index fingertip
                    ch, cw = crop.shape[:2]
                    hand_point = self._to_projector(lm.x * cw, lm.y * ch, off_x, off_y, scale)

                laser_px = self._detect_laser(crop)
                laser_point = None
                if laser_px is not None:
                    laser_point = self._to_projector(laser_px[0], laser_px[1], off_x, off_y, scale)

                if hand_point is not None and not hand_seen:
                    print(f"[hand] detected -> projector {hand_point}")
                elif hand_point is None and hand_seen:
                    print("[hand] lost")
                hand_seen = hand_point is not None

                if laser_point is not None and not laser_seen:
                    print(f"[laser] detected -> projector {laser_point}")
                elif laser_point is None and laser_seen:
                    print("[laser] lost")
                laser_seen = laser_point is not None

                with self.lock:
                    self.hand_point = hand_point
                    self.laser_point = laser_point
            except Exception as exc:
                print(f"[tracker] error: {exc}")

    def get_hand(self):
        with self.lock:
            return self.hand_point

    def get_laser(self):
        with self.lock:
            return self.laser_point

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
        self.seek_until = 0.0

    def _steer_toward(self, target, max_speed, arrive_radius):
        to_target = target - self.pos
        dist = to_target.length()
        if dist > 1e-3:
            desired = to_target.normalize() * max_speed * min(1.0, dist / arrive_radius)
        else:
            desired = pygame.Vector2()
        return (desired - self.vel) * 6

    def update(self, dt, hand, food, now):
        steer = pygame.Vector2()
        max_speed = WANDER_SPEED
        target = None

        if food is not None:
            target = pygame.Vector2(food)
            max_speed = FOOD_SPEED
            self.state = "eat"
        else:
            if hand is not None and (self.pos - pygame.Vector2(hand)).length() < SEEK_RADIUS:
                self.state = "seek"
                self.seek_until = now + SEEK_HOLD_SECONDS
            if self.state == "seek" and now < self.seek_until and hand is not None:
                target = pygame.Vector2(hand)
                max_speed = SEEK_SPEED
            elif self.state != "eat":
                self.state = "wander"

        if target is not None:
            arrive_radius = FOOD_ARRIVE_RADIUS if self.state == "eat" else SEEK_ARRIVE_RADIUS
            steer += self._steer_toward(target, max_speed, arrive_radius)
        else:
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
    parser.add_argument("--debug", action="store_true", help="draw tracked hand/laser position and seek radius")
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

    fish_surface = build_fish_surface()
    fish = Fish((mon.width, mon.height))
    food = None

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
        laser = tracker.get_laser()
        if laser is not None:
            food = laser

        fish.update(dt, hand, food, now)

        if food is not None and (fish.pos - pygame.Vector2(food)).length() < FOOD_EAT_RADIUS:
            food = None

        screen.fill(WATER_COLOR)
        if food is not None:
            pygame.draw.circle(screen, FOOD_COLOR, (int(food[0]), int(food[1])), 8)
        fish.draw(screen, fish_surface)
        if args.debug:
            pygame.draw.circle(screen, (0, 200, 0), fish.pos, SEEK_RADIUS, 1)
            if hand is not None:
                pygame.draw.circle(screen, (255, 0, 255), (int(hand[0]), int(hand[1])), 10)
                pygame.draw.line(screen, (255, 0, 255), fish.pos, hand, 1)
            if laser is not None:
                pygame.draw.circle(screen, (0, 255, 255), (int(laser[0]), int(laser[1])), 6)
        pygame.display.flip()

    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
