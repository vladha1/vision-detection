#!/usr/bin/env python3
"""
Laser dart detector.

Usage:
  python detect.py calibrate          # one-time board corner setup
  python detect.py play               # detect shots and score them
  python detect.py play --web         # + live score dashboard
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from score import dart_score
from logger import ShotLogger

CALIB_FILE = "calibration.json"
BOARD_PX   = 500   # side length of canonical board square (pixels)

# ── Calibration ───────────────────────────────────────────────────────────────

_clicks: list[tuple[int, int]] = []


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicks) < 4:
        _clicks.append((x, y))
        print(f"  Point {len(_clicks)}: ({x}, {y})")


def calibrate(camera_index: int = 0):
    """
    Show live camera feed; user clicks the 4 corners of the projected board
    (top-left → top-right → bottom-right → bottom-left), then presses 's'.
    Saves a homography matrix to calibration.json.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")

    print("[CALIBRATE] Click 4 corners of the projected dartboard:")
    print("  Order: top-left, top-right, bottom-right, bottom-left")
    print("  Keys:  r = reset  |  s = save  |  q = quit")

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", _on_click)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        for i, (x, y) in enumerate(_clicks):
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(_clicks) == 4:
            pts = np.array(_clicks, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        cv2.putText(display, f"Corners: {len(_clicks)}/4  |  s=save  r=reset",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
        cv2.imshow("Calibrate", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            _clicks.clear()
            print("[CALIBRATE] Reset — click the 4 corners again")
        elif key == ord('s'):
            if len(_clicks) == 4:
                _save_calibration(_clicks)
                break
            else:
                print(f"[CALIBRATE] Need 4 points, have {len(_clicks)}")
        elif key == ord('q'):
            print("[CALIBRATE] Aborted")
            break

    cap.release()
    cv2.destroyAllWindows()


def _save_calibration(corners: list[tuple[int, int]]):
    src = np.array(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [BOARD_PX, 0], [BOARD_PX, BOARD_PX], [0, BOARD_PX]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst)
    data = {
        "corners":     corners,
        "board_px":    BOARD_PX,
        "H":           H.tolist(),
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[CALIBRATE] Saved → {CALIB_FILE}")


def _load_calibration() -> tuple[np.ndarray, int]:
    if not os.path.exists(CALIB_FILE):
        sys.exit("[ERROR] No calibration file found. Run:  python detect.py calibrate")
    with open(CALIB_FILE) as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float64), data["board_px"]


# ── Laser detection ───────────────────────────────────────────────────────────

def detect_laser(
    frame: np.ndarray,
    color: str = "red",
    min_area: int = 4,
    max_area: int = 500,
) -> tuple[int, int] | None:
    """
    Returns (cx, cy) pixel of the laser dot centroid, or None if not found.

    color choices:
      "red"    — red laser (HSV wraps at 0/180)
      "green"  — green laser (~532 nm)
      "bright" — any very-bright spot regardless of hue (daylight fallback)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == "red":
        m1 = cv2.inRange(hsv, (0,   120, 200), (10,  255, 255))
        m2 = cv2.inRange(hsv, (160, 120, 200), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif color == "green":
        mask = cv2.inRange(hsv, (40, 120, 200), (80, 255, 255))
    else:
        # brightness only — works in dim rooms, colour-agnostic
        _, mask = cv2.threshold(hsv[:, :, 2], 240, 255, cv2.THRESH_BINARY)

    # Remove single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area and area > best_area:
            best, best_area = c, area

    if best is None:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


# ── Play loop ─────────────────────────────────────────────────────────────────

def play(
    camera_index: int = 0,
    color: str = "red",
    cooldown: float = 1.5,
    show: bool = True,
    web: bool = False,
    port: int = 5001,
    log_dir: str = "dart_logs",
):
    """
    Main detection loop.  A shot is registered when a laser dot is seen and
    at least `cooldown` seconds have passed since the last registered shot
    (prevents the same dart triggering multiple times).
    """
    H, board_px = _load_calibration()
    logger = ShotLogger(log_dir=log_dir)

    if web:
        from dashboard import start_dashboard
        start_dashboard(logger, port=port)
        print(f"[INFO] Dashboard → http://0.0.0.0:{port}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")

    print(f"[PLAY] Detecting {color} laser.  Cooldown: {cooldown}s.  Press q to quit.")

    last_shot_at = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        dot = detect_laser(frame, color=color)

        now = time.time()
        if dot and (now - last_shot_at) >= cooldown:
            # Map dot pixel → canonical board space via perspective transform
            pt = np.array([[list(dot)]], dtype=np.float64)
            warped = cv2.perspectiveTransform(pt, H)[0][0]

            # Normalise to [-1, 1] with board centre at origin
            nx = (warped[0] / board_px - 0.5) * 2
            ny = (warped[1] / board_px - 0.5) * 2

            score, label = dart_score(nx, ny)
            event = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pixel":    list(dot),
                "board_xy": [round(nx, 4), round(ny, 4)],
                "score":    score,
                "label":    label,
            }
            logger.log(event)
            print(f"[SHOT]  {label:<12}  {score:>3} pts  "
                  f"(running: {logger.running_score})")
            last_shot_at = now

        if show:
            display = frame.copy()
            if dot:
                cv2.circle(display, dot, 10, (0, 255, 255), 2)
                cv2.circle(display, dot,  2, (0,   0, 255), -1)
            cv2.putText(
                display,
                f"Shots: {logger.total_shots}  Score: {logger.running_score}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2,
            )
            cv2.imshow("Laser Dart", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Laser dart detection system")
    sub = ap.add_subparsers(dest="mode", required=True)

    # calibrate
    cal = sub.add_parser("calibrate", help="One-time board corner calibration")
    cal.add_argument("-c", "--camera", type=int, default=0,
                     help="Camera device index (default: 0)")

    # play
    pl = sub.add_parser("play", help="Detect laser shots and score them")
    pl.add_argument("-c", "--camera",   type=int,   default=0)
    pl.add_argument("--color",          default="red",
                    choices=["red", "green", "bright"],
                    help="Laser colour to track (default: red)")
    pl.add_argument("--cooldown",       type=float, default=1.5,
                    help="Minimum seconds between registered shots (default: 1.5)")
    pl.add_argument("--no-show",        action="store_true",
                    help="Disable the live preview window (headless)")
    pl.add_argument("--web",            action="store_true",
                    help="Start the score dashboard web server")
    pl.add_argument("--port",           type=int,   default=5001)
    pl.add_argument("--log",            default="dart_logs",
                    help="Directory for shot log files (default: dart_logs)")

    args = ap.parse_args()

    if args.mode == "calibrate":
        calibrate(args.camera)
    else:
        play(
            camera_index=args.camera,
            color=args.color,
            cooldown=args.cooldown,
            show=not args.no_show,
            web=args.web,
            port=args.port,
            log_dir=args.log,
        )
