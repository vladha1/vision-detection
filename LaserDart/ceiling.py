#!/usr/bin/env python3
"""
Ceiling target laser accuracy game.

Target: any fixed object on the ceiling (e.g. smoke detector).
Score:  how close the laser dot first appears to the target centre.

Anti-cheat rules (both enforced by state machine):
  - No drag-in  : laser must appear from absent, not slide in from off-target
  - No hold     : once a shot is registered the beam must disappear fully
                  before the next shot is accepted
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from detect import detect_laser
from logger import ShotLogger

CALIB_FILE = "ceiling_calibration.json"

# Scoring rings — (radius_multiplier, points, label)
# Distances are multiples of the configured target_radius
RINGS = [
    (0.5,  50, "Bullseye"),
    (1.0,  40, "On Target"),
    (2.0,  25, "Close"),
    (4.0,  10, "Near"),
    (8.0,   5, "Far"),
]

# ── Calibration ───────────────────────────────────────────────────────────────

_cal_click: list = []


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _cal_click.clear()
        _cal_click.append((x, y))
        print(f"  Target centre set: ({x}, {y})")


def calibrate(camera_index: int = 0, headless_target: str = None, headless_radius: int = None):
    """
    Two modes:
      GUI      — opens camera feed, click the target centre, +/- to size rings, s to save.
      Headless — pass --target x,y and --radius N; saves without opening a window.
                 Use this over SSH where no display is available.
    """
    if headless_target:
        try:
            x, y = [int(v.strip()) for v in headless_target.split(",")]
        except ValueError:
            sys.exit("[ERROR] --target must be x,y  e.g. --target 320,240")
        radius = headless_radius or 30
        with open(CALIB_FILE, "w") as f:
            json.dump({"target": [x, y], "radius": radius}, f, indent=2)
        print(f"[CALIBRATE] Saved headless → {CALIB_FILE}")
        print(f"  target=({x},{y})  radius={radius}px")
        print("  Tip: grab target pixel coords from a snapshot:")
        print("  python ceiling.py snapshot  →  saves ceiling_snapshot.jpg")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")

    print("[CALIBRATE] Click the centre of your target (smoke detector)")
    print("  +/-  adjust ring radius   |   s  save   |   q  quit")

    cv2.namedWindow("Calibrate Ceiling")
    cv2.setMouseCallback("Calibrate Ceiling", _on_click)

    radius = 30

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()

        if _cal_click:
            cx, cy = _cal_click[0]
            for mult, pts, label in RINGS:
                r = int(radius * mult)
                cv2.circle(display, (cx, cy), r, (0, 180, 255), 1)
                cv2.putText(display, f"{pts}", (cx + r + 3, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(display, f"Centre ({cx},{cy})  radius={radius}px",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)

        cv2.putText(display, "Click target | +/- ring size | s=save",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Calibrate Ceiling", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('+'), ord('=')):
            radius += 5
        elif key == ord('-'):
            radius = max(5, radius - 5)
        elif key == ord('s') and _cal_click:
            with open(CALIB_FILE, "w") as f:
                json.dump({"target": list(_cal_click[0]), "radius": radius}, f, indent=2)
            print(f"[CALIBRATE] Saved → {CALIB_FILE}")
            break
        elif key == ord('q'):
            print("[CALIBRATE] Aborted")
            break

    cap.release()
    cv2.destroyAllWindows()


def snapshot(camera_index: int = 0, out: str = "ceiling_snapshot.jpg"):
    """Grab one frame and save it as a JPEG — open it to find target pixel coords."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")
    for _ in range(10):          # let auto-exposure settle
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("[ERROR] Could not read frame")
    cv2.imwrite(out, frame)
    print(f"[SNAPSHOT] Saved → {out}  ({frame.shape[1]}×{frame.shape[0]})")
    print("  Open it, find the smoke detector centre pixel, then run:")
    print("  python ceiling.py calibrate --target x,y --radius 30")


def _load_calibration() -> tuple[tuple[int, int], int]:
    if not os.path.exists(CALIB_FILE):
        sys.exit("[ERROR] No ceiling calibration found. Run:  python ceiling.py calibrate")
    with open(CALIB_FILE) as f:
        data = json.load(f)
    return tuple(data["target"]), data["radius"]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_hit(
    dot: tuple[int, int],
    target: tuple[int, int],
    radius: int,
) -> tuple[int, str, float]:
    dist = math.hypot(dot[0] - target[0], dot[1] - target[1])
    for mult, pts, label in RINGS:
        if dist <= radius * mult:
            return pts, label, round(dist, 1)
    return 0, "Miss", round(dist, 1)


# ── Play loop ─────────────────────────────────────────────────────────────────

def play(
    camera_index: int = 0,
    color: str = "green",
    show: bool = True,
    web: bool = False,
    port: int = 5001,
    log_dir: str = "dart_logs",
):
    target, radius = _load_calibration()
    logger = ShotLogger(log_dir=log_dir)

    if web:
        from dashboard import start_dashboard
        start_dashboard(logger, port=port)
        print(f"[INFO] Dashboard → http://0.0.0.0:{port}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")

    print(f"[PLAY] Target {target}, ring radius={radius}px, laser={color}")
    print("[PLAY] Anti-cheat: snap-on only — no drag-in, no hold. Press q to quit.")

    # ── Anti-cheat state machine ──────────────────────────────────────────────
    #
    #   IDLE          laser absent — ready for next shot
    #   SCORED        shot registered this appearance — waiting for laser to leave
    #
    # Transition IDLE → SCORED : laser appears from absent → score immediately
    # Transition SCORED → IDLE : laser disappears → reset
    # Staying SCORED           : laser still visible after scoring → ignore (no hold)
    # Drag prevention          : because we score on the FIRST frame the dot appears
    #                            (not where it ends up), sliding the beam onto the
    #                            target scores the position where it first became
    #                            visible, not the target — natural penalty for dragging.

    laser_was_visible = False
    shot_registered   = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        dot           = detect_laser(frame, color=color)
        laser_visible = dot is not None

        if not laser_visible:
            laser_was_visible = False
            shot_registered   = False

        elif not laser_was_visible and not shot_registered:
            # Fresh appearance — valid shot
            pts, label, dist = score_hit(dot, target, radius)
            event = {
                "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                "pixel":       list(dot),
                "distance_px": dist,
                "score":       pts,
                "label":       label,
            }
            logger.log(event)
            print(f"[SHOT]  {label:<12}  {pts:>3} pts  "
                  f"dist={dist:.0f}px  (running: {logger.running_score})")
            shot_registered = True

        # laser visible + already scored → held/dragged → silently ignored

        laser_was_visible = laser_visible

        if show:
            display = frame.copy()

            tx, ty = int(target[0]), int(target[1])
            for mult, pts, _ in RINGS:
                cv2.circle(display, (tx, ty), int(radius * mult), (0, 180, 255), 1)
            cv2.circle(display, (tx, ty), 5, (0, 0, 255), -1)

            if dot:
                dot_colour = (0, 80, 80) if shot_registered else (0, 255, 255)
                cv2.circle(display, dot, 10, dot_colour, 2)
                cv2.circle(display, dot,  2, (255, 255, 255), -1)
                if shot_registered:
                    cv2.putText(display, "RELEASE TO RESET", (dot[0] + 12, dot[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 80), 1)

            cv2.putText(display,
                        f"Shots: {logger.total_shots}   Score: {logger.running_score}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
            try:
                cv2.imshow("Ceiling Target", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                # No display available — fall back to headless silently
                show = False
        else:
            # Headless: only exit via Ctrl+C
            pass

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ceiling target laser accuracy game")
    sub = ap.add_subparsers(dest="mode", required=True)

    cal = sub.add_parser("calibrate", help="Set target centre and ring size")
    cal.add_argument("-c", "--camera",   type=int, default=0)
    cal.add_argument("--target",         default=None,
                     help="Headless: pixel coords of target centre, e.g. 320,240")
    cal.add_argument("--radius",         type=int, default=30,
                     help="Ring base radius in pixels (default: 30)")

    snap = sub.add_parser("snapshot", help="Save one camera frame to find target coords")
    snap.add_argument("-c", "--camera",  type=int, default=0)
    snap.add_argument("-o", "--out",     default="ceiling_snapshot.jpg")

    pl = sub.add_parser("play", help="Run the game")
    pl.add_argument("-c", "--camera",  type=int,  default=0)
    pl.add_argument("--color",         default="green",
                    choices=["red", "green", "bright"])
    pl.add_argument("--no-show",       action="store_true")
    pl.add_argument("--web",           action="store_true")
    pl.add_argument("--port",          type=int,  default=5001)
    pl.add_argument("--log",           default="dart_logs")

    args = ap.parse_args()
    if args.mode == "calibrate":
        calibrate(args.camera, headless_target=args.target, headless_radius=args.radius)
    elif args.mode == "snapshot":
        snapshot(args.camera, out=args.out)
    else:
        play(
            camera_index=args.camera,
            color=args.color,
            show=not args.no_show,
            web=args.web,
            port=args.port,
            log_dir=args.log,
        )
