#!/usr/bin/env python3
"""
Real-time object, person, and gesture detection.
Logs detections with timestamps — no video stream.
"""
import argparse
import time
import sys
import cv2
import mediapipe as mp
from ultralytics import YOLO
from logger import DetectionLogger

# ── Gesture classifier ────────────────────────────────────────────────────────

def classify_gesture(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    extended = []
    is_right = handedness.classification[0].label == "Right"
    extended.append(lm[4].x < lm[3].x if is_right else lm[4].x > lm[3].x)
    for tip, pip in zip(tips[1:], pips[1:]):
        extended.append(lm[tip].y < lm[pip].y)

    thumb, index, middle, ring, pinky = extended

    if all(extended):                                           return "Open Hand"
    if not any(extended):                                       return "Fist"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up" if lm[4].y < lm[3].y else "Thumbs Down"
    if index and middle and not ring and not pinky and not thumb: return "Peace"
    if index and not middle and not ring and not pinky:         return "Pointing"
    if index and pinky and not middle and not ring:             return "Rock On"
    if thumb and pinky and not index and not middle and not ring: return "Call Me"
    if thumb and index and not middle and not ring and not pinky: return "Gun / L"
    return "Custom"


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(camera_index=0, width=320, height=240,
        yolo_conf=0.45, show_pose=True, show_hands=True,
        web=False, port=5000, log_file="detections.jsonl"):

    print("[INFO] Loading YOLOv8n …")
    yolo = YOLO("yolov8n.pt")

    print("[INFO] Loading MediaPipe …")
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                         model_complexity=0)
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           model_complexity=0)

    logger = DetectionLogger(log_file=log_file)

    if web:
        from dashboard import start_dashboard
        start_dashboard(logger, port=port)
        print(f"[INFO] Dashboard at http://0.0.0.0:{port}")

    print(f"[INFO] Opening camera {camera_index} …")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")

    print("[INFO] Running — Ctrl+C to quit")

    last_snapshot = None   # tracks previous detection state for change detection

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        event = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "objects": [], "persons": 0, "gestures": []}

        # ── YOLO ──────────────────────────────────────────────────────────
        results = yolo(frame, conf=yolo_conf, verbose=False)[0]
        for box in results.boxes:
            label = yolo.names[int(box.cls[0])]
            conf  = round(float(box.conf[0]), 2)
            if label == "person":
                event["persons"] += 1
            else:
                event["objects"].append({"label": label, "conf": conf})

        # ── Pose ──────────────────────────────────────────────────────────
        if show_pose:
            pose_res = pose.process(rgb)
            if pose_res.pose_landmarks and event["persons"] == 0:
                event["persons"] = 1   # pose detected even if YOLO missed

        # ── Hands / Gesture ───────────────────────────────────────────────
        if show_hands:
            hand_res = hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                for hand_lm, handedness in zip(hand_res.multi_hand_landmarks,
                                               hand_res.multi_handedness):
                    g = classify_gesture(hand_lm, handedness)
                    side = handedness.classification[0].label
                    event["gestures"].append({"hand": side, "gesture": g})

        # Only log when something is detected AND it changed from last frame
        snapshot = (
            event["persons"],
            tuple(sorted(o["label"] for o in event["objects"])),
            tuple(sorted(f"{g['hand']}:{g['gesture']}" for g in event["gestures"])),
        )
        has_detections = event["objects"] or event["persons"] or event["gestures"]
        if has_detections and snapshot != last_snapshot:
            logger.log(event)
        last_snapshot = snapshot if has_detections else None

    cap.release()
    pose.close()
    hands.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera",  type=int,   default=0)
    ap.add_argument("-W", "--width",   type=int,   default=320)
    ap.add_argument("-H", "--height",  type=int,   default=240)
    ap.add_argument("--conf",          type=float, default=0.45)
    ap.add_argument("--no-pose",       action="store_true")
    ap.add_argument("--no-hands",      action="store_true")
    ap.add_argument("--web",           action="store_true")
    ap.add_argument("--port",          type=int,   default=5000)
    ap.add_argument("--log",           default="detections.jsonl")
    args = ap.parse_args()

    run(camera_index=args.camera, width=args.width, height=args.height,
        yolo_conf=args.conf, show_pose=not args.no_pose,
        show_hands=not args.no_hands, web=args.web,
        port=args.port, log_file=args.log)
