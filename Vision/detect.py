#!/usr/bin/env python3
"""
Real-time object, person, and gesture detection.
Optimized for CPU-only hardware (Mac Mini 2011).
Uses MediaPipe (pose + hands) + YOLOv8n (objects).
"""
import argparse
import time
import sys
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ── Gesture classifier ────────────────────────────────────────────────────────

def classify_gesture(hand_landmarks, handedness):
    """Return a gesture label from MediaPipe hand landmarks."""
    lm = hand_landmarks.landmark

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    extended = []
    is_right = handedness.classification[0].label == "Right"
    if is_right:
        extended.append(lm[4].x < lm[3].x)
    else:
        extended.append(lm[4].x > lm[3].x)
    for tip, pip in zip(tips[1:], pips[1:]):
        extended.append(lm[tip].y < lm[pip].y)

    thumb, index, middle, ring, pinky = extended

    if all(extended):
        return "Open Hand"
    if not any(extended):
        return "Fist"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up" if lm[4].y < lm[3].y else "Thumbs Down"
    if index and middle and not ring and not pinky and not thumb:
        return "Peace / V"
    if index and not middle and not ring and not pinky:
        return "Pointing"
    if index and pinky and not middle and not ring:
        return "Rock On"
    if thumb and pinky and not index and not middle and not ring:
        return "Call Me"
    if thumb and index and not middle and not ring and not pinky:
        return "Gun / L"
    return "Custom"


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_label(frame, text, x, y, color=(0, 255, 0), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, color, thick, cv2.LINE_AA)


def draw_fps(frame, fps):
    draw_label(frame, f"FPS: {fps:.1f}", 8, 24, color=(255, 255, 0), bg=(0, 0, 80))


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(camera_index=0, width=640, height=480,
        yolo_conf=0.45, show_pose=True, show_hands=True,
        web_stream=False, web_port=5000):

    print("[INFO] Loading YOLOv8n …")
    yolo = YOLO("yolov8n.pt")

    print("[INFO] Loading MediaPipe …")
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                         model_complexity=0)
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           model_complexity=0)

    print(f"[INFO] Opening camera {camera_index} …")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}")

    stream_frame = [None]
    if web_stream:
        from stream import start_stream_server
        start_stream_server(stream_frame, port=web_port)
        print(f"[INFO] Web stream at http://0.0.0.0:{web_port}")

    print("[INFO] Running — press Q to quit")
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed, retrying …")
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── YOLO object detection ──────────────────────────────────────────
        results = yolo(frame, conf=yolo_conf, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = yolo.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 200, 255) if label == "person" else (200, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, f"{label} {conf:.0%}", x1, y1, color=color)

        # ── MediaPipe Pose ────────────────────────────────────────────────
        if show_pose:
            pose_res = pose.process(rgb)
            if pose_res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, pose_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

        # ── MediaPipe Hands + Gesture ─────────────────────────────────────
        if show_hands:
            hand_res = hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                for hand_lm, handedness in zip(hand_res.multi_hand_landmarks,
                                               hand_res.multi_handedness):
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

                    gesture = classify_gesture(hand_lm, handedness)
                    wx = int(hand_lm.landmark[0].x * w)
                    wy = int(hand_lm.landmark[0].y * h)
                    draw_label(frame, gesture, wx, wy - 12,
                               color=(0, 255, 180), bg=(0, 60, 0))

        # ── FPS overlay ───────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        draw_fps(frame, fps)

        # ── Output ────────────────────────────────────────────────────────
        if web_stream:
            stream_frame[0] = frame.copy()

        cv2.imshow("Vision Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-time detection — objects, persons, gestures")
    ap.add_argument("-c", "--camera",   type=int,   default=0,     help="Camera index (default 0)")
    ap.add_argument("-W", "--width",    type=int,   default=640,   help="Capture width")
    ap.add_argument("-H", "--height",   type=int,   default=480,   help="Capture height")
    ap.add_argument("--conf",           type=float, default=0.45,  help="YOLO confidence threshold")
    ap.add_argument("--no-pose",        action="store_true",       help="Disable pose skeleton")
    ap.add_argument("--no-hands",       action="store_true",       help="Disable hand/gesture detection")
    ap.add_argument("--web",            action="store_true",       help="Enable Flask web stream")
    ap.add_argument("--port",           type=int,   default=5000,  help="Web stream port")
    args = ap.parse_args()

    run(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        yolo_conf=args.conf,
        show_pose=not args.no_pose,
        show_hands=not args.no_hands,
        web_stream=args.web,
        web_port=args.port,
    )
