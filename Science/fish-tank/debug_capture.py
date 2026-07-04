import json
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
countdown = int(sys.argv[2]) if len(sys.argv) > 2 else 10
capture_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

CROP_PADDING_FRAC = 0.2
CROP_TARGET_SIZE = 640

crop_box = None
if os.path.exists("calibration.json"):
    with open("calibration.json") as f:
        calib = json.load(f)
    cam_index = calib.get("camera_index", cam_index)
    camera_points = calib.get("camera_points")
    if camera_points:
        pts = np.array(camera_points, dtype=np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_x = (x1 - x0) * CROP_PADDING_FRAC
        pad_y = (y1 - y0) * CROP_PADDING_FRAC
        crop_box = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
        print(f"using crop box from calibration.json: {crop_box}")
    else:
        print("calibration.json has no camera_points - capturing full frame")
else:
    print("no calibration.json found - capturing full frame")

for remaining in range(countdown, 0, -1):
    print(f"starting in {remaining}...", flush=True)
    time.sleep(1)
print("capturing now - hold your hand up near the wall", flush=True)

cap = cv2.VideoCapture(cam_index)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)
drawer = mp.solutions.drawing_utils

detected = 0
total = 0
last_full_frame = None
best_frame = None
start = time.time()
while time.time() - start < capture_seconds:
    ok, frame = cap.read()
    if not ok:
        continue
    total += 1
    last_full_frame = frame

    if crop_box is not None:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = crop_box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        crop = frame[y0:y1, x0:x1]
        scale = CROP_TARGET_SIZE / max(crop.shape[:2])
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        crop = frame

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        detected += 1
        annotated = crop.copy()
        for lm in result.multi_hand_landmarks:
            drawer.draw_landmarks(annotated, lm, mp.solutions.hands.HAND_CONNECTIONS)
        best_frame = annotated
    elif best_frame is None:
        best_frame = crop.copy()

cap.release()
print(f"detected {detected}/{total} frames, crop_shape={None if best_frame is None else best_frame.shape}")
if last_full_frame is not None:
    cv2.imwrite("debug_frame_full.jpg", last_full_frame)
if best_frame is not None:
    cv2.imwrite("debug_frame.jpg", best_frame)
    print("saved debug_frame.jpg (cropped view) and debug_frame_full.jpg (full camera view)")
else:
    print("no frames captured at all - camera read failing")
