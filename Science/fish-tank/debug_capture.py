import sys
import time

import cv2
import mediapipe as mp

cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
countdown = int(sys.argv[2]) if len(sys.argv) > 2 else 10
capture_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

for remaining in range(countdown, 0, -1):
    print(f"starting in {remaining}...", flush=True)
    time.sleep(1)
print("capturing now - hold your hand in front of the camera", flush=True)

cap = cv2.VideoCapture(cam_index)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)
drawer = mp.solutions.drawing_utils

detected = 0
total = 0
last_frame = None
best_frame = None
start = time.time()
while time.time() - start < capture_seconds:
    ok, frame = cap.read()
    if not ok:
        continue
    total += 1
    last_frame = frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        detected += 1
        annotated = frame.copy()
        for lm in result.multi_hand_landmarks:
            drawer.draw_landmarks(annotated, lm, mp.solutions.hands.HAND_CONNECTIONS)
        best_frame = annotated

cap.release()
print(f"detected {detected}/{total} frames, resolution={None if last_frame is None else last_frame.shape}")
out = best_frame if best_frame is not None else last_frame
if out is not None:
    cv2.imwrite("debug_frame.jpg", out)
    print("saved debug_frame.jpg")
else:
    print("no frames captured at all - camera read failing")
