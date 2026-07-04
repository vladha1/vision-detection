import argparse
import json
import os
import sys

import cv2
import numpy as np
import pygame
from screeninfo import get_monitors

MARKER_RADIUS = 14


def list_monitors():
    monitors = get_monitors()
    for i, m in enumerate(monitors):
        print(f"[{i}] {m.name} {m.width}x{m.height} at ({m.x},{m.y})")
    return monitors


def draw_marker(surface, pos, label):
    surface.fill((0, 0, 0))
    pygame.draw.circle(surface, (0, 255, 0), pos, MARKER_RADIUS, 3)
    pygame.draw.line(surface, (0, 255, 0), (pos[0] - 20, pos[1]), (pos[0] + 20, pos[1]), 2)
    pygame.draw.line(surface, (0, 255, 0), (pos[0], pos[1] - 20), (pos[0], pos[1] + 20), 2)
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"Click this marker in the camera window ({label})", True, (255, 255, 255))
    surface.blit(text, (40, 40))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--projector-monitor", type=int, default=None)
    parser.add_argument("--margin", type=int, default=120, help="inset of markers from projector edges, in px")
    parser.add_argument("--out", default="calibration.json")
    args = parser.parse_args()

    monitors = list_monitors()
    proj_index = args.projector_monitor
    if proj_index is None:
        proj_index = int(input("Which monitor index is the projector? "))
    mon = monitors[proj_index]

    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{mon.x},{mon.y}"
    pygame.init()
    screen = pygame.display.set_mode((mon.width, mon.height), pygame.NOFRAME)
    pygame.display.set_caption("fish-tank calibration")

    m = args.margin
    w, h = mon.width, mon.height
    projector_points = [(m, m), (w - m, m), (w - m, h - m), (m, h - m)]
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera_index}")

    camera_points = []
    clicked = {"pt": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    cv2.namedWindow("camera - click the marker")
    cv2.setMouseCallback("camera - click the marker", on_mouse)

    for pos, label in zip(projector_points, labels):
        clicked["pt"] = None
        draw_marker(screen, pos, label)
        pygame.display.flip()

        while clicked["pt"] is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit("Calibration cancelled")
            ok, frame = cap.read()
            if not ok:
                continue
            preview = frame.copy()
            cv2.putText(preview, f"click: {label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("camera - click the marker", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                sys.exit("Calibration cancelled")

        camera_points.append(clicked["pt"])
        print(f"{label}: camera={clicked['pt']} projector={pos}")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

    src = np.array(camera_points, dtype=np.float32)
    dst = np.array(projector_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst)

    data = {
        "camera_index": args.camera_index,
        "projector_monitor": proj_index,
        "projector_size": [mon.width, mon.height],
        "homography": homography.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved calibration to {args.out}")


if __name__ == "__main__":
    main()
