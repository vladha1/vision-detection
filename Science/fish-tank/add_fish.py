import argparse
import os
import shutil
import time

import cv2
import numpy as np

SATURATION_BOOST = 1.6
BRIGHTNESS_BOOST = 1.25


def boost_colors(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= SATURATION_BOOST
    hsv[..., 2] *= BRIGHTNESS_BOOST
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_drawing(src_path, out_path, target_width=140):
    img = cv2.imread(src_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20:
        return False
    pad = int(0.08 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop_bgr = boost_colors(img[y0:y1, x0:x1])
    crop_mask = cv2.GaussianBlur(mask[y0:y1, x0:x1], (5, 5), 0)
    b, g, r = cv2.split(crop_bgr)
    bgra = cv2.merge([b, g, r, crop_mask])
    h2, w2 = bgra.shape[:2]
    scale = target_width / w2
    bgra = cv2.resize(bgra, (target_width, max(1, int(h2 * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, bgra)
    return True


def sync_from_downloads(downloads_dir, inbox_dir, seen):
    try:
        current = set(os.listdir(downloads_dir))
    except FileNotFoundError:
        return seen
    for name in sorted(current - seen):
        src = os.path.join(downloads_dir, name)
        if not os.path.isfile(src):
            continue
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            if name.lower().endswith(".heic"):
                print(f"[downloads] ignoring {name} (HEIC) - set iPhone Settings > Camera > Formats > Most Compatible")
            continue
        time.sleep(0.5)  # let AirDrop finish writing the file
        try:
            shutil.move(src, os.path.join(inbox_dir, name))
            print(f"[downloads] moved {name} -> {inbox_dir}/")
        except OSError as exc:
            print(f"[downloads] could not move {name}: {exc}")
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", default="inbox", help="drop photos of drawings here")
    parser.add_argument("--sprites", default="sprites", help="processed cutouts land here")
    parser.add_argument("--downloads", default=os.path.expanduser("~/Downloads"),
                         help="watched for new AirDropped photos and auto-moved into --inbox")
    parser.add_argument("--no-downloads-watch", action="store_true", help="disable the Downloads watcher")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.inbox, exist_ok=True)
    os.makedirs(args.sprites, exist_ok=True)
    print(f"Watching {args.inbox}/ - drop a photo of your drawing in there to add a fish!")
    if not args.no_downloads_watch:
        print(f"Also watching {args.downloads} - AirDrop a photo there and it'll move to {args.inbox}/ automatically.")
    print("Ctrl+C to stop.")

    seen_downloads = set(os.listdir(args.downloads)) if not args.no_downloads_watch else set()
    next_n = len(os.listdir(args.sprites)) + 1
    while True:
        if not args.no_downloads_watch:
            seen_downloads = sync_from_downloads(args.downloads, args.inbox, seen_downloads)

        for name in sorted(os.listdir(args.inbox)):
            path = os.path.join(args.inbox, name)
            if not os.path.isfile(path) or name.startswith("."):
                continue
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"skipping {name} (not a jpg/png - iPhone photos should be 'Most Compatible' format)")
                os.remove(path)
                continue
            out_path = os.path.join(args.sprites, f"fish_{next_n}.png")
            ok = process_drawing(path, out_path)
            os.remove(path)
            if ok:
                print(f"New fish added: fish_{next_n}.png")
                next_n += 1
            else:
                print(f"couldn't find a clear drawing in {name} - try more contrast against the background")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
