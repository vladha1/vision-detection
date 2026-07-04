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
