#!/usr/bin/env python3
"""
Standard dartboard scoring geometry.

Input: (x, y) normalised to [-1, 1] from board centre.
       r = 1.0 is the outer edge of the double ring.
"""
import math

# Normalised radii relative to double-ring outer edge (170 mm physical)
_R_BULLSEYE     = 6.35  / 170   # inner bull / bullseye
_R_BULL         = 15.9  / 170   # outer bull
_R_TRIPLE_INNER = 107.0 / 170
_R_TRIPLE_OUTER = 115.0 / 170
_R_DOUBLE_INNER = 162.0 / 170
_R_DOUBLE_OUTER = 1.0

# Clockwise from 12 o'clock, one sector per 18°, centred on sector 20
_SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def dart_score(x: float, y: float) -> tuple[int, str]:
    """
    Returns (points, label) for a hit at normalised board position (x, y).
    x increases rightward, y increases downward; origin at board centre.
    """
    r = math.hypot(x, y)

    if r < _R_BULLSEYE:
        return 50, "Bullseye"
    if r < _R_BULL:
        return 25, "Bull"
    if r > _R_DOUBLE_OUTER:
        return 0, "Miss"

    # Angle from 12 o'clock, clockwise (0–360°)
    angle = math.degrees(math.atan2(x, -y)) % 360

    # Sector 20 is centred at 0°; shift by half-sector (9°) before binning
    sector_idx = int((angle + 9) / 18) % 20
    base = _SECTORS[sector_idx]

    if _R_TRIPLE_INNER <= r <= _R_TRIPLE_OUTER:
        return base * 3, f"T{base}"
    if _R_DOUBLE_INNER <= r <= _R_DOUBLE_OUTER:
        return base * 2, f"D{base}"
    return base, str(base)
