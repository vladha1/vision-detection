import json
import os
import tempfile
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template

from cutout import process_drawing
from fish_tank import WallTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(BASE_DIR, "sprites")
STATE_PATH = os.path.join(BASE_DIR, "fish_state.json")
SCENE_STATE_PATH = os.path.join(BASE_DIR, "scene_state.json")
CALIBRATION_PATH = os.path.join(BASE_DIR, "calibration.json")

MAX_FISH = 20
# Was 0.8 to exclude the Mac's dock/menu bar when the projector mirrored the
# desktop; not needed now that the tank runs as its own fullscreen browser page.
PLAYABLE_HEIGHT_FRAC = 1.0

# New scenes just need adding here + a matching branch in static/tank.js.
AVAILABLE_SCENES = ["fish", "flowers", "pacman", "driving"]
DEFAULT_SCENE = "fish"

DEFAULT_COLORS = [
    "#ff8c28", "#ff4646", "#5aaaff", "#ffd700", "#be5aff",
    "#5aff96", "#ff6eb4", "#78dcff", "#f5f5f5", "#aaff3c",
]

os.makedirs(SPRITES_DIR, exist_ok=True)
state_lock = threading.Lock()


def default_roster():
    now = time.time()
    return [
        {
            "id": f"default-{i}",
            "kind": "procedural",
            "color": color,
            "temperament": "seek" if i % 2 == 0 else "flee",
            "created_at": now + i * 0.001,
        }
        for i, color in enumerate(DEFAULT_COLORS)
    ]


def save_state():
    with open(STATE_PATH, "w") as f:
        json.dump(roster, f, indent=2)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return default_roster()


def load_scene():
    if os.path.exists(SCENE_STATE_PATH):
        with open(SCENE_STATE_PATH) as f:
            scene = json.load(f).get("scene")
            if scene in AVAILABLE_SCENES:
                return scene
    return DEFAULT_SCENE


def save_scene():
    with open(SCENE_STATE_PATH, "w") as f:
        json.dump({"scene": current_scene}, f)


with open(CALIBRATION_PATH) as f:
    calib = json.load(f)

homography = np.array(calib["homography"], dtype=np.float32)
PROJECTOR_SIZE = calib["projector_size"]
PLAYABLE_HEIGHT = int(PROJECTOR_SIZE[1] * PLAYABLE_HEIGHT_FRAC)

tracker = WallTracker(calib["camera_index"], homography, calib.get("camera_points"))
tracker.start()

roster = load_state()
if not os.path.exists(STATE_PATH):
    save_state()

current_scene = load_scene()
scene_lock = threading.Lock()

# Mobile controller relay. A phone at /controller posts arrow presses here and
# the projector display polls them to drive Pac-Man. Which input is live -
# camera hand or phone arrows - is an explicit switch (input_state["mode"]),
# set from the admin page or the controller page, not auto-detected. In
# "controller" mode the Pac-Man scene ignores the camera hand entirely.
CONTROL_DIRS = ("up", "down", "left", "right")
INPUT_MODES = ("hand", "controller")
pm_control = {"dir": None, "reset": 0}  # "reset" is a counter the display watches to restart
pm_state = {"score": 0, "lives": 3, "level": 1}
# Default to the phone controller so nothing moves until someone presses an
# arrow. Hand-gesture play only kicks in when explicitly switched to "hand".
input_state = {"mode": "controller"}
control_lock = threading.Lock()


app = Flask(__name__)


@app.route("/")
def tank():
    return render_template("tank.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/controller")
def controller():
    return render_template("controller.html")


@app.route("/sprites/<path:filename>")
def sprites(filename):
    return send_from_directory(SPRITES_DIR, filename)


@app.route("/api/config")
def api_config():
    return jsonify({"width": PROJECTOR_SIZE[0], "height": PROJECTOR_SIZE[1], "playable_height": PLAYABLE_HEIGHT})


@app.route("/api/scenes")
def api_scenes():
    return jsonify(AVAILABLE_SCENES)


@app.route("/api/scene")
def api_scene_get():
    with scene_lock:
        return jsonify({"scene": current_scene})


@app.route("/api/scene", methods=["POST"])
def api_scene_set():
    global current_scene
    data = request.get_json(force=True, silent=True) or {}
    scene = data.get("scene")
    if scene not in AVAILABLE_SCENES:
        return jsonify({"error": f"scene must be one of {AVAILABLE_SCENES}"}), 400
    with scene_lock:
        current_scene = scene
        save_scene()
    return jsonify({"scene": current_scene})


@app.route("/api/hand")
def api_hand():
    # While the phone controller is the chosen input, don't recognize the hand
    # at all on the Pac-Man scene - the phone is in charge.
    if current_scene == "pacman" and input_state["mode"] == "controller":
        return jsonify(None)
    hand = tracker.get_hand()
    if hand is not None and hand[1] > PLAYABLE_HEIGHT:
        hand = None
    return jsonify({"x": hand[0], "y": hand[1]} if hand else None)


@app.route("/api/control", methods=["POST"])
def api_control_set():
    data = request.get_json(force=True, silent=True) or {}
    direction = data.get("dir")
    with control_lock:
        if direction in CONTROL_DIRS:
            pm_control["dir"] = direction  # last press persists, classic-arcade style
    return jsonify({"ok": True})


@app.route("/api/control")
def api_control_get():
    with control_lock:
        return jsonify({
            "dir": pm_control["dir"],
            "mode": input_state["mode"],
            "reset": pm_control["reset"],
        })


@app.route("/api/pmreset", methods=["POST"])
def api_pmreset():
    # Bumps a counter the display watches; works for both hand and phone play.
    with control_lock:
        pm_control["reset"] += 1
        pm_control["dir"] = None
    return jsonify({"ok": True})


@app.route("/api/inputmode", methods=["POST"])
def api_inputmode_set():
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode")
    if mode not in INPUT_MODES:
        return jsonify({"error": f"mode must be one of {INPUT_MODES}"}), 400
    with control_lock:
        input_state["mode"] = mode
        if mode == "controller":
            pm_control["dir"] = None  # start fresh: wait for a press
    return jsonify({"mode": mode})


@app.route("/api/inputmode")
def api_inputmode_get():
    with control_lock:
        return jsonify({"mode": input_state["mode"]})


@app.route("/api/pmstate", methods=["POST"])
def api_pmstate_set():
    data = request.get_json(force=True, silent=True) or {}
    with control_lock:
        if isinstance(data.get("score"), int):
            pm_state["score"] = data["score"]
        if isinstance(data.get("lives"), int):
            pm_state["lives"] = data["lives"]
        if isinstance(data.get("level"), int):
            pm_state["level"] = data["level"]
    return jsonify({"ok": True})


@app.route("/api/pmstate")
def api_pmstate_get():
    with control_lock:
        return jsonify(dict(pm_state))


@app.route("/api/fish")
def api_fish_list():
    with state_lock:
        return jsonify(roster)


@app.route("/api/fish", methods=["POST"])
def api_fish_upload():
    file = request.files.get("photo")
    if file is None:
        return jsonify({"error": "no photo uploaded"}), 400
    temperament = request.form.get("temperament", "seek")
    if temperament not in ("seek", "flee"):
        temperament = "seek"

    fish_id = uuid.uuid4().hex[:10]
    suffix = os.path.splitext(file.filename or "photo.jpg")[1] or ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    file.save(tmp_path)
    out_filename = f"{fish_id}.png"
    out_path = os.path.join(SPRITES_DIR, out_filename)
    ok = process_drawing(tmp_path, out_path)
    os.remove(tmp_path)
    if not ok:
        return jsonify({"error": "couldn't find a clear drawing in that photo - try more contrast against the background"}), 400

    entry = {
        "id": fish_id,
        "kind": "image",
        "filename": out_filename,
        "temperament": temperament,
        "created_at": time.time(),
    }
    with state_lock:
        roster.append(entry)
        while len(roster) > MAX_FISH:
            oldest = roster.pop(0)
            if oldest.get("kind") == "image":
                old_path = os.path.join(SPRITES_DIR, oldest["filename"])
                if os.path.exists(old_path):
                    os.remove(old_path)
        save_state()
    return jsonify(entry), 201


@app.route("/api/fish/<fish_id>", methods=["PATCH"])
def api_fish_update(fish_id):
    data = request.get_json(force=True, silent=True) or {}
    temperament = data.get("temperament")
    if temperament not in ("seek", "flee"):
        return jsonify({"error": "temperament must be 'seek' or 'flee'"}), 400
    with state_lock:
        for entry in roster:
            if entry["id"] == fish_id:
                entry["temperament"] = temperament
                save_state()
                return jsonify(entry)
    return jsonify({"error": "not found"}), 404


@app.route("/api/fish/<fish_id>", methods=["DELETE"])
def api_fish_delete(fish_id):
    with state_lock:
        for i, entry in enumerate(roster):
            if entry["id"] == fish_id:
                removed = roster.pop(i)
                if removed.get("kind") == "image":
                    path = os.path.join(SPRITES_DIR, removed["filename"])
                    if os.path.exists(path):
                        os.remove(path)
                save_state()
                return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True)
