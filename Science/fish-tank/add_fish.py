import argparse
import os
import shutil
import time

import requests


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


def upload(server, path, temperament, name):
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{server}/api/fish",
                files={"photo": (name, f, "image/jpeg")},
                data={"temperament": temperament},
                timeout=15,
            )
        if resp.ok:
            print(f"New fish added from {name}: {resp.json().get('filename')}")
        else:
            print(f"upload failed for {name}: {resp.status_code} {resp.text}")
    except requests.RequestException as exc:
        print(f"could not reach {server} ({exc}) - is server.py running?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", default="inbox", help="drop photos of drawings here")
    parser.add_argument("--downloads", default=os.path.expanduser("~/Downloads"),
                         help="watched for new AirDropped photos and auto-moved into --inbox")
    parser.add_argument("--no-downloads-watch", action="store_true", help="disable the Downloads watcher")
    parser.add_argument("--server", default="http://localhost:5050", help="fish-tank server to upload into")
    parser.add_argument("--temperament", default="seek", choices=["seek", "flee"])
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.inbox, exist_ok=True)
    print(f"Watching {args.inbox}/ - drop a photo of your drawing in there to add a fish!")
    if not args.no_downloads_watch:
        print(f"Also watching {args.downloads} - AirDrop a photo there and it'll move to {args.inbox}/ automatically.")
    print(f"Uploading to {args.server} (make sure server.py is running). Ctrl+C to stop.")

    seen_downloads = set(os.listdir(args.downloads)) if not args.no_downloads_watch else set()
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
            upload(args.server, path, args.temperament, name)
            os.remove(path)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
