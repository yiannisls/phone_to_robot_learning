import json
from pathlib import Path
import subprocess
import sys

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
RECORDINGS = {
    "IMG_5539": [2.5, 7.3],
    "IMG_5540": [2.7, 6.6],
    "IMG_5541": [2.4, 6.3],
    "IMG_5542": [3.0, 7.0],
    "IMG_5543": [3.0, 5.8],
}


def run_script(name, *arguments):
    subprocess.run(
        [sys.executable, str(PROJECT_FOLDER / "scripts" / name), *arguments],
        cwd=PROJECT_FOLDER, check=True
    )


for source, (start, end) in RECORDINGS.items():
    transport_path = PROJECT_FOLDER / "data" / "processed" / f"{source}_transport.csv"
    if transport_path.exists():
        print(f"{source}: transport already prepared.", flush=True)
        continue

    print(f"Preparing {source}: {start} to {end} seconds.", flush=True)
    run_script("track_marker.py", str(PROJECT_FOLDER / "data" / "raw" / f"{source}.mp4"))
    run_script("select_movement.py", source, "--start", str(start), "--end", str(end))
    run_script("estimate_pose.py", source)
    run_script("make_transport.py", source)

manifest = {
    "intervals_seconds": RECORDINGS,
    "annotation_origin": {
        "IMG_5539": "User rechecked on 2026-10-07",
        "other_recordings": "Earlier manual review of the same footage",
    },
}
manifest_path = PROJECT_FOLDER / "data" / "processed" / "recording_intervals.json"
if manifest_path.exists():
    if json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("Saved intervals differ; use a new extraction version.")
else:
    with manifest_path.open("x") as file:
        json.dump(manifest, file, indent=2)

print("All five transport curves are prepared.")
