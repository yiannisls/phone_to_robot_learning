"""Record the fixed 18-attempt visual-data pilot and retain every result."""

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from generate_demo import SOURCE_SPLITS
from record_visual_episode import COLORS, PILOT_SLOTS

PROJECT = Path(__file__).resolve().parents[1]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(folder, manifest):
    temporary = folder / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(folder / "manifest.json")


def check_initial_scene(folder, cases):
    """Instructions may differ; their initial physical scene must not."""
    if any(case["status"] != "completed" for case in cases):
        return {"passed": False, "reason": "An episode did not complete recording."}

    reference = None
    max_image_difference = 0
    states_match = True
    cameras_match = True
    for case in cases:
        with np.load(folder / case["id"] / "episode.npz", allow_pickle=False) as saved:
            current = {
                key: saved[key][0].copy()
                for key in ["images", "robot_state", "qpos"]
            }
        if reference is None:
            reference = current
            camera = case["report"]["camera_settings"]
            continue
        states_match &= np.array_equal(current["robot_state"], reference["robot_state"])
        states_match &= np.array_equal(current["qpos"], reference["qpos"])
        cameras_match &= case["report"]["camera_settings"] == camera
        difference = np.abs(
            current["images"].astype(int) - reference["images"].astype(int)
        ).max()
        max_image_difference = max(max_image_difference, int(difference))

    return {
        "passed": bool(states_match and cameras_match and max_image_difference <= 1),
        "robot_and_physical_states_match": bool(states_match),
        "camera_settings_match": bool(cameras_match),
        "max_image_channel_difference": max_image_difference,
        "image_tolerance": 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    training_sources = [name for name, split in SOURCE_SPLITS.items() if split == "train"]
    parser.add_argument("--source", choices=training_sources, default="IMG_5539")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)

    recorder = PROJECT / "scripts/record_visual_episode.py"
    cases = [
        {
            "id": f"arrangement_{index:02d}_{color}",
            "arrangement": index,
            "slot_colors": list(arrangement),
            "target": color,
            "target_slot": arrangement.index(color),
            "status": "pending",
        }
        for index, arrangement in enumerate(permutations(COLORS))
        for color in COLORS
    ]
    manifest = {
        "purpose": "Demonstration-collection pilot; not a policy evaluation.",
        "source_family": args.source,
        "split": "train",
        "planned_attempts": 18,
        "slots_xy_m": PILOT_SLOTS.tolist(),
        "recorder_sha256": checksum(recorder),
        "collector_sha256": checksum(Path(__file__)),
        "automatic_retries": False,
        "training_inclusion": "None; candidates only.",
        "cases": cases,
    }
    shutil.copyfile(recorder, args.out / "record_visual_episode.py.txt")
    shutil.copyfile(__file__, args.out / "collect_visual_pilot.py.txt")
    save_manifest(args.out, manifest)

    for case in cases:
        command = [
            sys.executable, str(recorder),
            "--source", args.source,
            "--placement", "feedback",
            "--framing", "workspace",
            "--slot-colors", *case["slot_colors"],
            "--color", case["target"],
            "--out", str(args.out / case["id"]),
        ]
        with (args.out / f"{case['id']}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
            )
        case["exit_code"] = result.returncode
        report_path = args.out / case["id"] / "summary.json"
        if result.returncode == 0 and report_path.is_file():
            report = json.loads(report_path.read_text())
            if report["script_sha256"] != manifest["recorder_sha256"]:
                raise RuntimeError("Recorder changed during collection.")
            case["status"] = "completed"
            case["report"] = report
            case["episode_sha256"] = checksum(args.out / case["id"] / "episode.npz")
            case["report_sha256"] = checksum(report_path)
            case["eligible_before_scene_check"] = bool(
                report["task_success"]
                and report["quality_pass"]
                and report["camera_coverage"]["passed"]
                and report["recorded_transport_actions_preserved"] > 0
            )
            print(
                f"{case['id']}: task={report['task_success']}, "
                f"quality={report['quality_pass']}, "
                f"camera={report['camera_coverage']['passed']}",
                flush=True,
            )
        else:
            case["status"] = "process_error"
            case["eligible_before_scene_check"] = False
            print(f"{case['id']}: recording error; see its log.", flush=True)
        save_manifest(args.out, manifest)

    groups = {}
    for index in range(6):
        group = [case for case in cases if case["arrangement"] == index]
        check = check_initial_scene(args.out, group)
        groups[str(index)] = check
        for case in group:
            case["eligible_for_review"] = bool(
                case["eligible_before_scene_check"] and check["passed"]
            )
    manifest["initial_scene_checks"] = groups
    manifest["counts"] = {
        "attempted": len(cases),
        "recorded": sum(case["status"] == "completed" for case in cases),
        "task_success": sum(case.get("report", {}).get("task_success", False) for case in cases),
        "quality_pass": sum(case.get("report", {}).get("quality_pass", False) for case in cases),
        "eligible_for_review": sum(case["eligible_for_review"] for case in cases),
        "accepted_for_training": 0,
    }
    save_manifest(args.out, manifest)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
