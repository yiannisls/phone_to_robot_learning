"""Measure the fixed BC checkpoint on reproducible new starting positions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def failure_stage(result, folder):
    """A first diagnosis from measured outcomes, not a proven root cause."""
    if result["task_success"]:
        return None
    if result["out_of_bounds"]:
        return "out_of_bounds"
    if result["peak_cube_height_m"] <= 0.06:
        return "did_not_lift"
    with np.load(folder / "rollout.npz", allow_pickle=False) as episode:
        cube = episode["state_cube"][-1, :3]
        hand = episode["state_ee_full"][-1, :3]
        goal = episode["goal_pos"][-1]
    if np.max(np.abs(cube[:2] - goal[:2])) >= 0.04:
        return "did_not_reach_bin"
    if cube[2] >= 0.05:
        return "not_placed"
    if result["final_gripper_command"] <= 0.4:
        return "not_released"
    if np.linalg.norm(hand - cube) <= 0.07:
        return "did_not_retreat"
    return "did_not_remain_stable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_FOLDER / "checkpoints/bc_v1/policy.pt")
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "results/bc_v1_layouts")
    parser.add_argument("--layouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed cases from this exact layout manifest.")
    parser.add_argument("--jaw-torque-limit", type=float)
    parser.add_argument("--gripper-close-rate", type=float, default=2.0)
    parser.add_argument("--range-x", type=float, nargs=2, default=[-0.025, 0.025])
    parser.add_argument("--range-y", type=float, nargs=2, default=[-0.015, 0.025])
    args = parser.parse_args()
    bounds = np.array([args.range_x, args.range_y])
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        parser.error("Each layout range must contain finite increasing bounds")
    for value in (args.jaw_torque_limit, args.gripper_close_rate):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error("Torque and closing-rate limits must be finite and positive.")
    if args.layouts < 1 or args.seed < 0:
        parser.error("Use a positive layout count and a nonnegative seed.")
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=args.resume)

    rng = np.random.default_rng(args.seed)
    cases = [
        {"name": f"layout_{index:02d}", "offset_xy_m": rng.uniform(
            bounds[:, 0], bounds[:, 1]
        ).tolist()}
        for index in range(args.layouts)
    ]
    manifest = {
        "purpose": "development diagnostic; not the held-out recording test",
        "seed": args.seed, "range_xy_m": [bounds[:, 0].tolist(), bounds[:, 1].tolist()],
        "checkpoint_sha256": checkpoint_hash, "cases": cases,
        "jaw_torque_limit_nm": args.jaw_torque_limit,
        "gripper_close_rate_rad_s": args.gripper_close_rate,
    }
    # Save the positions before running; a failed process cannot hide its layout.
    manifest_path = args.out / "layouts.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("The saved layouts or checkpoint differ; use a new output folder.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    results = []
    for case in cases:
        folder = args.out / case["name"]
        command = [
            sys.executable, str(PROJECT_FOLDER / "scripts/evaluate_policy.py"),
            "--checkpoint", str(checkpoint), "--out", str(folder.resolve()),
            "--label", case["name"], "--offset-xy",
            *[np.format_float_positional(value, unique=True, trim="-")
              for value in case["offset_xy_m"]],
        ]
        command.extend(["--gripper-close-rate", str(args.gripper_close_rate)])
        if args.jaw_torque_limit is not None:
            command.extend(["--jaw-torque-limit", str(args.jaw_torque_limit)])
        if args.video:
            command.append("--video")
        result_path = folder / "result.json"
        if args.resume and result_path.exists():
            required = ["rollout.npz", "physics.npz"]
            if args.video:
                required.append("validation.avi")
            if not all((folder / name).exists() for name in required):
                raise SystemExit(f"Incomplete saved outputs in {folder}; preserve and inspect them.")
        else:
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise SystemExit(completed.stdout + completed.stderr)
        result = json.loads((folder / "result.json").read_text())
        assert result["checkpoint_sha256"] == checkpoint_hash
        assert result["offset_xy_m"] == case["offset_xy_m"]
        assert result["gripper_close_rate_rad_s"] == args.gripper_close_rate
        if args.jaw_torque_limit is not None:
            assert result["jaw_torque_range_nm"] == [-args.jaw_torque_limit, args.jaw_torque_limit]
        assert result["evaluation_script_sha256"] == hashlib.sha256(
            (PROJECT_FOLDER / "scripts/evaluate_policy.py").read_bytes()
        ).hexdigest()
        result["failure_stage"] = failure_stage(result, folder)
        results.append(result)
        # Keep completed outcomes available even if a later run is interrupted.
        (args.out / "completed.json").write_text(json.dumps(results, indent=2))
        print(f"{case['name']}: success={result['task_success']}, "
              f"contact_depth_pass={result['quality_pass']}, "
              f"failure_stage={result['failure_stage']}", flush=True)

    summary = {
        **manifest, "episodes": results,
        "task_successes": sum(result["task_success"] for result in results),
        "task_and_contact_depth_passes": sum(
            result["task_success"] and result["quality_pass"] for result in results
        ),
        "maximum_cube_penetration_m": max(result["max_cube_penetration_m"] for result in results),
        "maximum_peak_normal_force_per_jaw_n": np.max(
            [result["peak_normal_force_per_jaw_n"] for result in results], axis=0
        ).tolist(),
        "failure_stage_note": "Outcome-based diagnosis; inspect traces before assigning a cause.",
        "batch_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Completed: {summary['task_successes']}/{len(results)} task successes; "
          f"{summary['task_and_contact_depth_passes']}/{len(results)} also passed contact depth.", flush=True)


if __name__ == "__main__":
    main()
