"""Collect agent-operated command trials for force diagnostics, including failures."""
import argparse
import json
from pathlib import Path

import numpy as np

from record_gripper import GripperRecorder, create_environment, export_zarr

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_FOLDER / "data/demonstrations_v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--torque-limits", type=float, nargs="+", default=[3.5, 1.0, 0.5, 0.3])
    parser.add_argument("--zarr", action="store_true")
    args = parser.parse_args()
    import os
    if "ROBOT_LEARNING_HW3" not in os.environ:
        parser.error("Set ROBOT_LEARNING_HW3 to your homework folder")
    if not np.isfinite(args.torque_limits).all() or min(args.torque_limits) <= 0:
        parser.error("Torque limits must be finite and positive")
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    # These three sources already belong to training; held-out sources are not opened.
    for source in ("IMG_5539", "IMG_5540", "IMG_5541"):
        path = args.data / f"{source}.npz"
        with np.load(path, allow_pickle=False) as saved:
            assert saved["source_video"].item() == source and saved["split"].item() == "train"
            actions = saved["actions"].copy()
            offset = saved["cube_offset_xy_m"].tolist()
        for limit in args.torque_limits:
            env = create_environment(Path(os.environ["ROBOT_LEARNING_HW3"]), offset, limit)
            try:
                recorder = GripperRecorder(env, 1.0)
                for action in actions:
                    recorder.step(action[:3], action[3])
                folder = args.out / f"{source}_torque{limit:g}"
                folder.mkdir()
                config = {"offset_xy_m": offset, "jaw_torque_limit_nm": limit,
                          "source_video": source, "source_family_split": "train",
                          "purpose": "force_diagnostic_only", "move_mm": 5, "max_seconds": 120}
                recorder.save(folder, "agent_operated_saved_commands", "commands_complete", config, path)
                if args.zarr:
                    export_zarr(folder)
                cube, hand, goal = env.get_cube_state()[:3], env.get_ee_pos(), env.get_goal_pos()
                physics = np.array(recorder.physics)
                lifted = float(physics[:, 11].max())
                stable = np.linalg.norm(cube - recorder.states[-6]["state_cube"][:3]) < 0.005
                success = bool(lifted > 0.06 and np.max(np.abs(cube[:2] - goal[:2])) < 0.04
                               and cube[2] < 0.05 and np.linalg.norm(hand - cube) > 0.07 and stable)
                held = physics[np.all(physics[:, 6:8] > 1, axis=1) & (physics[:, 11] > 0.05), 6:8]
                row = {"source": source, "torque_limit_nm": limit, "task_success": success,
                       "quality_pass": float(physics[:, 8].max()) <= 0.002,
                       "peak_force_n": float(physics[:, 6:8].max()),
                       "holding_median_force_n": float(np.median(held)) if len(held) else None,
                       "penetration_mm": float(physics[:, 8].max() * 1000), "peak_cube_z_m": lifted}
                results.append(row)
                (folder / "result.json").write_text(json.dumps(row, indent=2))
                print(json.dumps(row), flush=True)
            finally:
                env.renderer.close()
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
