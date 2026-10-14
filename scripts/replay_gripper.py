"""Replay a saved gripper episode and compare its states and physics samples."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from record_gripper import GripperRecorder, create_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "ROBOT_LEARNING_HW3" not in os.environ:
        parser.error("Set ROBOT_LEARNING_HW3 to your homework folder")
    summary = json.loads((args.episode_folder / "summary.json").read_text())
    if summary["mujoco_version"] != mujoco.__version__:
        parser.error("Replay requires the recorded MuJoCo version")
    with np.load(args.episode_folder / "episode.npz", allow_pickle=False) as saved:
        original = {key: saved[key].copy() for key in saved.files}
    with np.load(args.episode_folder / "physics.npz", allow_pickle=False) as saved:
        original_physics = saved["samples"].copy()
    args.out.mkdir(parents=True, exist_ok=False)
    config = summary["config"]
    env = create_environment(Path(os.environ["ROBOT_LEARNING_HW3"]),
                             config["offset_xy_m"], config["jaw_torque_limit_nm"])
    try:
        signature = int(original["state_signature"])
        initial = original["initial_integration_state"]
        assert len(initial) == mujoco.mj_stateSize(env.model, signature)
        mujoco.mj_setState(env.model, env.data, initial, signature)
        mujoco.mj_forward(env.model, env.data)
        # Forward computes derived geometry; restore the saved solver warm start.
        mujoco.mj_setState(env.model, env.data, initial, signature)
        recorder = GripperRecorder(env, summary["gripper_close_rate_rad_s"])
        for action in original["actions"]:
            recorder.step(action[:3], action[3])
        recorder.save(args.out, "replay_verification", "commands_complete", config)
        with np.load(args.out / "episode.npz", allow_pickle=False) as saved:
            actual_qpos, actual_qvel = saved["qpos"], saved["qvel"]
        with np.load(args.out / "physics.npz", allow_pickle=False) as saved:
            actual_physics = saved["samples"]
        state_error = float(np.abs(actual_qpos - original["qpos"]).max())
        velocity_error = float(np.abs(actual_qvel - original["qvel"]).max())
        physics_error = float(np.abs(actual_physics - original_physics).max()) if len(original_physics) else 0
        passed = np.allclose(actual_qpos, original["qpos"], atol=1e-9, rtol=0) and np.allclose(
            actual_qvel, original["qvel"], atol=1e-9, rtol=0) and np.allclose(
            actual_physics, original_physics, atol=1e-7, rtol=0)
        report = {"passed": bool(passed), "max_qpos_error": state_error,
                  "max_qvel_error": velocity_error, "max_physics_sample_error": physics_error,
                  "steps": len(original["actions"]), "source": str(args.episode_folder.resolve()),
                  "note": "Maximum physics difference combines different physical units; arrays retain column names. No control improvement claimed."}
        (args.out / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise SystemExit("Replay differs: inspect verification.json before using these data")
    finally:
        env.renderer.close()


if __name__ == "__main__":
    main()
