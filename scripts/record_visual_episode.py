"""Replay a phone demonstration and record synchronized multicube images."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from generate_demo import SOURCE_SPLITS

PROJECT = Path(__file__).resolve().parents[1]
COLORS = ("red", "green", "blue")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(env, camera):
    """Read the image and state without advancing simulation time."""
    return {
        "images": env.render_rgb(camera),
        "robot_state": np.r_[
            env.get_joint_angles(),
            env.get_ee_pos(),
            env.data.mocap_pos[env.mocap_id],
            env.data.ctrl[env.act_ids[-1]],
        ],
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "cubes": env.get_all_cubes_state(),
        "mocap_target": env.data.mocap_pos[env.mocap_id].copy(),
        "time_s": np.array(env.data.time),
    }


def create_environment(homework, source, color):
    sys.path.insert(0, str(homework))
    from hw3.sim_env import SO100MulticubeSimEnv

    env = SO100MulticubeSimEnv(
        homework / "so101_gym/assets/so100_multicube_ee.xml",
        goal_cube=color,
        seed=0,
        cube_pos_std=0,
        shuffle_cubes=False,
        render_w=320,
        render_h=240,
    )
    target = COLORS.index(color)

    # Match the saved robot, selected cube and bin.
    # Put distractors aside for this first controlled recording.
    env.data.qpos[env.qpos_idx] = source["qpos"][0, :6]
    env.data.qvel[:6] = source["qvel"][0, :6]
    distractor_positions = iter([[-0.35, 0.35, 0.021], [-0.35, 0.52, 0.021]])
    for index, indices in enumerate(env.cube_qpos_slices):
        if index == target:
            env.data.qpos[indices] = source["qpos"][0, 6:13]
            joint = env.model.joint(f"{color}_box_joint")
            start = int(env.model.jnt_dofadr[joint.id])
            env.data.qvel[start:start + 6] = source["qvel"][0, 6:12]
        else:
            env.data.qpos[indices[:3]] = next(distractor_positions)

    env.model.body("bin").pos[:] = (
        source["goal_pos"][0] - env.model.site("bin_center").pos
    )
    mujoco.mj_forward(env.model, env.data)
    env.set_targets(source["qpos"][0, :6])
    env.set_gripper(float(source["state_gripper_command"][0, 0]))
    env.set_mocap_pos(source["state_mocap_target"][0])
    env.set_mocap_quat(source["state_ee_full"][0, 3:7])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    training_sources = [name for name, split in SOURCE_SPLITS.items() if split == "train"]
    parser.add_argument("--source", choices=training_sources, default="IMG_5539")
    parser.add_argument("--color", choices=COLORS, default="green")
    parser.add_argument("--camera", default="angle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configured_homework = os.environ.get("ROBOT_LEARNING_HW3")
    if not configured_homework:
        raise SystemExit("Set ROBOT_LEARNING_HW3 to your HW3 directory.")
    homework = Path(configured_homework).expanduser().resolve()
    source_path = PROJECT / "data/demonstrations_v2" / f"{args.source}.npz"
    with np.load(source_path, allow_pickle=False) as saved:
        source = {key: saved[key].copy() for key in saved.files}
    if source["split"].item() != "train" or not source["success"].item():
        raise ValueError("Use a successful demonstration from a training source.")

    raw_video = PROJECT / "data/raw" / f"{args.source}.mp4"
    transport = PROJECT / "data/processed" / f"{args.source}_transport.csv"
    if checksum(raw_video) != source["source_sha256"].item():
        raise ValueError("The phone video does not match the demonstration.")
    if checksum(transport) != source["transport_sha256"].item():
        raise ValueError("The transport path does not match the demonstration.")

    args.out.mkdir(parents=True, exist_ok=False)
    env = create_environment(homework, source, args.color)
    writer = None
    try:
        if not np.isclose(env.dt_ctrl, source["control_dt_s"].item()):
            raise ValueError("Control timestep differs from the source.")
        if not np.isclose(env.sim_dt, source["physics_dt_s"].item()):
            raise ValueError("Physics timestep differs from the source.")
        writer = cv2.VideoWriter(
            str(args.out / "episode.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            1 / env.dt_ctrl,
            (env.render_w, env.render_h),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open the video writer.")

        observations = [capture(env, args.camera)]
        writer.write(cv2.cvtColor(observations[0]["images"], cv2.COLOR_RGB2BGR))
        initial_cubes = env.get_all_cubes_xyz().reshape(3, 3)
        target_geometry = env.model.geom(f"{args.color}_box").id
        peak_height = float(env.get_cube_state()[2])
        max_penetration = 0.0
        actions = []
        close_rate = float(source["gripper_close_rate_rad_s"])

        for index, action in enumerate(source["actions"]):
            target = source["state_mocap_target"][index + 1].astype(np.float64)
            previous = env.data.mocap_pos[env.mocap_id].copy()
            requested_gripper = float(action[3])
            actions.append(np.r_[target - previous, requested_gripper])
            env.set_mocap_pos(target)

            for _ in range(env.substeps):
                applied = float(env.data.ctrl[env.act_ids[-1]])
                if requested_gripper < applied:
                    applied = max(requested_gripper, applied - close_rate * env.sim_dt)
                else:
                    applied = requested_gripper
                env.set_gripper(applied)
                mujoco.mj_step(env.model, env.data)
                peak_height = max(peak_height, float(env.get_cube_state()[2]))
                for contact in env.data.contact:
                    if target_geometry in (contact.geom1, contact.geom2):
                        max_penetration = max(max_penetration, -float(contact.dist))

            observation = capture(env, args.camera)
            observations.append(observation)
            writer.write(cv2.cvtColor(observation["images"], cv2.COLOR_RGB2BGR))

        arrays = {
            key: np.stack([observation[key] for observation in observations])
            for key in observations[0]
        }
        actions = np.asarray(actions)
        assert len(arrays["images"]) == len(actions) + 1
        assert np.allclose(np.diff(arrays["time_s"]), env.dt_ctrl)
        assert np.allclose(np.diff(arrays["mocap_target"], axis=0), actions[:, :3])
        assert np.allclose(actions, source["actions"], atol=1e-6)
        for key, values in arrays.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite values in {key}.")

        target_index = COLORS.index(args.color)
        cubes = arrays["cubes"].reshape(-1, 3, 7)
        final_cube = cubes[-1, target_index, :3]
        goal = env.get_goal_pos()
        distractors = [index for index in range(3) if index != target_index]
        distractor_motion = float(np.linalg.norm(
            cubes[:, distractors, :3] - initial_cubes[distractors], axis=-1
        ).max())
        task_success = bool(
            peak_height > 0.06
            and np.max(np.abs(final_cube[:2] - goal[:2])) < 0.04
            and 0.015 < final_cube[2] < 0.05
            and np.linalg.norm(env.get_ee_pos() - final_cube) > 0.07
            and np.linalg.norm(final_cube - cubes[-6, target_index, :3]) < 0.005
            and env.data.ctrl[env.act_ids[-1]] > 0.4
        )
        instruction = f"Put the {args.color} cube in the bin."
        np.savez_compressed(
            args.out / "episode.npz",
            **arrays,
            actions=actions,
            phase=source["phase"],
            instruction=np.array(instruction),
            source_video=np.array(args.source),
            split=np.array("train"),
        )

        report = {
            "instruction": instruction,
            "origin": "Phone-derived demonstration replay; not human teleoperation.",
            "source_family": args.source,
            "split": "train",
            "accepted_for_training": False,
            "review_status": "Candidate for visual inspection only.",
            "phone_contribution": "Transport trajectory; other motion phases are scripted.",
            "layout": "Fixed target and bin; two distractors placed aside; no shuffling.",
            "camera": args.camera,
            "image_shape": list(arrays["images"].shape),
            "robot_state_fields": [
                "6 joint angles in radians",
                "3 measured hand coordinates in metres",
                "3 commanded hand-target coordinates in metres",
                "1 applied native gripper command",
            ],
            "action_fields": [
                "3 hand-target deltas in world-frame metres per control step",
                "1 requested native gripper command; closing rate-limited",
            ],
            "alignment": "observation[t], action[t], observation[t+1]; final observation included",
            "diagnostics_only": ["qpos", "qvel", "cubes"],
            "control_dt_s": env.dt_ctrl,
            "physics_dt_s": env.sim_dt,
            "actions": len(actions),
            "observations": len(observations),
            "task_success": task_success,
            "max_target_penetration_m": max_penetration,
            "quality_pass": max_penetration <= 0.002 and distractor_motion < 0.01,
            "max_distractor_displacement_m": distractor_motion,
            "peak_target_height_m": peak_height,
            "gripper_close_rate_rad_s": close_rate,
            "mujoco_version": mujoco.__version__,
            "source_episode_sha256": checksum(source_path),
            "source_video_sha256": checksum(raw_video),
            "transport_sha256": checksum(transport),
            "script_sha256": checksum(Path(__file__)),
            "simulator_sha256": checksum(homework / "hw3/sim_env.py"),
            "scene_sha256": checksum(homework / "so101_gym/assets/so100_multicube_ee.xml"),
            "limitation": "Contact forces remain unvalidated; no VLA was trained or evaluated.",
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        cv2.imwrite(
            str(args.out / "first-frame.jpg"),
            cv2.cvtColor(arrays["images"][0], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(args.out / "last-frame.jpg"),
            cv2.cvtColor(arrays["images"][-1], cv2.COLOR_RGB2BGR),
        )
        print(json.dumps(report, indent=2))
    finally:
        if writer is not None:
            writer.release()
        env.renderer.close()


if __name__ == "__main__":
    main()
