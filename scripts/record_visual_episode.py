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



def place_with_feedback(env, execute):
    """Use simulator measurements to place a demonstration cube gently."""
    cube_geometry = env.model.geom(f"{env.goal_cube}_box").id
    floor_geometry = env.model.geom("bin_floor").id
    goal = env.get_goal_pos()
    closed = float(env.data.ctrl[env.act_ids[-1]])

    def bottom_height():
        rotation = env.data.geom_xmat[cube_geometry].reshape(3, 3)
        half_height = np.abs(rotation[2]) @ env.model.geom_size[cube_geometry]
        return float(env.data.geom_xpos[cube_geometry, 2] - half_height)

    floor_top = (
        env.data.geom_xpos[floor_geometry, 2]
        + env.model.geom_size[floor_geometry, 2]
    )

    # Clear the rim while correcting the horizontal position.
    rim_top = max(
        env.data.geom_xpos[env.model.geom(name).id, 2] + env.model.geom(name).size[2]
        for name in ("bin_wall", "bin_wall2", "bin_wall3", "bin_wall4")
    )
    lift_distance = max(0.0, rim_top + 0.015 - bottom_height())
    start = env.data.mocap_pos[env.mocap_id].copy()
    for fraction in np.linspace(0, 1, 21)[1:]:
        target = start + [0, 0, lift_distance * fraction]
        error = goal[:2] - env.get_cube_state()[:2]
        target[:2] = (
            env.data.mocap_pos[env.mocap_id, :2]
            + np.clip(error, -0.002, 0.002)
        )
        execute(target, closed, "placement_lift")

    for _ in range(60):
        error = goal[:2] - env.get_cube_state()[:2]
        if np.max(np.abs(error)) < 0.002:
            break
        target = env.data.mocap_pos[env.mocap_id].copy()
        target[:2] += np.clip(error, -0.002, 0.002)
        execute(target, closed, "placement_center")
    else:
        return False

    # Stop a few millimetres above the floor.
    # If the cube stops descending, stop the routine instead of pushing on.
    lowest_bottom = bottom_height()
    stalled_steps = 0
    for _ in range(120):
        clearance = bottom_height() - floor_top
        if clearance <= 0.003:
            break
        target = env.data.mocap_pos[env.mocap_id].copy()
        target[2] -= min(0.001, clearance - 0.003)
        error = goal[:2] - env.get_cube_state()[:2]
        target[:2] += np.clip(error, -0.001, 0.001)
        execute(target, closed, "placement_lower")
        current_bottom = bottom_height()
        if current_bottom < lowest_bottom - 0.0002:
            lowest_bottom = current_bottom
            stalled_steps = 0
        else:
            stalled_steps += 1
        if stalled_steps >= 15:
            return False
    else:
        return False

    # Release gradually while holding the hand target still.
    target = env.data.mocap_pos[env.mocap_id].copy()
    for opening in np.linspace(closed, 1.0, 13)[1:]:
        execute(target, float(opening), "placement_release")

    for fraction in np.linspace(0, 1, 21)[1:]:
        execute(target + [0, 0, 0.14 * fraction], 1.0, "retreat")
    for _ in range(15):
        execute(target + [0, 0, 0.14], 1.0, "settle")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    training_sources = [name for name, split in SOURCE_SPLITS.items() if split == "train"]
    parser.add_argument("--source", choices=training_sources, default="IMG_5539")
    parser.add_argument("--color", choices=COLORS, default="green")
    parser.add_argument("--camera", default="angle")
    parser.add_argument("--placement", choices=("replay", "feedback"), default="replay")
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
        phases = []
        worst_contact = None
        close_rate = float(source["gripper_close_rate_rad_s"])

        def execute(target, requested_gripper, phase):
            nonlocal peak_height, max_penetration, worst_contact
            previous = env.data.mocap_pos[env.mocap_id].copy()
            actions.append(np.r_[target - previous, requested_gripper])
            phases.append(phase)
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
                    if target_geometry not in (contact.geom1, contact.geom2):
                        continue
                    depth = max(0.0, -float(contact.dist))
                    if depth > max_penetration:
                        max_penetration = depth
                        other = contact.geom2 if contact.geom1 == target_geometry else contact.geom1
                        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                        body = int(env.model.geom_bodyid[other])
                        worst_contact = {
                            "other_geometry": name,
                            "other_body": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body),
                            "phase": phase,
                            "time_s": float(env.data.time),
                            "action_index": len(actions) - 1,
                            "depth_m": depth,
                        }

            observation = capture(env, args.camera)
            observations.append(observation)
            writer.write(cv2.cvtColor(observation["images"], cv2.COLOR_RGB2BGR))

        preserved_steps = 0
        placement_completed = None
        placement_handoff = None
        rim_top = max(
            env.data.geom_xpos[env.model.geom(name).id, 2] + env.model.geom(name).size[2]
            for name in ("bin_wall", "bin_wall2", "bin_wall3", "bin_wall4")
        )
        for index, action in enumerate(source["actions"]):
            phase = str(source["phase"][index])
            rotation = env.data.geom_xmat[target_geometry].reshape(3, 3)
            bottom = (
                env.data.geom_xpos[target_geometry, 2]
                - np.abs(rotation[2]) @ env.model.geom_size[target_geometry]
            )
            near_bin = np.max(np.abs(
                env.get_cube_state()[:2] - env.get_goal_pos()[:2]
            )) < 0.05
            clear_approach = (
                phase == "recorded_transport"
                and near_bin
                and bottom > rim_top + 0.005
            )
            original_placement = phase in ("gentle_place", "release")
            if args.placement == "feedback" and (clear_approach or original_placement):
                placement_handoff = "clear_bin_approach" if clear_approach else "source_placement_phase"
                placement_completed = place_with_feedback(env, execute)
                break
            target = source["state_mocap_target"][index + 1].astype(np.float64)
            execute(target, float(action[3]), phase)
            preserved_steps += 1
        if args.placement == "feedback" and placement_completed is None:
            raise ValueError("The source has no placement phase to replace.")

        arrays = {
            key: np.stack([observation[key] for observation in observations])
            for key in observations[0]
        }
        actions = np.asarray(actions)
        assert len(arrays["images"]) == len(actions) + 1
        assert np.allclose(np.diff(arrays["time_s"]), env.dt_ctrl)
        assert np.allclose(np.diff(arrays["mocap_target"], axis=0), actions[:, :3])
        assert np.allclose(actions[:preserved_steps], source["actions"][:preserved_steps], atol=1e-6)
        assert np.array_equal(phases[:preserved_steps], source["phase"][:preserved_steps])
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
            (args.placement == "replay" or placement_completed)
            and peak_height > 0.06
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
            phase=np.asarray(phases),
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
            "placement_mode": args.placement,
            "placement_completed": placement_completed,
            "preserved_source_actions": preserved_steps,
            "source_prefix_unchanged": True,
            "placement_handoff": placement_handoff,
            "recorded_transport_actions_preserved": int(np.sum(source["phase"][:preserved_steps] == "recorded_transport")),
            "recorded_transport_actions_total": int(np.sum(source["phase"] == "recorded_transport")),
            "recorded_transport_tail_replaced": bool(np.any(source["phase"][preserved_steps:] == "recorded_transport")),
            "placement_teacher": "Simulator cube pose and geometry feedback" if args.placement == "feedback" else "Original saved commands",
            "worst_target_contact": worst_contact,
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
