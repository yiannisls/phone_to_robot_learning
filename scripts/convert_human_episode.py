"""Convert human multicube recordings into aligned visual examples."""

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
import zarr

COLORS = ("red", "green", "blue")
JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
VALIDATION_EPISODES = (36, 9, 32)  # Existing lab split; zero-based IDs.


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_episode(lab, episode):
    """Follow the original merge order, excluding separate DAgger stores."""
    processed = zarr.open_group(
        str(lab / "datasets/processed/multi_cube/processed_ee_xyz.zarr"),
        mode="r",
    )
    ends = np.asarray(processed["meta/episode_ends"][:], dtype=int)
    if len(ends) != 42 or processed.attrs["num_dagger_episodes"] != 0:
        raise ValueError("Expected the original 42-episode teleoperation dataset.")
    if not 0 <= episode < len(ends):
        raise ValueError("Choose an episode from 0 to 41.")

    offset = 0
    selected = None
    for relative in processed.attrs["source_zarrs"]:
        path = (lab / relative).resolve()
        if not path.is_relative_to(lab):
            raise ValueError("A source store is outside the configured lab.")
        root = zarr.open_group(str(path), mode="r")
        raw_ends = np.asarray(root["meta/episode_ends"][:], dtype=int)
        if offset <= episode < offset + len(raw_ends):
            local = episode - offset
            start = int(raw_ends[local - 1]) if local else 0
            stop = int(raw_ends[local])
            selected = path, root, local, start, stop
        offset += len(raw_ends)
    if offset != len(ends) or selected is None:
        raise ValueError("Raw and processed episode counts disagree.")

    path, root, local, start, stop = selected
    expected = {
        "state_joints": 6, "state_ee": 7, "state_gripper": 1,
        "action_gripper": 1, "state_goal": 3, "goal_pos": 3,
        **{f"pos_cube_{color}": 7 for color in COLORS},
    }
    raw = {}
    for name, width in expected.items():
        array = root[f"data/{name}"]
        if array.shape != (int(root["meta/episode_ends"][-1]), width):
            raise ValueError(f"Unexpected shape for {name}.")
        raw[name] = np.asarray(array[start:stop])
        if not np.isfinite(raw[name]).all():
            raise ValueError(f"Non-finite values in {name}.")
    if stop - start < 2:
        raise ValueError("The episode has no transitions.")
    if tuple(root.attrs["joint_names"]) != JOINTS:
        raise ValueError("Unexpected robot joint order.")
    if tuple(root.attrs["cube_colors"]) != COLORS:
        raise ValueError("Unexpected colour order.")

    goal = raw["state_goal"]
    if not (
        np.isin(goal, [0, 1]).all()
        and np.all(goal.sum(axis=1) == 1)
        and np.all(goal == goal[0])
    ):
        raise ValueError("Expected one constant colour instruction per episode.")
    if not np.all(raw["goal_pos"] == raw["goal_pos"][0]):
        raise ValueError("The bin moves within this episode.")

    actions = np.column_stack([
        np.diff(raw["state_ee"][:, :3], axis=0),
        raw["action_gripper"][:-1],
    ])
    pstart = int(ends[episode - 1]) if episode else 0
    pstop = int(ends[episode])
    saved_actions = np.column_stack([
        processed["data/action_ee_xyz"][pstart:pstop],
        processed["data/action_gripper"][pstart:pstop],
    ])
    if not np.array_equal(actions, saved_actions):
        raise ValueError("Reconstructed actions differ from the lab labels.")
    hz = float(root.attrs["control_hz"])
    if not np.isclose(hz, 10.0):
        raise ValueError("This conversion expects the recorded 10 Hz format.")

    source = {
        "store": path.relative_to(lab).as_posix(),
        "local_episode": local,
        "global_episode": episode,
        "raw_rows": [start, stop],
        "processed_rows": [pstart, pstop],
        "control_hz": hz,
        "array_sha256": {
            key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in raw.items()
        },
    }
    return raw, actions, source


def convert_episode(lab, episode, camera, out):
    """Reconstruct one recording without stepping physics."""
    raw, actions, source = read_episode(lab, episode)
    split = "validation" if episode in VALIDATION_EPISODES else "train"
    color = COLORS[int(raw["state_goal"][0].argmax())]
    instruction = f"Put the {color} cube in the bin."
    count = len(raw["state_joints"])
    robot_state = np.column_stack([
        raw["state_joints"], raw["state_ee"][:, :3],
    ])
    time_s = np.arange(count) / source["control_hz"]

    sys.path.insert(0, str(lab))
    from hw3.sim_env import SO100MulticubeSimEnv

    out.mkdir(parents=True, exist_ok=False)
    scene = lab / "so101_gym/assets/so100_multicube_ee.xml"
    env = SO100MulticubeSimEnv(
        scene, goal_cube=color, seed=0, cube_pos_std=0,
        shuffle_cubes=False, render_w=320, render_h=240,
    )
    writer = None
    frames, errors = [], []
    try:
        writer = cv2.VideoWriter(
            str(out / "episode.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            source["control_hz"], (320, 240),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the preview video.")
        for row in range(count):
            env.data.qpos[env.qpos_idx] = raw["state_joints"][row]
            for cube_color, indices in zip(COLORS, env.cube_qpos_slices):
                env.data.qpos[indices] = raw[f"pos_cube_{cube_color}"][row]
            env.model.body("bin").pos[:] = (
                raw["goal_pos"][row] - env.model.site("bin_center").pos
            )
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            errors.append(float(np.linalg.norm(
                env.get_ee_pos() - raw["state_ee"][row, :3]
            )))
            image = env.render_rgb(camera).copy()
            frames.append(image)
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if max(errors) > 0.001:
            raise ValueError("Reconstructed hand positions differ by over 1 mm.")
    finally:
        if writer is not None:
            writer.release()
        env.renderer.close()

    images = np.stack(frames)
    np.savez_compressed(
        out / "episode.npz",
        images=images, robot_state=robot_state, actions=actions, time_s=time_s,
        instruction=np.array(instruction), split=np.array(split),
        source_kind=np.array("human_teleoperation"),
        source_episode=np.array(episode), **raw,
    )
    peak = int(raw[f"pos_cube_{color}"][:, 2].argmax())
    preview = np.hstack([images[0], images[peak], images[-1]])
    cv2.imwrite(
        str(out / "preview.jpg"),
        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
    )
    report = {
        "schema": "human_visual_v1",
        "source": source, "split": split, "instruction": instruction,
        "source_kind": "human_teleoperation",
        "accepted_for_training": False,
        "observations": count, "actions": len(actions),
        "image_shape": list(images.shape), "camera": camera,
        "robot_state_fields": ["6 measured joint angles, including jaw",
                               "3 measured hand coordinates"],
        "action_fields": ["3 next-minus-current measured hand coordinates",
                          "1 recorded native gripper command"],
        "alignment": "image/state[t], action[t], image/state[t+1]",
        "time_source": "Nominal row index / control_hz; no original timestamps",
        "images_reconstructed": True, "physics_rollout_performed": False,
        "contact_quality_evaluated": False,
        "max_reconstructed_hand_error_m": max(errors),
        "actions_match_processed_lab_labels": True,
        "validation_episode_ids_zero_based": list(VALIDATION_EPISODES),
        "normalization_fitted": False,
        "mujoco_version": mujoco.__version__,
        "scene_sha256": checksum(scene),
        "simulator_sha256": checksum(lab / "hw3/sim_env.py"),
        "converter_sha256": checksum(Path(__file__)),
        "episode_sha256": checksum(out / "episode.npz"),
        "limitation": "Original commands and velocities are absent; "
                      "this conversion is not a dynamic replay or policy score.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def convert_dataset(lab, camera, out):
    """Convert the original episodes and fit statistics on training rows."""
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    training_states = []
    training_actions = []

    for episode in range(42):
        folder = out / f"episode_{episode:03d}"
        report = convert_episode(lab, episode, camera, folder)
        is_training = report["split"] == "train"

        report["accepted_for_training"] = is_training
        report["dataset_use"] = "Experimental supervised fine-tuning"
        report["acceptance_basis"] = (
            "Original human recording with checked alignment; "
            "contact physics is not certified."
        )
        (folder / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        entries.append({
            "episode": episode,
            "split": report["split"],
            "folder": folder.name,
            "instruction": report["instruction"],
            "observations": report["observations"],
            "actions": report["actions"],
            "episode_sha256": report["episode_sha256"],
            "summary_sha256": checksum(folder / "summary.json"),
        })

        if is_training:
            with np.load(folder / "episode.npz", allow_pickle=False) as data:
                training_states.append(data["robot_state"][:-1].copy())
                training_actions.append(data["actions"].copy())

        print(f"Converted {episode + 1}/42: {report['split']}", flush=True)

    stats = {}
    for key, parts in [
        ("observation.state", training_states),
        ("action", training_actions),
    ]:
        values = np.concatenate(parts).astype(np.float64)
        stats[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
        }

    training_ids = [
        entry["episode"] for entry in entries if entry["split"] == "train"
    ]
    normalization = {
        "purpose": "Human corpus normalization; training transitions only",
        "episode_ids": training_ids,
        "transitions": sum(len(part) for part in training_actions),
        "terminal_observations_excluded": True,
        "minimum_std": 1e-6,
        "stats": stats,
    }
    (out / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n"
    )

    manifest = {
        "schema": "human_visual_dataset_v1",
        "source_kind": "human_teleoperation",
        "camera": camera,
        "training_episode_ids": training_ids,
        "validation_episode_ids": sorted(VALIDATION_EPISODES),
        "episodes": entries,
        "normalization_sha256": checksum(out / "normalization.json"),
        "images_reconstructed": True,
        "contact_physics_certified": False,
        "dagger_included": False,
        "limitation": (
            "XYZ labels are measured displacements, not original commands. "
            "No policy success rate is established by this conversion."
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episode", type=int)
    selection.add_argument("--all", action="store_true",
                           help="Prepare all 42 episodes with the existing split.")
    parser.add_argument("--camera", choices=("angle", "top", "left_wrist"),
                        default="angle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lab = args.lab.expanduser().resolve()
    if args.all:
        report = convert_dataset(lab, args.camera, args.out)
    else:
        episode = 0 if args.episode is None else args.episode
        report = convert_episode(lab, episode, args.camera, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
