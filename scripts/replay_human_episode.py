"""Replay human movement labels through physics, without a learned policy."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from convert_human_episode import (
    COLORS, VALIDATION_EPISODES, checksum, read_episode,
)


def observe(env):
    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "hand": env.get_ee_pos(),
        "cubes": env.get_all_cubes_state(),
        "hand_target": env.data.mocap_pos[env.mocap_id].copy(),
        "gripper_command": np.array(env.data.ctrl[env.act_ids[-1]]),
        "time_s": np.array(env.data.time),
    }


def describe_contact(env, contact, action_index, normal_force):
    """Describe a solver contact without modifying the simulation."""
    geometries = [int(contact.geom1), int(contact.geom2)]
    return {
        "logged_after_step_time_s": float(env.data.time),
        "action_index": action_index,
        "geometries": [
            env.model.geom(index).name or f"geometry_{index}"
            for index in geometries
        ],
        "bodies": [
            env.model.body(int(env.model.geom_bodyid[index])).name
            for index in geometries
        ],
        "penetration_m": max(0.0, -float(contact.dist)),
        "normal_force_n": normal_force,
        "contact_position_m": contact.pos.tolist(),
        "gripper_command": float(env.data.ctrl[env.act_ids[-1]]),
        "hand_target_xyz_m": env.data.mocap_pos[env.mocap_id].tolist(),
        "post_step_cube_xyz_m": env.get_cube_state()[:3].tolist(),
        "post_step_qpos": env.data.qpos.tolist(),
        "post_step_qvel": env.data.qvel.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--contact-trace", action="store_true",
        help="Save target-cube contacts and snapshots of peak events.",
    )
    args = parser.parse_args()
    if args.episode in VALIDATION_EPISODES:
        parser.error("Use a training episode for this development check.")
    lab = args.lab.expanduser().resolve()
    raw, actions, source = read_episode(lab, args.episode)
    color = COLORS[int(raw["state_goal"][0].argmax())]
    target_index = COLORS.index(color)

    sys.path.insert(0, str(lab))
    from hw3.sim_env import SO100MulticubeSimEnv

    args.out.mkdir(parents=True, exist_ok=False)
    scene = lab / "so101_gym/assets/so100_multicube_ee.xml"
    env = SO100MulticubeSimEnv(
        scene, goal_cube=color, seed=0, cube_pos_std=0,
        shuffle_cubes=False, control_hz=source["control_hz"],
        render_w=320, render_h=240,
    )
    writer = None
    try:
        # Restore the starting configuration only.
        env.data.qpos[env.qpos_idx] = raw["state_joints"][0]
        for cube_color, indices in zip(COLORS, env.cube_qpos_slices):
            env.data.qpos[indices] = raw[f"pos_cube_{cube_color}"][0]
        env.model.body("bin").pos[:] = (
            raw["goal_pos"][0] - env.model.site("bin_center").pos
        )
        env.data.qvel[:] = 0
        env.set_targets(raw["state_joints"][0])
        env.set_gripper(float(raw["state_gripper"][0, 0]))
        env.set_mocap_pose(raw["state_ee"][0, :3], raw["state_ee"][0, 3:])
        mujoco.mj_forward(env.model, env.data)

        writer = cv2.VideoWriter(
            str(args.out / "rollout.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            source["control_hz"], (320, 240),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the replay video.")
        states = [observe(env)]
        writer.write(cv2.cvtColor(env.render_rgb("angle"), cv2.COLOR_RGB2BGR))
        executed, physics = [], []
        contacts = []
        peak_penetration_contact = None
        peak_force_contact = None
        terminated = None
        target_geom = env.model.geom(f"{color}_box").id
        robot_root = int(env.model.body_rootid[env.model.body("Fixed_Jaw").id])
        bad_warnings = [
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        ]
        initial_warnings = sum(int(env.data.warning[w].number) for w in bad_warnings)

        for action in actions:
            # This matches the lab's XYZ-delta plus native gripper executor.
            target = env.data.mocap_pos[env.mocap_id] + action[:3]
            env.set_mocap_pos(target)
            env.set_gripper(float(action[3]))
            for _ in range(env.substeps):
                mujoco.mj_step(env.model, env.data)
                target_depth, self_depth, normal_force = 0.0, 0.0, 0.0
                for index, contact in enumerate(env.data.contact):
                    depth = max(0.0, -float(contact.dist))
                    if target_geom in (contact.geom1, contact.geom2):
                        target_depth = max(target_depth, depth)
                        force = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, index, force)
                        contact_force = abs(float(force[0]))
                        normal_force = max(normal_force, contact_force)
                        if args.contact_trace:
                            other = (
                                int(contact.geom2) if contact.geom1 == target_geom
                                else int(contact.geom1)
                            )
                            contacts.append([
                                env.data.time, len(executed), other, depth, contact_force,
                            ])
                            if (
                                peak_penetration_contact is None
                                or depth > peak_penetration_contact["penetration_m"]
                            ):
                                peak_penetration_contact = describe_contact(
                                    env, contact, len(executed), contact_force
                                )
                            if (
                                peak_force_contact is None
                                or contact_force > peak_force_contact["normal_force_n"]
                            ):
                                peak_force_contact = describe_contact(
                                    env, contact, len(executed), contact_force
                                )
                    body1 = int(env.model.geom_bodyid[contact.geom1])
                    body2 = int(env.model.geom_bodyid[contact.geom2])
                    if (
                        body1 != body2
                        and env.model.body_rootid[body1] == robot_root
                        and env.model.body_rootid[body2] == robot_root
                    ):
                        self_depth = max(self_depth, depth)
                physics.append([
                    env.data.time, env.get_cube_state()[2],
                    target_depth, self_depth, normal_force,
                ])
                warning_count = sum(
                    int(env.data.warning[w].number) for w in bad_warnings
                )
                if warning_count > initial_warnings:
                    terminated = "numerical_warning"
                    break

            executed.append(action.copy())
            states.append(observe(env))
            finite = all(
                np.isfinite(value).all() for value in states[-1].values()
            )
            if not finite:
                terminated = "nonfinite_state"
                break
            writer.write(cv2.cvtColor(
                env.render_rgb("angle"), cv2.COLOR_RGB2BGR
            ))
            cube = env.get_cube_state()[:3]
            if (
                cube[2] < -0.01 or abs(cube[0]) > 0.6
                or not 0.1 <= cube[1] <= 1.1
            ):
                terminated = "cube_out_of_bounds"
            if terminated is not None:
                break

        arrays = {
            key: np.stack([state[key] for state in states])
            for key in states[0]
        }
        physics = np.asarray(physics)
        cubes = arrays["cubes"].reshape(-1, 3, 7)
        final_cube = cubes[-1, target_index, :3]
        goal = raw["goal_pos"][0]
        separation = float(np.linalg.norm(arrays["hand"][-1] - final_cube))
        stable = len(states) >= 6 and np.linalg.norm(
            final_cube - cubes[-6, target_index, :3]
        ) < 0.005
        distractors = [i for i in range(3) if i != target_index]
        distractor_motion = float(np.linalg.norm(
            cubes[:, distractors, :3] - cubes[0, distractors, :3], axis=-1
        ).max())
        peak_height = float(physics[:, 1].max())
        target_depth = float(physics[:, 2].max())
        self_depth = float(physics[:, 3].max())
        task_success = bool(
            terminated is None
            and peak_height > 0.06
            and np.max(np.abs(final_cube[:2] - goal[:2])) < 0.04
            and 0.015 < final_cube[2] < 0.05
            and separation > 0.07 and stable
            and arrays["gripper_command"][-1] > 0.4
        )
        hand_errors = np.linalg.norm(
            arrays["hand"] - raw["state_ee"][:len(states), :3], axis=1
        )
        np.savez_compressed(
            args.out / "rollout.npz", **arrays, actions=np.asarray(executed)
        )
        np.savez_compressed(
            args.out / "physics.npz", samples=physics,
            columns=np.array([
                "time_s", "target_height_m", "target_penetration_m",
                "robot_self_penetration_m", "max_target_contact_normal_force_n",
            ]),
        )
        if args.contact_trace:
            np.savez_compressed(
                args.out / "contacts.npz",
                samples=np.asarray(contacts).reshape(-1, 5),
                columns=np.array([
                    "logged_after_step_time_s", "action_index",
                    "other_geometry_id", "penetration_m", "normal_force_n",
                ]),
                geometry_names=np.array([
                    env.model.geom(index).name or f"geometry_{index}"
                    for index in range(env.model.ngeom)
                ]),
            )
            contact_report = {
                "peak_penetration": peak_penetration_contact,
                "peak_normal_force": peak_force_contact,
                "contact_rows": len(contacts),
                "snapshot_note": "Solver contact logged after mj_step; saved "
                                 "qpos/qvel are post-step values.",
            }
            (args.out / "contacts.json").write_text(
                json.dumps(contact_report, indent=2) + "\n"
            )
        report = {
            "purpose": "One training-episode action replay; no policy evaluation.",
            "contact_trace_enabled": args.contact_trace,
            "contact_trace_sha256": checksum(args.out / "contacts.json") if args.contact_trace else None,
            "source": source, "split": "train",
            "instruction": f"Put the {color} cube in the bin.",
            "planned_actions": len(actions), "executed_actions": len(executed),
            "termination": terminated, "task_success": task_success,
            "quality_pass": bool(
                terminated is None and target_depth <= 0.002
                and self_depth == 0.0 and distractor_motion < 0.01
            ),
            "peak_target_height_m": peak_height,
            "final_target_xyz_m": final_cube.tolist(),
            "goal_xyz_m": goal.tolist(),
            "final_hand_cube_distance_m": separation,
            "final_gripper_command": float(arrays["gripper_command"][-1]),
            "stable_final_cube": bool(stable),
            "max_target_penetration_m": target_depth,
            "max_robot_self_penetration_m": self_depth,
            "max_distractor_displacement_m": distractor_motion,
            "peak_target_contact_normal_force_n": float(physics[:, 4].max()),
            "mean_hand_path_error_m": float(hand_errors.mean()),
            "max_hand_path_error_m": float(hand_errors.max()),
            "initialization": "Recorded initial poses; all unknown velocities set to zero",
            "executor": "Accumulate measured XYZ deltas into hand target; native instantaneous gripper target",
            "orientation": "Initial measured quaternion held fixed; no later pose restoration",
            "extra_settling_steps": 0,
            "control_dt_s": env.dt_ctrl, "physics_dt_s": env.sim_dt,
            "jaw_torque_range_nm": env.model.actuator_forcerange[env.act_ids[-1]].tolist(),
            "mujoco_version": mujoco.__version__,
            "scene_sha256": checksum(scene),
            "simulator_sha256": checksum(lab / "hw3/sim_env.py"),
            "script_sha256": checksum(Path(__file__)),
            "rollout_sha256": checksum(args.out / "rollout.npz"),
            "accepted_for_training": False,
            "limitation": "Original commands and velocities are missing. "
                          "Contact-force limits remain unvalidated. "
                          "One replay is not a policy success rate.",
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        if writer is not None:
            writer.release()
        env.renderer.close()


if __name__ == "__main__":
    main()
