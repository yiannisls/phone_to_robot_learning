"""Run the learned policy on the validation layout or a supplied cube offset."""
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
import torch

from generate_demo import PROJECT_FOLDER, observe
from policy import HORIZON, Policy, make_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_FOLDER / "checkpoints/bc_v1/policy.pt")
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "results/bc_v1_validation")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--offset-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--label", default="validation")
    parser.add_argument("--jaw-torque-limit", type=float, help="Symmetric jaw actuator torque limit in N m.")
    parser.add_argument("--gripper-close-rate", type=float, default=2.0, help="Maximum closing target rate in rad/s.")
    args = parser.parse_args()
    for value in (args.jaw_torque_limit, args.gripper_close_rate):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error("Torque and closing-rate limits must be finite and positive.")
    if "ROBOT_LEARNING_HW3" not in os.environ:
        parser.error("Set ROBOT_LEARNING_HW3 to your homework folder.")
    homework = Path(os.environ["ROBOT_LEARNING_HW3"])
    sys.path.insert(0, str(homework))
    from hw3.sim_env import SO100SimEnv

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["horizon"] == HORIZON
    model = Policy(checkpoint["input_size"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset_report = json.loads((PROJECT_FOLDER / "results/dataset_v1.json").read_text())
    offset = args.offset_xy
    if offset is None:
        offset = dataset_report["base_offsets_xy_m"]["IMG_5542"]
    if not np.isfinite(offset).all():
        parser.error("Cube offsets must be finite numbers.")
    args.out.mkdir(parents=True, exist_ok=False)

    env = SO100SimEnv(
        homework / "so101_gym/assets/so100_transfer_cube_obstacle_ee.xml",
        seed=0, cube_pos_std=0, obstacle_pos_std=0,
    )
    for name in ("obstacle_wall", "upper_obstacle"):
        geometry = env.model.geom(name)
        geometry.contype = geometry.conaffinity = 0
        geometry.rgba[3] = 0
    env.model.body("bin").pos[:] = [-0.23, 0.57, 0.001]
    jaw_actuator = env.act_ids[-1]
    if args.jaw_torque_limit is not None:
        env.model.actuator_forcelimited[jaw_actuator] = True
        env.model.actuator_forcerange[jaw_actuator] = [-args.jaw_torque_limit, args.jaw_torque_limit]
    env.reset()
    env.data.qpos[env.cube_qpos_idx[:2]] += offset
    mujoco.mj_forward(env.model, env.data)

    writer = None
    try:
        if args.video:
            writer = cv2.VideoWriter(
                str(args.out / "validation.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                10, (env.render_w, env.render_h),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the evaluation video.")

        states = [observe(env)]
        actions, physics = [], []
        queue = []
        peak_height, max_penetration = 0.0, 0.0
        held_success = 0
        task_success, out_of_bounds = False, False
        cube_geometry = env.model.geom("red_box").id
        jaw_bodies = [env.model.body(name).id for name in ("Fixed_Jaw", "Moving_Jaw")]

        for step in range(240):
            if not queue:
                features = torch.from_numpy(make_features(states[-1]))
                features = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
                with torch.no_grad():
                    prediction = model(features[None])[0].numpy()
                base = env.data.mocap_pos[env.mocap_id].copy()
                targets = base + prediction[:4, :3] * checkpoint["position_scale"].numpy()
                grippers = np.where(prediction[:4, 3] > 0, 1.0, -0.174)
                # Execute four predictions, then measure again and replan.
                queue = list(zip(targets, grippers))

            target, gripper_goal = queue.pop(0)
            previous_target = env.data.mocap_pos[env.mocap_id].copy()
            target = np.clip(target, [-0.45, 0.30, 0.025], [-0.02, 0.72, 0.32])
            delta = target - previous_target
            distance = np.linalg.norm(delta)
            if distance > 0.012:
                delta *= 0.012 / distance
            target = previous_target + delta
            env.set_mocap_pos(target)

            for _ in range(env.substeps):
                applied = float(env.data.ctrl[env.act_ids[-1]])
                applied = max(gripper_goal, applied - args.gripper_close_rate * env.sim_dt) if gripper_goal < applied else gripper_goal
                env.set_gripper(applied)
                mujoco.mj_step(env.model, env.data)
                peak_height = max(peak_height, float(env.get_cube_state()[2]))
                jaw_forces = [0.0, 0.0]
                penetration = 0.0
                for index, contact in enumerate(env.data.contact):
                    if cube_geometry not in (contact.geom1, contact.geom2):
                        continue
                    penetration = max(penetration, -float(contact.dist))
                    other = contact.geom2 if contact.geom1 == cube_geometry else contact.geom1
                    body = int(env.model.geom_bodyid[other])
                    if body in jaw_bodies:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, index, force)
                        jaw_forces[jaw_bodies.index(body)] += max(0.0, float(force[0]))
                max_penetration = max(max_penetration, penetration)
                physics.append([env.data.time, *jaw_forces, penetration])

            actions.append(np.r_[delta, gripper_goal])
            state = observe(env)
            states.append(state)
            cube = state["state_cube"][:3]
            hand = state["state_ee_full"][:3]
            goal = state["goal_pos"]
            stable = len(states) >= 6 and np.linalg.norm(cube - states[-6]["state_cube"][:3]) < 0.005
            done = (
                np.max(np.abs(cube[:2] - goal[:2])) < 0.04 and cube[2] < 0.05
                and peak_height > 0.06 and stable
                and np.linalg.norm(hand - cube) > 0.07
                and state["state_gripper_command"][0] > 0.4
            )
            held_success = held_success + 1 if done else 0
            out_of_bounds = bool(
                not np.isfinite(env.data.qpos).all() or cube[2] < -0.03
                or abs(cube[0]) > 0.6 or cube[1] < 0.1 or cube[1] > 0.85
            )
            if writer is not None:
                image = cv2.cvtColor(env.render_rgb(), cv2.COLOR_RGB2BGR)
                cv2.putText(image, f"Learned BC | {args.label}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                writer.write(image)
            if held_success >= 5:
                task_success = True
                break
            if out_of_bounds:
                break

        arrays = {key: np.stack([state[key] for state in states]) for key in states[0]}
        np.savez_compressed(args.out / "rollout.npz", **arrays, actions=np.asarray(actions))
        physics = np.asarray(physics)
        np.savez_compressed(
            args.out / "physics.npz", samples=physics,
            columns=np.array(["time_s", "fixed_jaw_normal_force_n", "moving_jaw_normal_force_n", "cube_penetration_m"]),
        )
        result = {
            "source": "IMG_5542" if args.offset_xy is None else None,
            "split": "validation" if args.offset_xy is None else "diagnostic",
            "label": args.label, "offset_xy_m": offset,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "task_success": task_success, "quality_pass": max_penetration <= 0.002,
            "steps": len(actions), "out_of_bounds": out_of_bounds,
            "peak_cube_height_m": peak_height, "final_cube_xyz": cube.tolist(),
            "final_hand_xyz": hand.tolist(), "final_gripper_command": float(applied),
            "max_cube_penetration_m": max_penetration,
            "peak_normal_force_per_jaw_n": physics[:, 1:3].max(axis=0).tolist(),
            "gripper_close_rate_rad_s": args.gripper_close_rate,
            "jaw_torque_range_nm": env.model.actuator_forcerange[jaw_actuator].tolist(),
            "mujoco_version": mujoco.__version__,
            "execution": "four learned targets per observation; 12 mm step limit; workspace clipping",
            "quality_note": "Contact-depth check only; force peaks are reported separately.",
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.release()
        env.renderer.close()


if __name__ == "__main__":
    main()
