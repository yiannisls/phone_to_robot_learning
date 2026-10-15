"""Run a saved SmolVLA adapter in the human multicube simulator.

The first rollout starts from a training recording for development.
Only the initial poses are restored; later motion comes from the VLA.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
import torch

from convert_human_episode import COLORS, checksum, read_visual_dataset
from replay_human_episode import observe
from test_vla_memory import POLICY_REVISION, BACKBONE_REVISION, load_policy

MAX_STEPS = 400
EXECUTION_HORIZON = 4
MAX_MOVEMENT = 0.012  # Metres per 0.1-second control step.


def load_trained_policy(checkpoint, cache, example):
    report = json.loads((checkpoint / "report.json").read_text())
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    if (
        report["status"] != "completed"
        or report["policy_revision"] != POLICY_REVISION
        or report["backbone_revision"] != BACKBONE_REVISION
        or checksum(adapter) != report["adapter_sha256"]
        or checksum(checkpoint / "normalization.json") != report["normalization_sha256"]
        or checksum(Path(__file__).with_name("test_vla_memory.py")) != report["model_loader_sha256"]
    ):
        raise ValueError("Checkpoint or pinned model setup differs from training.")
    model, config, camera = load_policy(cache, example)
    saved = load_file(str(adapter))
    set_peft_model_state_dict(model, saved)
    restored = get_peft_model_state_dict(model)
    if set(saved) != set(restored) or not all(
        torch.equal(value.cpu(), saved[key]) for key, value in restored.items()
    ):
        raise ValueError("Adapter weights did not restore exactly.")
    model.eval()
    model.get_base_model().reset()
    return model, config, camera, report


def make_observation(image, robot_state, normalizer, language, camera):
    stats = normalizer["stats"]["observation.state"]
    state = (robot_state - np.array(stats["mean"])) / np.array(stats["std"])
    return {
        camera: (torch.from_numpy(image).permute(2, 0, 1).float() / 255).unsqueeze(0).cuda(),
        "observation.state": torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0),
        "observation.language.tokens": language["input_ids"].cuda(),
        "observation.language.attention_mask": language["attention_mask"].bool().cuda(),
    }


@torch.no_grad()
def predict_actions(model, batch, normalizer):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = model.get_base_model().predict_action_chunk(batch)
    stats = normalizer["stats"]["action"]
    return predicted[0].float().cpu().numpy() * np.array(stats["std"]) + np.array(stats["mean"])


def contact_measurements(env, target_geom, robot_root):
    target_depth, self_depth, normal_force = 0.0, 0.0, 0.0
    for index, contact in enumerate(env.data.contact):
        depth = max(0.0, -float(contact.dist))
        if target_geom in (contact.geom1, contact.geom2):
            target_depth = max(target_depth, depth)
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, index, force)
            normal_force = max(normal_force, abs(float(force[0])))
        body1 = int(env.model.geom_bodyid[contact.geom1])
        body2 = int(env.model.geom_bodyid[contact.geom2])
        if (
            body1 != body2 and env.model.body_rootid[body1] == robot_root
            and env.model.body_rootid[body2] == robot_root
        ):
            self_depth = max(self_depth, depth)
    return target_depth, self_depth, normal_force


def run_rollout(env, model, config, camera, normalizer, instruction, camera_name, args):
    tokenizer = model.get_base_model().model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    language = tokenizer(
        [instruction.strip() + "\n"], padding="max_length", truncation=True,
        max_length=config.tokenizer_max_length, return_tensors="pt",
    )
    target_index = COLORS.index(env.goal_cube)
    target_geom = env.model.geom(f"{env.goal_cube}_box").id
    robot_root = int(env.model.body_rootid[env.model.body("Fixed_Jaw").id])
    jaw_range = env.model.actuator_ctrlrange[env.act_ids[-1]]
    warnings = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC]
    initial_warnings = sum(int(env.data.warning[w].number) for w in warnings)
    states = [observe(env)]
    robot_states = [np.r_[env.get_joint_angles(), env.get_ee_pos()]]
    images = [env.render_rgb(camera_name).copy()]
    requested, executed, physics, chunks, prediction_steps, latencies = [], [], [], [], [], []
    clipped = []
    queue = deque()
    termination = None
    writer = cv2.VideoWriter(
        str(args.out / "rollout.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
        1 / env.dt_ctrl, (images[0].shape[1], images[0].shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create the rollout video.")
    try:
        writer.write(cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        for step in range(args.max_steps):
            if not queue:
                batch = make_observation(images[-1], robot_states[-1], normalizer, language, camera)
                torch.cuda.synchronize()
                start = time.perf_counter()
                chunk = predict_actions(model, batch, normalizer)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)
                if chunk.shape != (config.chunk_size, 4) or not np.isfinite(chunk).all():
                    termination = "invalid_prediction"
                    break
                chunks.append(chunk.copy())
                prediction_steps.append(step)
                queue.extend(chunk[:args.execution_horizon])

            action = queue.popleft().copy()
            command = action.copy()
            length = np.linalg.norm(command[:3])
            if length > MAX_MOVEMENT:
                command[:3] *= MAX_MOVEMENT / length
            command[3] = np.clip(command[3], *jaw_range)
            clipped.append([length > MAX_MOVEMENT, command[3] != action[3]])
            env.set_mocap_pos(env.data.mocap_pos[env.mocap_id] + command[:3])
            env.set_gripper(float(command[3]))

            for _ in range(env.substeps):
                mujoco.mj_step(env.model, env.data)
                if not (np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()):
                    termination = "nonfinite_state"
                    break
                physics.append([
                    env.data.time, env.get_cube_state()[2],
                    *contact_measurements(env, target_geom, robot_root),
                ])
                if sum(int(env.data.warning[w].number) for w in warnings) > initial_warnings:
                    termination = "numerical_warning"
                    break
            requested.append(action)
            executed.append(command)
            states.append(observe(env))
            robot_states.append(np.r_[env.get_joint_angles(), env.get_ee_pos()])
            if termination is not None:
                break
            images.append(env.render_rgb(camera_name).copy())
            writer.write(cv2.cvtColor(images[-1], cv2.COLOR_RGB2BGR))
            cubes = env.get_all_cubes_state().reshape(3, 7)[:, :3]
            if np.any((cubes[:, 2] < -0.01) | (np.abs(cubes[:, 0]) > 0.6)
                      | (cubes[:, 1] < 0.1) | (cubes[:, 1] > 1.1)):
                termination = "cube_out_of_bounds"
                break
            if (step + 1) % 40 == 0:
                print(f"Step {step + 1}/{args.max_steps}", flush=True)
    finally:
        writer.release()

    arrays = {key: np.stack([state[key] for state in states]) for key in states[0]}
    physics = np.asarray(physics).reshape(-1, 5)
    cubes = arrays["cubes"].reshape(-1, 3, 7)
    final_cube = cubes[-1, target_index, :3]
    stable = len(states) >= 6 and np.linalg.norm(final_cube - cubes[-6, target_index, :3]) < 0.005
    separation = float(np.linalg.norm(arrays["hand"][-1] - final_cube))
    distractors = [i for i in range(3) if i != target_index]
    distractor_motion = float(np.linalg.norm(cubes[:, distractors, :3] - cubes[0, distractors, :3], axis=-1).max())
    peak_height = float(physics[:, 1].max()) if len(physics) else float(cubes[0, target_index, 2])
    target_depth = float(physics[:, 2].max()) if len(physics) else None
    self_depth = float(physics[:, 3].max()) if len(physics) else None
    task_success = bool(
        termination is None and peak_height > 0.06
        and np.max(np.abs(final_cube[:2] - env.get_goal_pos()[:2])) < 0.04
        and 0.015 < final_cube[2] < 0.05 and separation > 0.07 and stable
        and arrays["gripper_command"][-1] > 0.4
    )
    np.savez_compressed(
        args.out / "rollout.npz", **arrays, robot_state=np.asarray(robot_states),
        images=np.asarray(images), requested_actions=np.asarray(requested).reshape(-1, 4),
        executed_actions=np.asarray(executed).reshape(-1, 4),
        predicted_chunks=np.asarray(chunks).reshape(-1, config.chunk_size, 4),
        prediction_steps=np.asarray(prediction_steps),
        inference_seconds=np.asarray(latencies), clipping=np.asarray(clipped).reshape(-1, 2),
    )
    np.savez_compressed(args.out / "physics.npz", samples=physics, columns=np.array([
        "time_s", "target_height_m", "target_penetration_m",
        "robot_self_penetration_m", "max_target_contact_normal_force_n",
    ]))
    return {
        "task_success": task_success,
        "quality_pass": bool(termination is None and target_depth is not None
                             and target_depth <= 0.002 and self_depth == 0 and distractor_motion < 0.01),
        "termination": termination or "step_limit", "executed_steps": len(executed),
        "simulated_seconds": float(env.data.time), "prediction_calls": len(chunks),
        "inference_seconds_mean": float(np.mean(latencies)),
        "inference_seconds_max": float(np.max(latencies)),
        "peak_target_height_m": peak_height, "final_target_xyz_m": final_cube.tolist(),
        "final_hand_cube_distance_m": separation, "stable_final_cube": bool(stable),
        "final_gripper_command": float(arrays["gripper_command"][-1]),
        "max_target_penetration_m": target_depth, "max_robot_self_penetration_m": self_depth,
        "max_distractor_displacement_m": distractor_motion,
        "peak_target_contact_normal_force_n": float(physics[:, 4].max()) if len(physics) else None,
        "movement_clips": int(np.sum(np.asarray(clipped).reshape(-1, 2)[:, 0])),
        "gripper_clips": int(np.sum(np.asarray(clipped).reshape(-1, 2)[:, 1])),
        "rollout_sha256": checksum(args.out / "rollout.npz"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--execution-horizon", type=int, default=EXECUTION_HORIZON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("work/vla-cache/hub"))
    args = parser.parse_args()
    if args.max_steps < 1 or not 1 <= args.execution_horizon <= EXECUTION_HORIZON:
        parser.error("Use positive steps and an execution horizon from 1 to 4.")
    root = args.dataset.expanduser().resolve()
    lab = args.lab.expanduser().resolve()
    manifest, normalizer = read_visual_dataset(root)
    if args.episode not in manifest["training_episode_ids"]:
        parser.error("This development rollout must start from a training episode.")
    entry = manifest["episodes"][args.episode]
    with np.load(root / entry["folder"] / "episode.npz", allow_pickle=False) as data:
        initial = {key: data[key][0].copy() for key in [
            "state_joints", "state_gripper", "state_ee", "goal_pos",
            *[f"pos_cube_{color}" for color in COLORS],
        ]}
        example = {"images": data["images"][:1], "robot_state": data["robot_state"][:1]}
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires CUDA with BF16.")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model, config, camera, training = load_trained_policy(args.checkpoint, args.cache, example)
    if (checksum(root / "manifest.json") != training["manifest_sha256"]
            or checksum(root / "normalization.json") != training["normalization_sha256"]):
        raise ValueError("Dataset differs from the trained adapter's corpus.")

    sys.path.insert(0, str(lab))
    from hw3.sim_env import SO100MulticubeSimEnv
    color = entry["instruction"].split()[2]
    scene = lab / "so101_gym/assets/so100_multicube_ee.xml"
    env = SO100MulticubeSimEnv(scene, goal_cube=color, seed=args.seed,
                              cube_pos_std=0, shuffle_cubes=False, control_hz=10,
                              render_w=320, render_h=240)
    try:
        # Initialize once. The VLA receives live observations after this point.
        env.data.qpos[env.qpos_idx] = initial["state_joints"]
        for cube_color, indices in zip(COLORS, env.cube_qpos_slices):
            env.data.qpos[indices] = initial[f"pos_cube_{cube_color}"]
        env.model.body("bin").pos[:] = initial["goal_pos"] - env.model.site("bin_center").pos
        env.data.qvel[:] = 0
        env.set_targets(initial["state_joints"])
        env.set_gripper(float(initial["state_gripper"][0]))
        env.set_mocap_pose(initial["state_ee"][:3], initial["state_ee"][3:])
        mujoco.mj_forward(env.model, env.data)
        report = run_rollout(env, model, config, camera, normalizer, entry["instruction"], manifest["camera"], args)
        report.update({
            "purpose": "First VLA development rollout; one training start, not a success-rate benchmark.",
            "controller": "SmolVLA with human-only LoRA adapter", "episode": args.episode,
            "split": "train", "instruction": entry["instruction"], "seed": args.seed,
            "camera": manifest["camera"], "max_steps": args.max_steps,
            "execution_horizon": args.execution_horizon, "action_chunk": config.chunk_size,
            "max_movement_m": MAX_MOVEMENT, "control_dt_s": env.dt_ctrl,
            "physics_dt_s": env.sim_dt,
            "jaw_torque_range_nm": env.model.actuator_forcerange[env.act_ids[-1]].tolist(),
            "jaw_command_range_rad": env.model.actuator_ctrlrange[env.act_ids[-1]].tolist(),
            "executor": "Accumulate predicted achieved-motion deltas into mocap target; instant gripper target",
            "orientation": "Initial measured quaternion held fixed",
            "initialization": "Recorded initial poses; unknown velocities set to zero",
            "physics_advances_during_inference": False,
            "scripted_grasp_or_placement": False, "world_model_used": False,
            "phone_data_included_in_training": False,
            "mujoco_version": mujoco.__version__,
            "adapter_sha256": training["adapter_sha256"],
            "manifest_sha256": training["manifest_sha256"],
            "normalization_sha256": training["normalization_sha256"],
            "initial_episode_sha256": entry["episode_sha256"],
            "scene_sha256": checksum(scene), "simulator_sha256": checksum(lab / "hw3/sim_env.py"),
            "script_sha256": checksum(Path(__file__)),
            "limitation": "Small 200-update pilot; original command labels and contact physics remain imperfect."
        })
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.renderer.close()


if __name__ == "__main__":
    main()
