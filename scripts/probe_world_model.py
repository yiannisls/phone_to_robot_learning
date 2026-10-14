"""Test alternative actions from identical simulator states, without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from generate_demo import observe
from world_model import WorldModel, load_episodes, make_state, predict_next

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
HORIZON = 5  # Five 0.1-second commands.
SHIFT_M = 0.002
VARIANTS = ["nominal", "x_plus", "x_minus", "y_plus", "y_minus",
            "z_plus", "z_minus", "flip_gripper"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_anchors(phases):
    """Choose moments by phase, before looking at prediction errors."""
    grasp = np.flatnonzero(phases == "grasp")
    transport = np.flatnonzero(phases == "recorded_transport")
    placement = np.flatnonzero(phases == "gentle_place")
    release = np.flatnonzero(phases == "release")
    return {
        "start_grasp": int(grasp[0]),
        "end_grasp": int(grasp[-1]),
        "early_transport": int(transport[min(3, len(transport) - 1)]),
        "middle_transport": int(transport[len(transport) // 2]),
        "placement": int(placement[0] if len(placement) else transport[-1]),
        "release": int(release[0]),
    }


def execute(env, action, close_rate):
    """Apply the same target increments and closing rule as dataset v2."""
    env.set_mocap_pos(env.data.mocap_pos[env.mocap_id] + action[:3])
    requested = float(action[3])
    for _ in range(env.substeps):
        applied = float(env.data.ctrl[env.act_ids[-1]])
        if requested < applied:
            applied = max(requested, applied - close_rate * env.sim_dt)
        else:
            applied = requested
        env.set_gripper(applied)
        mujoco.mj_step(env.model, env.data)
    return make_state(observe(env))


def action_sequences(recorded):
    sequences = np.repeat(recorded[None], len(VARIANTS), axis=0)
    for axis in range(3):
        sequences[1 + 2 * axis, 0, axis] += SHIFT_M
        sequences[2 + 2 * axis, 0, axis] -= SHIFT_M
    sequences[-1, 0, 3] = -0.174 if recorded[0, 3] > 0.4 else 1.0
    # Subsequent increments remain unchanged, so the target shift persists.
    # Only the first gripper request is flipped; subsequent requests resume.
    return sequences


def summarize(rows, actual, predicted):
    """Measure endpoint errors and changes relative to the nominal action."""
    result = {}
    for step in (1, HORIZON):
        measured = actual[:, :, step]
        estimated = predicted[:, :, step]
        groups = {"nominal": [0], "position_changes": list(range(1, 7)),
                  "gripper_flip": [7]}
        summary = {}
        for name, indices in groups.items():
            difference = estimated[:, indices] - measured[:, indices]
            summary[name] = {
                "branches": len(rows) * len(indices),
                "cube_mae_mm": float(np.abs(difference[..., 6:9]).mean() * 1000),
                "hand_mae_mm": float(np.abs(difference[..., 25:28]).mean() * 1000),
                "gripper_angle_mae_rad": float(np.abs(difference[..., 5]).mean()),
            }
            if name != "nominal":
                real_effect = measured[:, indices] - measured[:, [0]]
                predicted_effect = estimated[:, indices] - estimated[:, [0]]
                effect_error = predicted_effect - real_effect
                summary[name].update({
                    "cube_effect_error_mm": float(np.abs(effect_error[..., 6:9]).mean() * 1000),
                    "ignore_action_change_cube_error_mm": float(np.abs(real_effect[..., 6:9]).mean() * 1000),
                    "gripper_effect_error_rad": float(np.abs(effect_error[..., 5]).mean()),
                    "ignore_action_change_gripper_error_rad": float(np.abs(real_effect[..., 5]).mean()),
                })
        result[f"{step}_step"] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_FOLDER / "data/demonstrations_v2")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_FOLDER / "checkpoints/world_model_v1/world_model.pt")
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "results/world_model_v1_action_probes")
    args = parser.parse_args()
    homework = Path(os.environ["ROBOT_LEARNING_HW3"])
    sys.path.insert(0, str(homework))
    from hw3.sim_env import SO100SimEnv

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = WorldModel()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    episodes = []
    for split in ("train", "validation"):
        loaded, _ = load_episodes(args.data, split, checkpoint["profile"])
        episodes.extend((path, data) for path, data in loaded if "_aug" not in path.stem)

    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "profile": checkpoint["profile"], "horizon_steps": HORIZON,
        "position_shift_m": SHIFT_M, "variants": VARIANTS,
        "source_sha256": {str(path): sha256(path) for path, _ in episodes},
        "code_sha256": {name: sha256(Path(__file__).parent / name)
                        for name in ("probe_world_model.py", "world_model.py", "generate_demo.py")},
        "test_source_used": False,
        "anchors": {path.stem: choose_anchors(data["phase"]) for path, data in episodes},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows, all_actual, all_predicted, all_actions = [], [], [], []

    for path, episode in episodes:
        env = SO100SimEnv(homework / "so101_gym/assets/so100_transfer_cube_obstacle_ee.xml",
                         seed=0, cube_pos_std=0, obstacle_pos_std=0)
        try:
            for name in ("obstacle_wall", "upper_obstacle"):
                geom = env.model.geom(name)
                geom.contype = geom.conaffinity = 0
                geom.rgba[3] = 0
            env.model.body("bin").pos[:] = [-0.23, 0.57, 0.001]
            env.reset()
            env.data.qpos[env.cube_qpos_idx[:2]] += episode["cube_offset_xy_m"]
            mujoco.mj_forward(env.model, env.data)
            profile = checkpoint["profile"]
            assert profile == {
                "gripper_close_rate_rad_s": float(episode["gripper_close_rate_rad_s"]),
                "control_dt_s": env.dt_ctrl, "physics_dt_s": env.sim_dt,
                "jaw_torque_range_nm": env.model.actuator_forcerange[env.act_ids[-1]].tolist(),
                "mujoco_version": mujoco.__version__,
            }
            rate = profile["gripper_close_rate_rad_s"]
            saved_states = make_state(episode)
            anchors = {index: name for name, index in choose_anchors(episode["phase"]).items()}
            snapshot = mujoco.MjData(env.model)

            for index, action in enumerate(episode["actions"]):
                if index in anchors:
                    initial = make_state(observe(env))
                    # Copy ALL simulator data, including contact solver state.
                    mujoco.mj_copyData(snapshot, env.model, env.data)
                    sequences = action_sequences(episode["actions"][index:index + HORIZON])
                    assert sequences.shape == (len(VARIANTS), HORIZON, 4)
                    actual = np.empty((len(VARIANTS), HORIZON + 1, len(initial)), np.float32)
                    actual[:, 0] = initial
                    for branch, sequence in enumerate(sequences):
                        mujoco.mj_copyData(env.data, env.model, snapshot)
                        assert np.array_equal(make_state(observe(env)), initial)
                        for step, command in enumerate(sequence):
                            actual[branch, step + 1] = execute(env, command, rate)

                    # Repeating the nominal branch must reproduce every state.
                    mujoco.mj_copyData(env.data, env.model, snapshot)
                    repeated = np.stack([execute(env, command, rate) for command in sequences[0]])
                    assert np.array_equal(repeated, actual[0, 1:]), "Branch restoration changed the rollout"
                    mujoco.mj_copyData(env.data, env.model, snapshot)

                    with torch.no_grad():
                        state = torch.from_numpy(actual[:, 0].copy())
                        predicted = [state.numpy().copy()]
                        for step in range(HORIZON):
                            state = predict_next(model, checkpoint, state, torch.from_numpy(sequences[:, step]))
                            predicted.append(state.numpy().copy())
                    predicted = np.stack(predicted, axis=1)
                    assert np.isfinite(actual).all() and np.isfinite(predicted).all()
                    assert np.allclose(predicted[..., 28:35], actual[..., 28:35], atol=1e-6)
                    rows.append({
                        "source": path.stem, "split": episode["split"].item(),
                        "anchor": anchors[index], "action_index": index,
                        "replay_cube_max_error_mm": float(np.abs(initial[6:9] - saved_states[index, 6:9]).max() * 1000),
                        "replay_hand_max_error_mm": float(np.abs(initial[25:28] - saved_states[index, 25:28]).max() * 1000),
                    })
                    all_actual.append(actual)
                    all_predicted.append(predicted)
                    all_actions.append(sequences)
                execute(env, action, rate)
            print(f"Completed {path.stem}: {len(anchors)} states x {len(VARIANTS)} action choices", flush=True)
        finally:
            env.renderer.close()

    actual, predicted = np.stack(all_actual), np.stack(all_predicted)
    np.savez_compressed(args.out / "branches.npz", actual=actual, predicted=predicted,
                        actions=np.stack(all_actions), variants=np.array(VARIANTS),
                        source=np.array([row["source"] for row in rows]),
                        split=np.array([row["split"] for row in rows]),
                        anchor=np.array([row["anchor"] for row in rows]))
    report = {"anchors": rows, "summary": {}, "by_phase": {},
              "identical_state_restore_and_nominal_repeat": "passed",
              "limitations": "Four base episodes, six anchors each. Validation only for development; no test source, training or model-based control in this experiment."}
    for split in ("train", "validation"):
        indices = [i for i, row in enumerate(rows) if row["split"] == split]
        report["summary"][split] = summarize([rows[i] for i in indices], actual[indices], predicted[indices])
        report["by_phase"][split] = {}
        for phase in dict.fromkeys(row["anchor"] for row in rows):
            selected = [i for i in indices if rows[i]["anchor"] == phase]
            report["by_phase"][split][phase] = summarize([rows[i] for i in selected], actual[selected], predicted[selected])
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
