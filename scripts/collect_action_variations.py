"""Collect alternative-action outcomes from training demonstrations only."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from generate_demo import observe
from probe_world_model import HORIZON, VARIANTS, action_sequences, choose_anchors, execute, sha256
from world_model import load_episodes, make_state

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_FOLDER / "data/demonstrations_v2")
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "data/action_variations_v1")
    args = parser.parse_args()
    homework = Path(os.environ["ROBOT_LEARNING_HW3"])
    sys.path.insert(0, str(homework))
    from hw3.sim_env import SO100SimEnv

    # The loader filters source families before opening episode files.
    episodes, profile = load_episodes(args.data, "train")
    args.out.mkdir(parents=True, exist_ok=False)
    names = VARIANTS + ["flip_gripper_3_steps", "flip_gripper_5_steps"]
    manifest = {
        "complete": False, "split": "train", "profile": profile,
        "variants": names, "horizon_steps": HORIZON,
        "position_shift_m": 0.002,
        "source_sha256": {path.name: sha256(path) for path, _ in episodes},
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                        ("collect_action_variations.py", "probe_world_model.py",
                         "generate_demo.py", "world_model.py")},
        "anchors": {path.stem: choose_anchors(ep["phase"]) for path, ep in episodes},
        "files": {}, "branches": 0, "transitions": 0,
        "validation_or_test_used": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for number, (path, episode) in enumerate(episodes, start=1):
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
            assert profile["mujoco_version"] == mujoco.__version__
            assert profile["control_dt_s"] == env.dt_ctrl
            assert profile["physics_dt_s"] == env.sim_dt
            assert profile["jaw_torque_range_nm"] == env.model.actuator_forcerange[env.act_ids[-1]].tolist()
            rate = profile["gripper_close_rate_rad_s"]
            snapshot = mujoco.MjData(env.model)
            anchors = {index: name for name, index in choose_anchors(episode["phase"]).items()}
            states, commands, labels, variants = [], [], [], []

            for index, action in enumerate(episode["actions"]):
                if index in anchors:
                    mujoco.mj_copyData(snapshot, env.model, env.data)
                    initial = make_state(observe(env))
                    recorded = episode["actions"][index:index + HORIZON]
                    sequences = list(action_sequences(recorded))
                    # Teach the consequences of longer opening/closing requests too.
                    for duration in (3, 5):
                        sequence = recorded.copy()
                        sequence[:duration, 3] = -0.174 if recorded[0, 3] > 0.4 else 1.0
                        sequences.append(sequence)
                    for name, sequence in zip(names, sequences):
                        mujoco.mj_copyData(env.data, env.model, snapshot)
                        assert np.array_equal(make_state(observe(env)), initial)
                        trajectory = [initial]
                        trajectory.extend(execute(env, command, rate) for command in sequence)
                        states.append(np.stack(trajectory))
                        commands.append(sequence)
                        labels.append(anchors[index])
                        variants.append(name)
                    mujoco.mj_copyData(env.data, env.model, snapshot)
                execute(env, action, rate)

            states, commands = np.stack(states), np.stack(commands)
            assert states.shape == (60, HORIZON + 1, 35)
            assert commands.shape == (60, HORIZON, 4)
            assert np.isfinite(states).all() and np.isfinite(commands).all()
            assert np.allclose(np.diff(states[..., 28:31], axis=1), commands[..., :3], atol=1e-6)
            output = args.out / path.name
            np.savez_compressed(output, states=states, actions=commands,
                                source_video=episode["source_video"], split=np.array("train"),
                                anchor=np.array(labels), variant=np.array(variants))
            manifest["files"][output.name] = sha256(output)
            manifest["branches"] += len(states)
            manifest["transitions"] += len(states) * HORIZON
            print(f"{number}/{len(episodes)}: {path.stem}, {len(states) * HORIZON} transitions", flush=True)
        finally:
            env.renderer.close()

    manifest["complete"] = True
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {manifest['transitions']} training transitions in {args.out}", flush=True)


if __name__ == "__main__":
    main()
