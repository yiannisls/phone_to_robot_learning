import hashlib
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
SOURCE_SPLITS = {
    "IMG_5539": "train",
    "IMG_5540": "train",
    "IMG_5541": "train",
    "IMG_5542": "validation",
    "IMG_5543": "test",
}


def observe(env):
    observation = env.get_obs()
    return {
        "state_joints": observation["joints"].copy(),
        "state_ee_full": observation["ee"].copy(),
        "state_cube": observation["cube"].copy(),
        "state_gripper": observation["gripper"].copy(),
        "goal_pos": observation["goal_pos"].copy(),
        "state_mocap_target": env.data.mocap_pos[env.mocap_id].copy(),
        "state_gripper_command": np.array([env.data.ctrl[env.act_ids[-1]]]),
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
    }


def generate_demo(source="IMG_5539", name=None, offset_xy=(0, 0), video=True,
                  version="v1", gripper_close_rate=2.0):
    if not version.isidentifier():
        raise ValueError("Use a simple version name, such as v2.")
    if not np.isfinite(gripper_close_rate) or gripper_close_rate <= 0:
        raise ValueError("Closing rate must be finite and positive.")
    name = name or source
    split = SOURCE_SPLITS[source]
    if split != "train" and name != source:
        raise ValueError("Create augmented examples from training sources only.")

    if "ROBOT_LEARNING_HW3" not in os.environ:
        raise SystemExit("Set ROBOT_LEARNING_HW3 to your HW3 project folder.")

    homework_folder = Path(os.environ["ROBOT_LEARNING_HW3"])
    sys.path.insert(0, str(homework_folder))
    from hw3.sim_env import SO100SimEnv

    transport_path = PROJECT_FOLDER / "data" / "processed" / f"{source}_transport.csv"
    transport = np.genfromtxt(transport_path, delimiter=",", names=True)

    data_folder = PROJECT_FOLDER / "data" / f"demonstrations_{version}"
    video_folder = PROJECT_FOLDER / "results" / f"demonstrations_{version}"
    output_path = data_folder / f"{name}.npz"
    video_path = video_folder / f"{name}.avi"

    if output_path.exists() or (video and video_path.exists()):
        raise FileExistsError(f"Outputs for {name} already exist. Choose a new name.")

    data_folder.mkdir(parents=True, exist_ok=True)
    video_folder.mkdir(parents=True, exist_ok=True)

    env = SO100SimEnv(
        homework_folder / "so101_gym/assets/so100_transfer_cube_obstacle_ee.xml",
        seed=0, cube_pos_std=0, obstacle_pos_std=0
    )

    for obstacle in ("obstacle_wall", "upper_obstacle"):
        geometry = env.model.geom(obstacle)
        geometry.contype = 0
        geometry.conaffinity = 0
        geometry.rgba[3] = 0

    env.model.body("bin").pos[:] = [-0.23, 0.57, 0.001]
    env.reset()
    env.data.qpos[env.cube_qpos_idx[:2]] += np.asarray(offset_xy)
    mujoco.mj_forward(env.model, env.data)

    writer = None
    try:
        if video:
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                10, (env.render_w, env.render_h)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the demonstration video.")

        states = [observe(env)]
        actions = []
        phases = []
        peak_height = float(env.get_cube_state()[2])
        max_penetration = 0.0
        cube_geometry = env.model.geom("red_box").id

        def execute(target, gripper_goal, phase):
            nonlocal peak_height, max_penetration
            previous_target = env.data.mocap_pos[env.mocap_id].copy()
            env.set_mocap_pos(target)

            # Limit closing to the selected rate; opening is immediate.
            for _ in range(env.substeps):
                applied = float(env.data.ctrl[env.act_ids[-1]])
                if gripper_goal < applied:
                    applied = max(gripper_goal, applied - gripper_close_rate * env.sim_dt)
                else:
                    applied = gripper_goal

                env.set_gripper(applied)
                mujoco.mj_step(env.model, env.data)
                peak_height = max(peak_height, float(env.get_cube_state()[2]))

                for contact in env.data.contact:
                    if cube_geometry in (contact.geom1, contact.geom2):
                        max_penetration = max(max_penetration, -float(contact.dist))

            # Actions store the requested gripper goal; observations store
            # the applied command and measured jaw position.
            actions.append(np.r_[target - previous_target, gripper_goal])
            phases.append(phase)
            states.append(observe(env))

            if writer is not None:
                writer.write(cv2.cvtColor(env.render_rgb(), cv2.COLOR_RGB2BGR))

        def move_between(start, end, steps, gripper_goal, phase):
            for fraction in np.linspace(0, 1, steps):
                execute(start + fraction * (end - start), gripper_goal, phase)

        cube_start = env.get_cube_state()[:3].copy()
        goal = env.get_goal_pos().copy()
        grasp = cube_start + [-0.025, 0, 0.04]
        above_grasp = grasp + [0, 0, 0.10]

        move_between(env.get_ee_pos(), above_grasp, 25, 1.0, "approach")
        move_between(above_grasp, grasp, 25, 1.0, "lower")
        move_between(grasp, grasp, 15, -0.174, "grasp")

        displacement = goal[:2] - cube_start[:2]
        direction = displacement / np.linalg.norm(displacement)
        sideways = np.array([-direction[1], direction[0]])

        ready_to_place = False
        for point in transport[1:]:
            target = grasp.copy()
            target[:2] += point["progress"] * displacement + point["lateral_m"] * sideways
            target[2] += point["height_m"]
            execute(target, -0.174, "recorded_transport")

            cube = env.get_cube_state()[:3]
            ready_to_place = (
                peak_height > 0.06
                and np.max(np.abs(cube[:2] - goal[:2])) < 0.03
                and cube[2] < 0.045
            )
            if ready_to_place:
                break

        if ready_to_place:
            target = env.data.mocap_pos[env.mocap_id].copy()
            for _ in range(12):
                cube_height = float(env.get_cube_state()[2])
                if cube_height <= 0.026:
                    break
                target[2] -= min(0.003, cube_height - 0.023)
                execute(target, -0.174, "gentle_place")

            move_between(target, target, 2, 1.0, "release")
            retreat = target + [0, 0, 0.14]
            move_between(target, retreat, 20, 1.0, "retreat")
            move_between(retreat, retreat, 10, 1.0, "settle")

        arrays = {
            key: np.stack([state[key] for state in states]).astype(np.float32)
            for key in states[0]
        }
        action_array = np.asarray(actions, dtype=np.float32)
        final_cube = arrays["state_cube"][-1, :3]
        final_hand = arrays["state_ee_full"][-1, :3]

        task_success = bool(
            ready_to_place
            and peak_height > 0.06
            and np.max(np.abs(final_cube[:2] - goal[:2])) < 0.04
            and final_cube[2] < 0.05
            and np.linalg.norm(final_hand - final_cube) > 0.07
            and np.linalg.norm(final_cube - arrays["state_cube"][-6, :3]) < 0.005
            and arrays["state_gripper_command"][-1, 0] > 0.4
        )
        quality_pass = max_penetration <= 0.002

        assert len(states) == len(actions) + 1
        assert np.allclose(
            np.diff(arrays["state_mocap_target"], axis=0),
            action_array[:, :3], atol=1e-6
        )

        np.savez_compressed(
            output_path, **arrays, actions=action_array,
            phase=np.array(phases), time_s=np.arange(len(states)) * env.dt_ctrl,
            source_video=np.array(source), split=np.array(split),
            success=np.array(task_success and quality_pass),
            task_success=np.array(task_success), quality_pass=np.array(quality_pass),
            max_cube_penetration_m=np.array(max_penetration),
            peak_cube_height_m=np.array(peak_height),
            cube_offset_xy_m=np.asarray(offset_xy),
            gripper_close_rate_rad_s=np.array(gripper_close_rate),
            dataset_version=np.array(version), control_dt_s=np.array(env.dt_ctrl),
            physics_dt_s=np.array(env.sim_dt),
            jaw_torque_range_nm=env.model.actuator_forcerange[env.act_ids[-1]].copy(),
            generator_sha256=np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            action_gripper_semantics=np.array("requested native actuator target"),
            mujoco_version=np.array(mujoco.__version__),
            source_sha256=np.array(hashlib.sha256(
                (PROJECT_FOLDER / "data" / "raw" / f"{source}.mp4").read_bytes()
            ).hexdigest()),
            transport_sha256=np.array(hashlib.sha256(transport_path.read_bytes()).hexdigest()),
        )

        print(f"Task success: {task_success}; quality check: {quality_pass}")
        print(f"Recorded {len(actions)} actions and {len(states)} states.")
        print(f"Maximum cube contact penetration: {max_penetration * 1000:.3f} mm")
        print(f"Saved {output_path}")
        if video:
            print(f"Video: {video_path}")

    finally:
        if writer is not None:
            writer.release()
        env.renderer.close()


if __name__ == "__main__":
    generate_demo()
