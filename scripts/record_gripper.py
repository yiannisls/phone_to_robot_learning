"""Record keyboard teleoperation and jaw contact forces in the homework simulator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from generate_demo import observe

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
PHYSICS_COLUMNS = [
    "time_s", "requested_gripper_rad", "applied_gripper_rad", "jaw_angle_rad",
    "jaw_velocity_rad_s", "jaw_actuator_torque_nm", "fixed_jaw_normal_force_n",
    "moving_jaw_normal_force_n", "cube_penetration_m",
    "cube_x_m", "cube_y_m", "cube_z_m", "cube_vx_m_s", "cube_vy_m_s", "cube_vz_m_s",
    "hand_x_m", "hand_y_m", "hand_z_m",
    "cube_minus_hand_x_m", "cube_minus_hand_y_m", "cube_minus_hand_z_m",
]
# Keys change fixed simulation axes; use the top view to see their direction.
MOVEMENT_KEYS = {
    "a": (-1, 0, 0), "d": (1, 0, 0),
    "s": (0, -1, 0), "w": (0, 1, 0),
    "f": (0, 0, -1), "r": (0, 0, 1),
}
EVENT_KEYS = {"b": "stable_hold", "n": "slip_or_drop",
              "t": "successful_transfer", "g": "failed_attempt"}


def create_environment(homework, offset_xy, torque_limit):
    sys.path.insert(0, str(homework))
    from hw3.sim_env import SO100SimEnv
    env = SO100SimEnv(homework / "so101_gym/assets/so100_transfer_cube_obstacle_ee.xml",
                         seed=0, cube_pos_std=0, obstacle_pos_std=0)
    for name in ("obstacle_wall", "upper_obstacle"):
        geometry = env.model.geom(name)
        geometry.contype = geometry.conaffinity = 0
        geometry.rgba[3] = 0
    env.model.body("bin").pos[:] = [-0.23, 0.57, 0.001]
    jaw = env.act_ids[-1]
    env.model.actuator_forcelimited[jaw] = True
    env.model.actuator_forcerange[jaw] = [-torque_limit, torque_limit]
    env.reset()
    env.data.qpos[env.cube_qpos_idx[:2]] += offset_xy
    mujoco.mj_forward(env.model, env.data)
    return env


class GripperRecorder:
    def __init__(self, env, close_rate):
        self.env, self.close_rate = env, close_rate
        self.states, self.actions, self.physics, self.events = [observe(env)], [], [], []
        self.jaw = env.act_ids[-1]
        self.jaw_qpos = int(env.qpos_idx[-1])
        self.jaw_dof = int(env.model.jnt_dofadr[env.model.joint("Jaw").id])
        self.cube = env.model.geom("red_box").id
        self.cube_dof = int(env.model.jnt_dofadr[env.model.joint("red_box_joint").id])
        self.jaw_bodies = [env.model.body(name).id for name in ("Fixed_Jaw", "Moving_Jaw")]
        self.last_force_peaks = np.zeros(2)
        self.state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.initial_state = np.empty(mujoco.mj_stateSize(env.model, self.state_signature))
        mujoco.mj_getState(env.model, env.data, self.initial_state, self.state_signature)

    def physics_sample(self, requested):
        env = self.env
        forces, penetration = np.zeros(2), 0.0
        for index, contact in enumerate(env.data.contact):
            if self.cube not in (contact.geom1, contact.geom2):
                continue
            penetration = max(penetration, -float(contact.dist))
            other = contact.geom2 if contact.geom1 == self.cube else contact.geom1
            body = int(env.model.geom_bodyid[other])
            if body in self.jaw_bodies:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, index, wrench)
                forces[self.jaw_bodies.index(body)] += max(0.0, float(wrench[0]))
        cube, hand = env.get_cube_state()[:3], env.get_ee_pos()
        return np.r_[env.data.time, requested, env.data.ctrl[self.jaw],
                     env.data.qpos[self.jaw_qpos], env.data.qvel[self.jaw_dof],
                     env.data.actuator_force[self.jaw], forces, penetration,
                     cube, env.data.qvel[self.cube_dof:self.cube_dof + 3], hand, cube - hand]

    def step(self, delta, requested):
        env = self.env
        previous = env.data.mocap_pos[env.mocap_id].copy()
        target = np.clip(previous + delta, [-0.45, 0.30, 0.025], [-0.02, 0.72, 0.32])
        delta = target - previous
        distance = np.linalg.norm(delta)
        if distance > 0.012:
            delta *= 0.012 / distance
        requested = float(np.clip(requested, -0.174, 1.0))
        env.set_mocap_pos(previous + delta)
        self.last_force_peaks[:] = 0
        for _ in range(env.substeps):
            applied = float(env.data.ctrl[self.jaw])
            applied = max(requested, applied - self.close_rate * env.sim_dt) if requested < applied else requested
            env.set_gripper(applied)
            mujoco.mj_step(env.model, env.data)
            sample = self.physics_sample(requested)
            if not np.isfinite(sample).all() or not np.isfinite(env.data.qpos).all():
                raise RuntimeError("Non-finite simulator state; stop and inspect this run")
            self.physics.append(sample)
            self.last_force_peaks = np.maximum(self.last_force_peaks, sample[6:8])
        self.actions.append(np.r_[delta, requested])
        self.states.append(observe(env))

    def mark(self, label):
        self.events.append({"label": label, "state_index": len(self.actions),
                            "time_s": float(self.env.data.time)})

    def save(self, out, origin, stop_reason, config, command_path=None):
        env = self.env
        states = {key: np.stack([state[key] for state in self.states]) for key in self.states[0]}
        actions = np.asarray(self.actions, dtype=np.float64).reshape(-1, 4)
        physics = np.asarray(self.physics, dtype=np.float64).reshape(-1, len(PHYSICS_COLUMNS))
        assert len(states["qpos"]) == len(actions) + 1
        assert len(physics) == len(actions) * env.substeps
        assert np.allclose(np.diff(states["state_mocap_target"], axis=0), actions[:, :3], atol=1e-9)
        np.savez_compressed(out / "episode.npz", **states, actions=actions,
                            initial_integration_state=self.initial_state,
                            state_signature=np.array(self.state_signature),
                            time_s=np.arange(len(states["qpos"])) * env.dt_ctrl,
                            split=np.array("unassigned"), origin=np.array(origin))
        np.savez_compressed(out / "physics.npz", samples=physics, columns=np.array(PHYSICS_COLUMNS))
        (out / "events.json").write_text(json.dumps(self.events, indent=2))
        forces = physics[:, 6:8]
        summary = {
            "origin": origin, "stop_reason": stop_reason, "steps": len(actions),
            "simulation_duration_s": len(actions) * env.dt_ctrl,
            "control_hz": 1 / env.dt_ctrl, "physics_hz": 1 / env.sim_dt,
            "gripper_close_rate_rad_s": self.close_rate,
            "jaw_torque_range_nm": env.model.actuator_forcerange[self.jaw].tolist(),
            "cube_mass_kg": float(env.model.body_mass[env.model.geom_bodyid[self.cube]]),
            "cube_friction": env.model.geom_friction[self.cube].tolist(),
            "jaw_contact_geometries": {
                name: {"friction": env.model.geom_friction[i].tolist(),
                       "solref": env.model.geom_solref[i].tolist(),
                       "solimp": env.model.geom_solimp[i].tolist()}
                for i in range(env.model.ngeom) if int(env.model.geom_bodyid[i]) in self.jaw_bodies
                and env.model.geom_contype[i] != 0
                for name in [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or str(i)]
            },
            "peak_normal_force_per_jaw_n": forces.max(axis=0).tolist() if len(forces) else [0, 0],
            "p95_normal_force_per_jaw_n_all_samples": np.percentile(forces, 95, axis=0).tolist() if len(forces) else [0, 0],
            "peak_abs_jaw_actuator_torque_nm": float(np.abs(physics[:, 5]).max()) if len(physics) else 0,
            "max_cube_penetration_m": float(physics[:, 8].max()) if len(physics) else 0,
            "max_cube_height_m": float(physics[:, 11].max()) if len(physics) else float(states["state_cube"][0, 2]),
            "events": self.events, "config": config, "split": "unassigned",
            "mujoco_version": mujoco.__version__,
            "action_gripper_semantics": "requested native actuator target; opening immediate, closing rate-limited",
            "command_source_sha256": hashlib.sha256(command_path.read_bytes()).hexdigest() if command_path else None,
            "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": "Normal contact loads are simulation measurements. Relative cube/hand motion and manual labels do not prove slip or an optimal grip force. No automatic training inclusion or success claim.",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Saved {len(actions)} actions and {len(physics)} force samples to {out}", flush=True)


def export_zarr(folder):
    """Export one complete recorded episode with lab-compatible array names."""
    import zarr
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "episode.npz", allow_pickle=False) as saved:
        episode = {key: saved[key].copy() for key in saved.files}
    with np.load(folder / "physics.npz", allow_pickle=False) as saved:
        samples, columns = saved["samples"].copy(), saved["columns"].tolist()
    count = len(episode["actions"])
    store = zarr.open_group(str(folder / "episode.zarr"), mode="w-", zarr_format=3)
    current, following = store.create_group("data"), store.create_group("next")

    def put(group, name, array):
        chunks = (max(1, min(len(array), 1024)),) + array.shape[1:]
        group.create_array(name, data=array, chunks=chunks)

    for name in ("state_joints", "state_ee_full", "state_cube", "state_gripper",
                 "state_gripper_command", "state_mocap_target", "qpos", "qvel", "goal_pos"):
        label = "state_ee" if name == "state_ee_full" else name
        put(current, label, episode[name][:-1])
        put(following, label, episode[name][1:])
    put(current, "actions", episode["actions"])
    put(current, "action_gripper", episode["actions"][:, 3:4])
    put(current, "state_obstacle", np.zeros((count, 3)))
    meta = store.create_group("meta")
    put(meta, "episode_ends", np.array([count], dtype=np.int64))
    put(meta, "initial_integration_state", episode["initial_integration_state"])
    meta.attrs["state_signature"] = int(episode["state_signature"])
    physics = store.create_group("physics")
    put(physics, "samples", samples)
    physics.attrs["columns"] = columns
    store.attrs.update({"schema": "phone_robot_gripper_v1", "summary": summary,
                        "origin": summary["origin"], "split": summary["split"],
                        "state_key": "state_ee", "action_key": "actions",
                        "control_hz": summary["control_hz"],
                        "alignment": "data[t], actions[t], next[t]; episode_ends count transitions",
                        "action_gripper_spec": "requested native actuator target, not measured jaw angle",
                        "state_obstacle_spec": "zeros for disabled obstacles",
                        "npz_sha256": hashlib.sha256((folder / "episode.npz").read_bytes()).hexdigest(),
                        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "complete": True})
    print(f"Saved {count} transitions to {folder / 'episode.zarr'}", flush=True)


def display_frame(recorder, paused=False):
    env = recorder.env
    views = np.concatenate([env.render("angle"), env.render("top")], axis=1)
    banner = np.zeros((130, views.shape[1], 3), dtype=np.uint8)
    forces = recorder.physics[-1][6:8] if recorder.physics else [0, 0]
    texts = [
        f"{'PAUSED: Space to start' if paused else 'RECORDING'} | t={env.data.time:.1f}s | angle view / top view",
        "A/D x-/+ | S/W y-/+ | F/R down/up | O open | C close | [/] fine close/open",
        "Space pause | B stable hold | N slip/drop | T transfer done | G failed attempt | Esc save and quit",
        f"Jaw normal loads N: {forces[0]:.1f} / {forces[1]:.1f} | last-step peaks: {recorder.last_force_peaks[0]:.1f} / {recorder.last_force_peaks[1]:.1f}",
        f"Jaw angle={env.get_gripper_angle():.3f} rad | applied target={env.data.ctrl[recorder.jaw]:.3f} | motor torque={env.data.actuator_force[recorder.jaw]:.3f} N m",
    ]
    for index, text in enumerate(texts):
        cv2.putText(banner, text, (10, 22 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (230, 230, 230), 1)
    return np.concatenate([banner, views])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New folder for one episode")
    parser.add_argument("--gripper-close-rate", type=float, default=1.0)
    parser.add_argument("--jaw-torque-limit", type=float, default=3.5)
    parser.add_argument("--offset-xy", type=float, nargs=2, default=[0, 0])
    parser.add_argument("--move-mm", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=120.0, help="Simulated time limit, excluding pause")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--zarr", action="store_true", help="Also save a lab-compatible episode.zarr store")
    parser.add_argument("--commands", type=Path, help="Execute saved demonstration actions for a scripted recorder check")
    parser.add_argument("--headless", action="store_true", help="Only valid with --commands")
    args = parser.parse_args()
    if "ROBOT_LEARNING_HW3" not in os.environ:
        parser.error("Set ROBOT_LEARNING_HW3 to your homework folder")
    if args.headless and args.commands is None:
        parser.error("--headless requires --commands")
    if not np.isfinite(args.offset_xy).all():
        parser.error("Cube offsets must be finite")
    for value in (args.gripper_close_rate, args.jaw_torque_limit, args.move_mm, args.max_seconds):
        if not np.isfinite(value) or value <= 0:
            parser.error("Rates, limits and step sizes must be finite and positive")
    commands = None
    if args.commands:
        with np.load(args.commands, allow_pickle=False) as saved:
            commands = saved["actions"].copy()
        if commands.ndim != 2 or commands.shape[1] != 4 or not np.isfinite(commands).all():
            parser.error("Saved commands must be a finite T-by-4 array")
    args.out.mkdir(parents=True, exist_ok=False)
    homework = Path(os.environ["ROBOT_LEARNING_HW3"])
    env = create_environment(homework, args.offset_xy, args.jaw_torque_limit)
    recorder = GripperRecorder(env, args.gripper_close_rate)
    paused, requested, writer = True, 1.0, None
    origin = "scripted_recorder_check" if commands is not None else "human_keyboard_teleoperation"
    stop_reason = "time_limit"
    config = {"offset_xy_m": args.offset_xy, "jaw_torque_limit_nm": args.jaw_torque_limit,
              "move_mm": args.move_mm, "max_seconds": args.max_seconds}
    try:
        if not args.headless:
            cv2.namedWindow("Gripper recording", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gripper recording", 1280, 610)
        if args.video:
            writer = cv2.VideoWriter(str(args.out / "recording.avi"),
                                    cv2.VideoWriter_fourcc(*"MJPG"), 1 / env.dt_ctrl,
                                    (2 * env.render_w, env.render_h + 130))
            if not writer.isOpened():
                raise RuntimeError("Cannot create recording video")
        while len(recorder.actions) * env.dt_ctrl < args.max_seconds:
            started = time.monotonic()
            delta = np.zeros(3)
            if commands is not None:
                if len(recorder.actions) == len(commands):
                    stop_reason = "scripted_commands_complete"
                    break
                delta, requested = commands[len(recorder.actions), :3], commands[len(recorder.actions), 3]
            else:
                cv2.imshow("Gripper recording", display_frame(recorder, paused))
                key = cv2.waitKey(20) & 0xFF
                if key == 27 or cv2.getWindowProperty("Gripper recording", cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "user_exit"
                    break
                char = chr(key).lower()
                if char == " ":
                    paused = not paused
                if char in EVENT_KEYS:
                    recorder.mark(EVENT_KEYS[char])
                if paused:
                    continue
                if char in MOVEMENT_KEYS:
                    delta = np.array(MOVEMENT_KEYS[char]) * args.move_mm / 1000
                if char == "o": requested = 1.0
                if char == "c": requested = -0.174
                if char == "[": requested = max(-0.174, requested - 0.05)
                if char == "]": requested = min(1.0, requested + 0.05)
            recorder.step(delta, requested)
            if writer is not None or (commands is not None and not args.headless):
                frame = display_frame(recorder)
                if writer is not None: writer.write(frame)
                if not args.headless:
                    cv2.imshow("Gripper recording", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        stop_reason = "user_exit"
                        break
            if not args.headless:
                time.sleep(max(0, env.dt_ctrl - (time.monotonic() - started)))
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        if writer is not None: writer.release()
        if not args.headless: cv2.destroyAllWindows()
        try:
            recorder.save(args.out, origin, stop_reason, config, args.commands)
            if args.zarr:
                export_zarr(args.out)
        finally:
            env.renderer.close()


if __name__ == "__main__":
    main()
