"""A small model of robot and cube motion after a commanded action."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

PHYSICAL_SIZE = 28  # qpos[13], qvel[12], hand position[3].
STATE_SIZE = 35     # Plus target[3], applied gripper command[1], goal[3].
SOURCE_SPLITS = {
    "IMG_5539": "train", "IMG_5540": "train", "IMG_5541": "train",
    "IMG_5542": "validation", "IMG_5543": "test",
}


def make_state(episode):
    return np.concatenate([
        episode["qpos"], episode["qvel"], episode["state_ee_full"][..., :3],
        episode["state_mocap_target"], episode["state_gripper_command"],
        episode["goal_pos"],
    ], axis=-1).astype(np.float32)


def load_episodes(folder, split, expected_profile=None):
    episodes = []
    profile = expected_profile
    for path in sorted(Path(folder).glob("*.npz")):
        source = path.stem.split("_aug")[0]
        if SOURCE_SPLITS[source] != split:
            continue
        with np.load(path, allow_pickle=False) as saved:
            episode = {key: saved[key].copy() for key in saved.files}
        assert episode["source_video"].item() == source
        assert episode["split"].item() == split
        current_profile = {
            "gripper_close_rate_rad_s": float(episode["gripper_close_rate_rad_s"]),
            "control_dt_s": float(episode["control_dt_s"]),
            "physics_dt_s": float(episode["physics_dt_s"]),
            "jaw_torque_range_nm": episode["jaw_torque_range_nm"].tolist(),
            "mujoco_version": episode["mujoco_version"].item(),
        }
        if profile is None:
            profile = current_profile
        if profile != current_profile:
            raise ValueError(f"Mixed execution profiles in {path}")
        state = make_state(episode)
        actions = episode["actions"]
        assert state.shape == (len(actions) + 1, STATE_SIZE)
        assert actions.shape[1] == 4
        assert np.isfinite(state).all() and np.isfinite(actions).all()
        # Explicitly check the course simulator's joint/cube ordering.
        assert np.allclose(episode["qpos"][:, :6], episode["state_joints"])
        assert np.allclose(episode["qpos"][:, 6:13], episode["state_cube"])
        assert np.allclose(np.diff(state[:, 28:31], axis=0), actions[:, :3], atol=1e-6)
        episodes.append((path, episode))  # Failed runs also teach dynamics.
    if not episodes:
        raise ValueError(f"No {split} episodes in {folder}")
    return episodes, profile


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE + 4, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, PHYSICAL_SIZE),
        )

    def forward(self, inputs):
        return self.network(inputs)


def predict_next(model, checkpoint, state, action):
    inputs = torch.cat([state, action], dim=-1)
    inputs = (inputs - checkpoint["input_mean"]) / checkpoint["input_std"]
    delta = model(inputs) * checkpoint["delta_scale"]
    physical = state[..., :PHYSICAL_SIZE] + delta
    # Keep the cube's predicted orientation quaternion at unit length.
    quaternion = physical[..., 9:13]
    norm = quaternion.norm(dim=-1, keepdim=True)
    quaternion = torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), state[..., 9:13])
    physical = torch.cat([physical[..., :9], quaternion, physical[..., 13:]], dim=-1)

    # Controller bookkeeping is known; only physical response is learned.
    target = state[..., 28:31] + action[..., :3]
    applied = state[..., 31:32]
    requested = action[..., 3:4]
    profile = checkpoint["profile"]
    closing_step = profile["gripper_close_rate_rad_s"] * profile["control_dt_s"]
    gripper = torch.where(requested < applied,
                          torch.maximum(requested, applied - closing_step), requested)
    return torch.cat([physical, target, gripper, state[..., 32:35]], dim=-1)
