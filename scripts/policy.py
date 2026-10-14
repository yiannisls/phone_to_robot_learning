"""The measurements and neural network shared by training and evaluation."""
import numpy as np
from torch import nn

HORIZON = 8


def make_features(state):
    hand = state["state_ee_full"][..., :3]
    cube = state["state_cube"][..., :3]
    goal = state["goal_pos"]
    return np.concatenate([
        hand,
        state["state_mocap_target"] - hand,  # Controller tracking error.
        cube - hand,
        goal - cube,
        cube[..., 2:3], goal[..., 2:3],
        state["state_gripper"], state["state_gripper_command"],
        state["state_joints"], state["qvel"][..., :6],
    ], axis=-1).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, HORIZON * 4),
        )

    def forward(self, features):
        # Each future step has three position offsets and one gripper logit.
        return self.network(features).reshape(-1, HORIZON, 4)
