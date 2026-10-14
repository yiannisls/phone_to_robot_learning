"""Use the learned world model to compare three short gripper sequences."""
import numpy as np
import torch

from world_model import WorldModel, make_state, predict_next

HORIZON = 5
OVERRIDE_STEPS = 3
IMPROVEMENT_MARGIN_M = 0.005
OPEN, CLOSED = 1.0, -0.174
CHOICES = ['bc_sequence', 'open_three_steps', 'close_three_steps']


def settings():
    return {
        'horizon_steps': HORIZON, 'override_steps': OVERRIDE_STEPS,
        'improvement_margin_m': IMPROVEMENT_MARGIN_M,
        'choices': CHOICES, 'carry_height_m': 0.045,
        'placement_height_m': 0.035, 'bin_xy_tolerance_m': 0.04,
        'score': 'mean predicted cube distance to carry/placement reference, metres',
        'carry_reference': 'initial cube-hand offset plus nominal predicted hand position',
        'placement_reference': 'initial cube xy and bin height plus 20 mm cube half-size',
        'execution': 'execute only first chosen gripper request; observe and compare again',
        'force_prediction': False, 'controller_revision': 2,
        'placement_open_gate': 'Keep BC when the cube is in the bin, low, and the applied gripper command is open',
    }


def action_sequence(current_target, future_targets, grippers):
    """Apply the actual workspace and 12 mm movement limits before prediction."""
    previous = np.asarray(current_target).copy()
    actions = []
    for target, gripper in zip(future_targets[:HORIZON], grippers[:HORIZON]):
        target = np.clip(target, [-0.45, 0.30, 0.025], [-0.02, 0.72, 0.32])
        delta = target - previous
        distance = np.linalg.norm(delta)
        if distance > 0.012:
            delta *= 0.012 / distance
        actions.append(np.r_[delta, gripper])
        previous = previous + delta
    assert len(actions) == HORIZON
    return np.asarray(actions)


class GripperPlanner:
    def __init__(self, path, actual_profile):
        self.checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if self.checkpoint['profile'] != actual_profile:
            raise ValueError('World model and simulator execution profiles differ')
        self.model = WorldModel()
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()

    def choose(self, observation, nominal):
        requested = float(nominal[0,3])
        cube, hand, goal = (observation[key][:3] for key in
                            ('state_cube', 'state_ee_full', 'goal_pos'))
        if cube[2] > 0.045:
            purpose = 'carry'
        elif np.max(np.abs(cube[:2] - goal[:2])) < 0.04 and cube[2] < 0.035:
            if observation['state_gripper_command'][0] > 0.4:
                return requested, {'purpose': 'released', 'choice': CHOICES[0],
                                    'override': False, 'reason': 'placed cube and gripper already open'}, None
            purpose = 'placement'
        else:
            return requested, {'purpose': 'inactive', 'choice': CHOICES[0],
                                'override': False, 'reason': 'leave approach and grasp to BC'}, None

        candidates = np.repeat(nominal[None], len(CHOICES), axis=0)
        candidates[1,:OVERRIDE_STEPS,3] = OPEN
        candidates[2,:OVERRIDE_STEPS,3] = CLOSED
        with torch.inference_mode():
            state = torch.from_numpy(np.repeat(make_state(observation)[None], len(CHOICES), axis=0))
            forecasts = [state.numpy().copy()]
            for step in range(HORIZON):
                state = predict_next(self.model, self.checkpoint, state,
                                     torch.from_numpy(candidates[:,step].astype(np.float32)))
                forecasts.append(state.numpy().copy())
        predicted = np.stack(forecasts, axis=1)
        if purpose == 'carry':
            reference = predicted[0,1:,25:28] + (cube - hand)
        else:
            reference = np.repeat(np.r_[cube[:2], goal[2] + 0.02][None], HORIZON, axis=0)
        cubes = predicted[:,1:,6:9]
        valid = np.isfinite(predicted).all(axis=(1,2))
        valid &= (cubes[:,:,2] >= -0.03).all(axis=1) & (cubes[:,:,2] <= 0.4).all(axis=1)
        valid &= (np.abs(cubes[:,:,0]) <= 0.6).all(axis=1)
        valid &= (cubes[:,:,1] >= 0.1).all(axis=1) & (cubes[:,:,1] <= 0.85).all(axis=1)
        errors = np.linalg.norm(cubes - reference[None], axis=2).mean(axis=1)
        errors[~valid] = np.inf
        best = int(np.argmin(errors))
        reason = 'BC retained: predicted improvement below margin'
        if not valid[0]:
            best, reason = 0, 'BC fallback: nominal forecast invalid'
        elif errors[0] - errors[best] < IMPROVEMENT_MARGIN_M:
            best = 0
        else:
            reason = 'alternative forecast improves cube reference tracking'
        selected = requested if best == 0 else (OPEN if best == 1 else CLOSED)
        info = {'purpose': purpose, 'choice': CHOICES[best], 'override': selected != requested,
                'reason': reason, 'errors_m': [float(x) if np.isfinite(x) else None for x in errors],
                'candidate_valid': valid.tolist()}
        return selected, info, (predicted, candidates, reference)
