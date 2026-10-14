"""Compare frozen BC movement with or without learned gripper look-ahead."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco
import numpy as np
import torch

from gripper_planner import GripperPlanner, action_sequence, settings
from policy import HORIZON, Policy, make_features
from record_gripper import GripperRecorder, create_environment, PHYSICS_COLUMNS

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=PROJECT_FOLDER/'checkpoints/bc_v2/policy.pt')
    parser.add_argument('--world-model', type=Path, default=PROJECT_FOLDER/'checkpoints/world_model_v2/world_model.pt')
    parser.add_argument('--mode', choices=['bc', 'wm'], default='wm')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--offset-xy', type=float, nargs=2)
    parser.add_argument('--label', default='validation')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--stress-open', action='store_true', help='Inject one opening request during early elevated transport')
    args = parser.parse_args()
    if 'ROBOT_LEARNING_HW3' not in os.environ:
        parser.error('Set ROBOT_LEARNING_HW3 to your homework folder')
    offset = args.offset_xy
    if offset is None:
        offset = json.loads((PROJECT_FOLDER/'results/dataset_v2.json').read_text())['base_offsets_xy_m']['IMG_5542']
    if not np.isfinite(offset).all():
        parser.error('Cube offsets must be finite')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.policy, map_location='cpu', weights_only=True)
    assert checkpoint['horizon'] == HORIZON
    policy = Policy(checkpoint['input_size'])
    policy.load_state_dict(checkpoint['model'])
    policy.eval()
    env = create_environment(Path(os.environ['ROBOT_LEARNING_HW3']), offset, 3.5)
    writer = None
    try:
        profile = {'gripper_close_rate_rad_s': 1.0, 'control_dt_s': env.dt_ctrl,
                   'physics_dt_s': env.sim_dt, 'jaw_torque_range_nm': [-3.5,3.5],
                   'mujoco_version': mujoco.__version__}
        planner = GripperPlanner(args.world_model, profile) if args.mode == 'wm' else None
        args.out.mkdir(parents=True, exist_ok=False)
        if args.video:
            writer = cv2.VideoWriter(str(args.out/'rollout.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                    10, (env.render_w,env.render_h))
            if not writer.isOpened():
                raise RuntimeError('Cannot create rollout video')
        recorder = GripperRecorder(env, 1.0)
        cursor = 4
        decisions, timings, prediction_steps, predictions, candidate_actions, references = [], [], [], [], [], []
        success_count = 0
        task_success, out_of_bounds, pulse_fired = False, False, False
        injected_step = None
        peak_height = 0.0
        for step in range(240):
            observation = recorder.states[-1]
            if cursor == 4:
                features = (torch.from_numpy(make_features(observation)) - checkpoint['feature_mean']) / checkpoint['feature_std']
                with torch.inference_mode():
                    predicted = policy(features[None])[0].numpy()
                base = observation['state_mocap_target']
                targets = base + predicted[:HORIZON,:3] * checkpoint['position_scale'].numpy()
                grippers = np.where(predicted[:HORIZON,3] > 0, 1.0, -0.174)
                cursor = 0
            future_grippers = grippers[cursor:cursor+5].copy()
            cube, goal = observation['state_cube'][:3], observation['goal_pos']
            injected = bool(args.stress_open and not pulse_fired and cube[2] >= 0.07
                            and np.max(np.abs(cube[:2] - goal[:2])) >= 0.04
                            and observation['state_gripper_command'][0] < 0.4
                            and future_grippers[0] < 0.4)
            original_request = float(future_grippers[0])
            if injected:
                future_grippers[0] = 1.0
                pulse_fired, injected_step = True, step
            nominal_request = float(future_grippers[0])
            info = {'purpose': 'bc_only', 'choice': 'bc_sequence', 'override': False}
            chosen = nominal_request
            if planner is not None:
                nominal = action_sequence(observation['state_mocap_target'],
                                          targets[cursor:cursor+5], future_grippers)
                start = time.perf_counter()
                chosen, info, trace = planner.choose(observation, nominal)
                timings.append((time.perf_counter()-start)*1000)
                if trace is not None:
                    prediction_steps.append(step)
                    p, a, reference = trace
                    predictions.append(p); candidate_actions.append(a); references.append(reference)
            decisions.append({'step': step, 'original_bc_request': original_request,
                              'injected_opening': injected, 'nominal_request': nominal_request,
                              'executed_request': chosen, **info})
            # BC movement still replans every four commands. Only the grip can change.
            recorder.step(targets[cursor] - observation['state_mocap_target'], chosen)
            cursor += 1
            samples = np.asarray(recorder.physics[-env.substeps:])
            peak_height = max(peak_height, float(samples[:,11].max()))
            state = recorder.states[-1]
            cube, hand, goal = state['state_cube'][:3], state['state_ee_full'][:3], state['goal_pos']
            stable = len(recorder.states) >= 6 and np.linalg.norm(cube-recorder.states[-6]['state_cube'][:3]) < .005
            done = (np.max(np.abs(cube[:2]-goal[:2])) < .04 and cube[2] < .05
                    and peak_height > .06 and stable and np.linalg.norm(hand-cube) > .07
                    and state['state_gripper_command'][0] > .4)
            success_count = success_count+1 if done else 0
            out_of_bounds = bool(not np.isfinite(env.data.qpos).all() or cube[2] < -.03
                                 or abs(cube[0]) > .6 or cube[1] < .1 or cube[1] > .85)
            if writer is not None:
                frame = cv2.cvtColor(env.render_rgb(), cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"{args.mode} | {args.label} | grip: {info['choice']}",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                if args.stress_open:
                    cv2.putText(frame, 'Artificial opening pulse: '+('injected' if pulse_fired else 'pending'),
                                (10,45), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                writer.write(frame)
            if success_count >= 5:
                task_success = True
                break
            if out_of_bounds:
                break
        arrays = {key: np.stack([state[key] for state in recorder.states]) for key in recorder.states[0]}
        np.savez_compressed(args.out/'rollout.npz', **arrays, actions=np.asarray(recorder.actions))
        physics = np.asarray(recorder.physics)
        np.savez_compressed(args.out/'physics.npz', samples=physics, columns=np.array(PHYSICS_COLUMNS))
        if predictions:
            np.savez_compressed(args.out/'predictions.npz', steps=np.array(prediction_steps),
                                predicted=np.stack(predictions), actions=np.stack(candidate_actions),
                                references=np.stack(references))
        (args.out/'decisions.json').write_text(json.dumps(decisions, indent=2))
        result = {
            'source': 'IMG_5542' if args.offset_xy is None else None,
            'split': 'validation' if args.offset_xy is None else 'diagnostic',
            'label': args.label, 'offset_xy_m': offset, 'controller': args.mode,
            'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_sha256': sha256(args.policy),
            'world_model_sha256': sha256(args.world_model) if planner else None,
            'task_success': task_success, 'quality_pass': float(physics[:,8].max()) <= .002,
            'steps': len(recorder.actions), 'out_of_bounds': out_of_bounds,
            'peak_cube_height_m': peak_height, 'final_cube_xyz': cube.tolist(),
            'final_hand_xyz': hand.tolist(), 'final_gripper_command': float(state['state_gripper_command'][0]),
            'max_cube_penetration_m': float(physics[:,8].max()),
            'peak_normal_force_per_jaw_n': physics[:,6:8].max(axis=0).tolist(),
            'profile': profile, 'planner_settings': settings() if planner else None,
            'gripper_overrides': sum(row['override'] for row in decisions),
            'active_prediction_steps': len(prediction_steps),
            'planner_latency_ms_p50_p95': np.percentile(timings,[50,95]).tolist() if timings else None,
            'stress_open': args.stress_open, 'injected_step': injected_step,
            'injected_opening_overridden': bool(injected_step is not None and decisions[injected_step]['override']),
            'test_used': False,
            'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                            ('evaluate_model_control.py','gripper_planner.py','record_gripper.py','policy.py','world_model.py')},
            'limitations': 'Short learned state forecasts, task-specific reference and margin; no force prediction. Artificial opening pulse is a separate stress test.'
        }
        (args.out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if writer is not None: writer.release()
        env.renderer.close()


if __name__ == '__main__':
    main()
