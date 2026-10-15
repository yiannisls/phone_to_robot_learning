"""Replay a frozen lab policy and record evidence without changing training."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=987)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    sys.path.insert(0, str(args.lab.resolve()))
    import cv2
    import mujoco
    import numpy as np
    import torch
    from hw3.eval_utils import (load_checkpoint, infer_action_chunk, apply_action,
                               check_success, check_cube_out_of_bounds)
    from hw3.sim_env import SO100SimEnv
    torch.set_num_threads(1)
    policy, normalizer, chunk, state_keys, action_keys = load_checkpoint(args.checkpoint, torch.device('cpu'))
    assert chunk == 16, 'This diagnostic reproduces the selected 16-action lab policy.'
    args.out.mkdir(parents=True, exist_ok=False)
    env = SO100SimEnv(args.lab.resolve()/'so101_gym/assets/so100_transfer_cube_obstacle_ee.xml',
                      seed=args.seed, obstacle_mode='adversarial', render_w=640, render_h=480)
    rows = []
    try:
        for episode in range(1, args.episodes+1):
            obs = env.reset()
            queue = deque()
            trace, contacts, actions, frames = [], [], [], []
            initial_cube = env.get_cube_state()[:3].copy()
            success = False
            for step in range(800):
                if not queue:
                    queue.extend(infer_action_chunk(model=policy, normalizer=normalizer,
                                 obs=obs, state_keys=state_keys, device=torch.device('cpu'))[:16])
                action = queue.popleft()
                apply_action(env, action, action_keys)
                obs = env.step()
                cube, hand = env.get_cube_state()[:3], env.get_ee_state()[:3]
                trace.append(np.r_[cube, hand, env.get_gripper_angle(), env.get_goal_pos()[:3]])
                actions.append(action)
                names = []
                for contact in env.data.contact:
                    pair = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
                            for g in (contact.geom1,contact.geom2)]
                    if any('obstacle' in x for x in pair):
                        names.append({'geometries':pair, 'distance_m':float(contact.dist)})
                if names:
                    contacts.append({'step':step+1, 'contacts':names})
                if episode == 3 and step % 5 == 0:
                    frames.append(cv2.cvtColor(env.render_rgb('angle'), cv2.COLOR_RGB2BGR))
                success = check_success(env)
                if success or check_cube_out_of_bounds(env):
                    break
            data = np.array(trace)
            np.savez_compressed(args.out/f'episode_{episode:03d}.npz',
                                trace=data, actions=np.array(actions))
            (args.out/f'episode_{episode:03d}_contacts.json').write_text(json.dumps(contacts,indent=2))
            if frames:
                writer=cv2.VideoWriter(str(args.out/f'episode_{episode:03d}.avi'),
                                       cv2.VideoWriter_fourcc(*'MJPG'), 2, (640,480))
                if not writer.isOpened():
                    raise RuntimeError('Cannot write diagnostic video')
                for frame in frames:
                    writer.write(frame)
                writer.release()
                for label,index in [('initial',0),('middle',len(frames)//2),('final',len(frames)-1)]:
                    cv2.imwrite(str(args.out/f'episode_{episode:03d}_{label}.jpg'),frames[index])
            row={'episode':episode,'success':bool(success),'steps':len(actions),
                 'initial_cube_xyz':initial_cube.tolist(), 'max_cube_height_m':float(data[:,2].max()),
                 'final_cube_xyz':data[-1,:3].tolist(),'final_hand_xyz':data[-1,3:6].tolist(),
                 'goal_xyz':data[-1,7:10].tolist(),
                 'obstacle_xyz':env.get_obstacle_pos().tolist(),
                 'steps_with_obstacle_contacts':len(contacts),
                 'gripper_command_min_max':[float(np.array(actions)[:,3].min()),float(np.array(actions)[:,3].max())]}
            rows.append(row)
            print(json.dumps(row),flush=True)
    finally:
        env.renderer.close()
    report={'seed':args.seed, 'execution_horizon':16,'max_steps':800,'mujoco_version':mujoco.__version__,
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'trace_columns':['cube_x','cube_y','cube_z','hand_x','hand_y','hand_z','gripper_angle','goal_x','goal_y','goal_z'],
            'contact_sampling':'End of each control step; transient contacts may be missed.',
            'source_sha256':{str(p.relative_to(args.lab)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [args.lab/'hw3/eval_utils.py',args.lab/'hw3/model.py',args.lab/'hw3/sim_env.py']},
            'episodes':rows, 'purpose':'Failure diagnosis on previously evaluated layouts; not a new success-rate estimate.'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
