"""Test a phone-derived transport path in the lab scene, without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--source', choices=['IMG_5539', 'IMG_5540', 'IMG_5541'], default='IMG_5539')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--without-obstacles', action='store_true', help='Diagnostic control only; not the lab task')
    parser.add_argument('--approach', choices=['scripted', 'human'], default='scripted')
    args = parser.parse_args()
    import cv2
    import mujoco
    import numpy as np
    sys.path.insert(0, str(args.lab.resolve()))
    from hw3.sim_env import SO100SimEnv
    from hw3.eval_utils import check_success

    transport_path = PROJECT/'data/processed'/f'{args.source}_transport.csv'
    transport = np.genfromtxt(transport_path, delimiter=',', names=True)
    if len(transport) < 2 or not np.isfinite(transport.tolist()).all():
        raise ValueError('Transport must contain finite samples.')
    if not np.allclose(np.diff(transport['time_seconds']), .1):
        raise ValueError('This pilot expects the saved 10 Hz transport curve.')
    args.out.mkdir(parents=True, exist_ok=False)
    env = SO100SimEnv(args.lab.resolve()/'so101_gym/assets/so100_transfer_cube_obstacle_ee.xml',
                      seed=0, cube_pos_std=0, obstacle_pos_std=0)
    if args.without_obstacles:
        for name in ['obstacle_wall', 'upper_obstacle']:
            geom = env.model.geom(name)
            geom.contype = geom.conaffinity = 0
            geom.rgba[3] = 0
    env.reset()
    assert np.isclose(env.dt_ctrl, .1)
    # Use the lab's native instantaneous gripper target executor.
    # These are chosen teaching commands, not grip commands measured in the phone video.
    open_grip, closed_grip = .7, 0.0
    cube_id = env.model.geom('red_box').id
    obstacle_ids = {env.model.geom(n).id for n in ['obstacle_wall', 'upper_obstacle']}

    def observe():
        return {'state_ee_full':env.get_ee_state().copy(), 'state_joints':env.get_joint_angles().copy(),
                'state_cube':env.get_cube_state().copy(), 'state_gripper':np.array([env.get_gripper_angle()]),
                'state_obstacle':env.get_obstacle_pos().copy(), 'goal_pos':env.get_goal_pos().copy(),
                'state_mocap_target':env.data.mocap_pos[0].copy(), 'qpos':env.data.qpos.copy(),
                'qvel':env.data.qvel.copy()}

    human = None
    human_handoff = None
    human_hash = None
    if args.approach == 'human':
        import zarr
        store = zarr.open_group(str(args.lab/'datasets/curated/single_cube/teleop/successful_demos.zarr'), mode='r')
        assert list(store.attrs['source_episode_numbers'])[0] == 1
        assert np.isclose(float(store.attrs['control_hz']), 10)
        count = int(store['meta/episode_ends'][0])
        human = {key:np.asarray(store['data/'+key][:count]) for key in
                 ['state_ee','state_cube','state_obstacle','action_gripper']}
        candidates = np.flatnonzero((human['state_cube'][:,2]>.06) &
                     (human['state_cube'][:,1]>human['state_obstacle'][0,1]+.08))
        if not len(candidates):raise ValueError('Human episode has no carrying handoff.')
        human_handoff = int(candidates[0])
        human_hash = hashlib.sha256(b''.join(human[key].tobytes() for key in sorted(human))).hexdigest()
        closed_grip = float(human['action_gripper'][human_handoff,0])

    states, commands, phases, contact_rows = [observe()], [], [], []
    peak_height = float(env.get_cube_state()[2])
    writer = cv2.VideoWriter(str(args.out/'rollout.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640,480))
    if not writer.isOpened():
        env.renderer.close()
        raise RuntimeError('Cannot write video.')
    try:
        def execute(target, grip, phase, quaternion=None):
            nonlocal peak_height
            previous = env.data.mocap_pos[0].copy()
            env.set_mocap_pos(target)
            if quaternion is not None:
                env.set_mocap_quat(quaternion)
            env.set_gripper(grip)
            obstacle_contacts = 0
            obstacle_depth, cube_depth = 0.0, 0.0
            for _ in range(env.substeps):
                mujoco.mj_step(env.model, env.data)
                peak_height = max(peak_height, float(env.get_cube_state()[2]))
                for c in env.data.contact:
                    pair = {int(c.geom1), int(c.geom2)}
                    if pair & obstacle_ids:
                        obstacle_contacts += 1
                        obstacle_depth = max(obstacle_depth, -float(c.dist))
                    if cube_id in pair:
                        cube_depth = max(cube_depth, -float(c.dist))
            commands.append(np.r_[target-previous, grip])
            phases.append(phase)
            states.append(observe())
            contact_rows.append({'step':len(commands), 'phase':phase,
                                 'obstacle_contact_samples':obstacle_contacts,
                                 'max_obstacle_penetration_m':obstacle_depth,
                                 'max_cube_penetration_m':cube_depth})
            frame = cv2.cvtColor(env.render_rgb('angle'), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f'Phone path in lab | {phase}', (10,25), 0, .55, (255,255,255), 1)
            cv2.putText(frame, 'Demo generation; obstacles '+('OFF (control)' if args.without_obstacles else 'ON'),
                        (10,48), 0, .5, (255,255,255), 1)
            writer.write(frame)

        def move(start, end, steps, grip, phase):
            for fraction in np.linspace(0, 1, steps):
                execute(start + fraction*(end-start), grip, phase)

        start_cube = env.get_cube_state()[:3].copy()
        goal = env.get_goal_pos().copy()
        grasp = start_cube + [-.025, 0, .04]
        above = grasp + [0, 0, .10]
        if human is None:
            move(env.get_ee_pos(), above, 25, open_grip, 'approach')
            move(above, grasp, 25, open_grip, 'lower')
            move(grasp, grasp, 15, closed_grip, 'grasp')
            transport_start = start_cube.copy()
            handoff_hand = grasp.copy()
        else:
            # Only translate the measured training route; no state teleporting.
            shift = start_cube - human['state_cube'][0,:3]
            shift[2] = 0
            for i in range(human_handoff+1):
                execute(human['state_ee'][i,:3]+shift,
                        float(human['action_gripper'][i,0]), 'human_approach',
                        human['state_ee'][i,3:])
            transport_start = env.get_cube_state()[:3].copy()
            handoff_hand = env.get_ee_pos().copy()
        actual_handoff = env.get_cube_state()[:3].copy()
        secure_handoff = bool(actual_handoff[2]>.06 and
                              actual_handoff[1]>env.get_obstacle_pos()[1]+.08)
        displacement = goal[:2]-transport_start[:2]
        direction = displacement/np.linalg.norm(displacement)
        sideways = np.array([-direction[1], direction[0]])
        ready = False
        transport_enabled = human is None or secure_handoff
        for point in transport[1:] if transport_enabled else []:
            target = handoff_hand.copy()
            progress = point['progress']
            target[:2] += progress*displacement + point['lateral_m']*sideways
            if human is None:
                target[2] += point['height_m']
            else:
                # Already lifted: retain 25% of the phone lift shape at carrying height.
                target[2] += .25*point['height_m']
            execute(target, closed_grip, 'phone_transport')
            cube = env.get_cube_state()[:3]
            ready = bool(peak_height>.06 and np.max(np.abs(cube[:2]-goal[:2]))<.03 and cube[2]<.045)
            if ready and human is None:
                break
        if human is not None and secure_handoff:
            # Explicit teaching feedback; this is not motion measured in the phone video.
            for _ in range(40):
                cube = env.get_cube_state()[:3]
                if np.max(np.abs(cube[:2]-goal[:2]))<.015 and .06<cube[2]<.15:
                    break
                error = np.r_[goal[:2]-cube[:2], .10-cube[2]]
                target = env.data.mocap_pos[0].copy() + np.clip(error,-.003,.003)
                target = np.clip(target, env.get_ee_pos()-.03, env.get_ee_pos()+.03)
                execute(target,closed_grip,'placement_feedback')
            cube = env.get_cube_state()[:3]
            ready = bool(np.max(np.abs(cube[:2]-goal[:2]))<.03 and .06<cube[2]<.15)
        if ready:
            target = env.data.mocap_pos[0].copy()
            for _ in range(12 if human is None else 0):
                height = float(env.get_cube_state()[2])
                if height<=.026:break
                target[2] -= min(.003, height-.023)
                execute(target, closed_grip, 'place')
            move(target, target, 2, open_grip, 'release')
            retreat = target + [0,0,.14]
            move(target, retreat, 20, open_grip, 'retreat')
            move(retreat, retreat, 10, open_grip, 'settle')
        arrays = {k:np.stack([s[k] for s in states]) for k in states[0]}
        commands = np.array(commands)
        # These labels match the human lab's achieved-hand difference convention.
        actions = np.c_[np.diff(arrays['state_ee_full'][:,:3], axis=0), commands[:,3]]
        assert len(states)==len(actions)+1
        assert np.allclose(np.diff(arrays['state_mocap_target'],axis=0),commands[:,:3],atol=1e-10)
        assert all(np.isfinite(a).all() for a in arrays.values()) and np.isfinite(actions).all()
        cube, hand = arrays['state_cube'][-1,:3], arrays['state_ee_full'][-1,:3]
        settled = np.linalg.norm(cube-arrays['state_cube'][-6,:3])<.005
        released = bool(ready and settled and np.linalg.norm(hand-cube)>.07)
        native_success = bool(check_success(env))
        result = {'source':args.source, 'split':'train', 'scripted_demonstration':True, 'approach':args.approach,
                  'human_source_episode':1 if human is not None else None,
                  'human_episode_array_sha256':human_hash, 'human_handoff_index':human_handoff,
                  'actual_handoff_cube_xyz':actual_handoff.tolist(), 'secure_handoff':secure_handoff,
                  'obstacles_enabled':not args.without_obstacles, 'states':len(states), 'actions':len(actions),
                  'control_dt_s':env.dt_ctrl, 'physics_dt_s':env.sim_dt, 'mujoco_version':mujoco.__version__,
                  'native_lab_position_success':native_success, 'placement_gate_reached':ready,
                  'released_and_settled':released,'task_success':bool(native_success and released and peak_height>.06),
                  'peak_cube_height_m':peak_height, 'final_cube_xyz':cube.tolist(), 'goal_xyz':goal.tolist(),
                  'obstacle_contact_samples':sum(r['obstacle_contact_samples'] for r in contact_rows),
                  'max_obstacle_penetration_m':max(r['max_obstacle_penetration_m'] for r in contact_rows),
                  'max_cube_penetration_m':max(r['max_cube_penetration_m'] for r in contact_rows),
                  'gripper_target_open_closed_rad':[open_grip,closed_grip],
                  'transport_sha256':hashlib.sha256(transport_path.read_bytes()).hexdigest(),
                  'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'lab_simulator_sha256':hashlib.sha256((args.lab/'hw3/sim_env.py').read_bytes()).hexdigest(),
                  'action_semantics':'Measured EE XYZ differences plus requested native gripper target',
                  'command_semantics':'Mocap target XYZ increments plus requested native gripper target',
                  'limitations':'One deterministic training-source pilot; scripted approach/grasp/release; no policy training or success-rate estimate.'}
        result['contact_depth_pass'] = result['max_cube_penetration_m']<=.002
        result['obstacle_clearance_pass'] = result['obstacle_contact_samples']==0
        result['accepted_for_training'] = False  # Audit-only output; never auto-import.
        np.savez_compressed(args.out/'demonstration.npz', **arrays, actions=actions, commands=commands,
                            phase=np.array(phases), time_s=np.arange(len(states))*env.dt_ctrl,
                            source_video=np.array(args.source),split=np.array('train'))
        (args.out/'contacts.json').write_text(json.dumps(contact_rows,indent=2))
        (args.out/'result.json').write_text(json.dumps(result,indent=2))
        cv2.imwrite(str(args.out/'final.jpg'),cv2.cvtColor(env.render_rgb('angle'),cv2.COLOR_RGB2BGR))
        print(json.dumps(result,indent=2))
    finally:
        writer.release()
        env.renderer.close()


if __name__=='__main__':main()
