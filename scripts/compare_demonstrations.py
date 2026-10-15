"""Compare one human and one phone-derived demonstration without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab',type=Path,required=True)
    p.add_argument('--phone',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import zarr
    import mujoco
    import cv2
    sys.path.insert(0,str(args.lab.resolve()))
    from hw3.sim_env import SO100SimEnv
    args.out.mkdir(parents=True,exist_ok=False)
    rawpath=args.lab/'datasets/curated/single_cube/teleop/successful_demos.zarr'
    processedpath=args.lab/'datasets/processed/single_cube/processed_ee_xyz.zarr'
    raw=zarr.open_group(str(rawpath),mode='r')
    processed=zarr.open_group(str(processedpath),mode='r')
    end=int(raw['meta/episode_ends'][0])
    pend=int(processed['meta/episode_ends'][0])
    assert pend==end-1
    lab={k:np.asarray(raw['data/'+k][:end]) for k in
         ['state_joints','state_ee','state_cube','state_gripper','action_gripper']}
    saved_actions=np.asarray(processed['data/action_ee_xyz'][:pend])
    assert np.allclose(saved_actions,np.diff(lab['state_ee'][:,:3],axis=0),atol=1e-7,rtol=0)
    attrs=dict(raw.attrs)
    assert attrs['joint_names']==['Rotation','Pitch','Elbow','Wrist_Pitch','Wrist_Roll','Jaw']
    assert list(attrs.get('source_episode_numbers',range(1,22)))[0]==1
    with np.load(args.phone,allow_pickle=False) as f:phone={k:f[k] for k in f.files}
    assert phone['split'].item()=='train' and phone['source_video'].item()=='IMG_5539'
    phone_actions=phone['actions']
    assert np.allclose(phone_actions[:,:3],np.diff(phone['state_mocap_target'],axis=0),atol=1e-7,rtol=0)
    names=['Rotation','Pitch','Elbow','Wrist_Pitch','Wrist_Roll','Jaw']
    env=SO100SimEnv(args.lab.resolve()/'so101_gym/assets/so100_transfer_cube_obstacle_ee.xml',
                    seed=0,cube_pos_std=0,obstacle_pos_std=0)
    # Common geometry only: replaying recorded poses does not simulate actions.
    for name in ['obstacle_wall','upper_obstacle']:
        g=env.model.geom(name);g.contype=g.conaffinity=0;g.rgba[3]=0
    env.reset()
    goal=env.get_goal_pos().copy()
    common={'bin_body_xyz':env.model.body('bin').pos.tolist(),'goal_xyz':goal.tolist(),
            'obstacles_enabled':False,'replay':'Recorded poses with mj_forward; no physics stepping or task score.'}
    sources=[('human',lab['state_joints'],lab['state_cube'],lab['state_ee'],
              np.arange(end)/float(attrs['control_hz'])),
             ('phone',phone['state_joints'],phone['state_cube'],phone['state_ee_full'],phone['time_s'])]
    results={}
    try:
        for label,joints,cubes,ee,times in sources:
            writer=cv2.VideoWriter(str(args.out/f'{label}_recorded_poses.avi'),
                cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480))
            if not writer.isOpened():raise RuntimeError('Cannot open replay video')
            errors=[]
            # Output time grid is 10 Hz, with nearest recorded pose; no interpolation.
            try:
                for second in np.arange(0,times[-1]+1e-8,.1):
                    i=int(np.argmin(np.abs(times-second)))
                    env.data.qpos[env.qpos_idx]=joints[i]
                    env.data.qpos[env.cube_qpos_idx]=cubes[i]
                    env.data.qvel[:]=0
                    mujoco.mj_forward(env.model,env.data)
                    errors.append(float(np.linalg.norm(env.get_ee_pos()-ee[i,:3])))
                    frame=cv2.cvtColor(env.render_rgb('angle'),cv2.COLOR_RGB2BGR)
                    cv2.putText(frame,f'{label} recorded poses | {second:.1f}s',(10,25),0,.6,(255,255,255),1)
                    cv2.putText(frame,'Common bin; obstacles off; no physics rollout',(10,48),0,.5,(255,255,255),1)
                    writer.write(frame)
                    if second==0:cv2.imwrite(str(args.out/f'{label}_first.jpg'),frame)
                cv2.imwrite(str(args.out/f'{label}_last.jpg'),frame)
            finally:writer.release()
            results[label]={'states':len(joints),'duration_s':float(times[-1]-times[0]),
                'max_reconstructed_hand_error_m':max(errors),'initial_cube_xyz':cubes[0,:3].tolist(),
                'final_cube_xyz':cubes[-1,:3].tolist(),
                'episode_array_sha256':hashlib.sha256(joints.tobytes()+cubes.tobytes()+ee.tobytes()).hexdigest()}
    finally:env.renderer.close()
    human_deltas=np.diff(lab['state_ee'][:,:3],axis=0)
    phone_deltas=np.diff(phone['state_ee_full'][:,:3],axis=0)
    results['human'].update({'control_dt_s':1/float(attrs['control_hz']),
        'native_goal_xyz':goal.tolist(),'xyz_action_semantics':'Measured EE position differences',
        'gripper_action_semantics':'Saved native actuator target; no explicit closing-rate limiter in recorder',
        'gripper_command_range':[float(lab['action_gripper'].min()),float(lab['action_gripper'].max())]})
    results['phone'].update({'control_dt_s':float(phone['control_dt_s']),
        'native_goal_xyz':phone['goal_pos'][0].tolist(),'xyz_action_semantics':'Commanded mocap target differences',
        'gripper_action_semantics':str(phone['action_gripper_semantics'].item()),
        'close_rate_rad_s':float(phone['gripper_close_rate_rad_s']),
        'gripper_command_range':[float(phone_actions[:,3].min()),float(phone_actions[:,3].max())],
        'mean_norm_target_delta_minus_measured_delta_m':float(np.linalg.norm(phone_actions[:,:3]-phone_deltas,axis=1).mean())})
    output={'human_episode':1,'human_source':'Personally collected human teleoperation, original curated training episode 1',
            'phone_source':'IMG_5539 training family; measured phone transport plus scripted robot phases',
            'phone_sha256':hashlib.sha256(args.phone.read_bytes()).hexdigest(),
            'common_scene':common,'demonstrations':results,
            'checks':{'human_actions_match_measured_deltas':True,'phone_actions_match_target_deltas':True,
                      'source_families_training_only':True},
            'compatible_for_naive_concatenation':False,
            'reason':'Different XYZ action labels, goals, obstacle profiles and gripper execution. Pose reconstruction is not dynamic replay.'}
    (args.out/'comparison.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
