"""Run a frozen, paired BC/world-model experiment on saved fresh positions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from gripper_planner import settings

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layouts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.layouts < 1 or args.seed < 0:
        parser.error('Use a positive layout count and nonnegative seed')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    rng = np.random.default_rng(args.seed)
    cases = [{'name': f'layout_{i:02d}',
              'offset_xy_m': rng.uniform([-.04,-.03],[.04,.04]).tolist()}
             for i in range(args.layouts)]
    manifest = {
        'purpose': 'Frozen paired simulated-layout comparison and separate artificial opening stress test',
        'seed': args.seed, 'range_xy_m': [[-.04,-.03],[.04,.04]], 'cases': cases,
        'controllers': ['bc','wm'], 'conditions': ['normal','opening_pulse'],
        'policy_sha256': sha256(PROJECT_FOLDER/'checkpoints/bc_v2/policy.pt'),
        'world_model_sha256': sha256(PROJECT_FOLDER/'checkpoints/world_model_v2/world_model.pt'),
        'settings': settings(), 'test_source_used': False,
        'stress_rule': 'One opening request after cube z >= 70 mm, outside bin xy tolerance, with applied and requested gripper closed; subsequent BC requests resume.',
        'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                        ['compare_model_control.py','evaluate_model_control.py','gripper_planner.py',
                         'record_gripper.py','policy.py','world_model.py']},
    }
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit('Saved protocol differs; use a new output folder')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    completed = []
    for condition in manifest['conditions']:
        for case in cases:
            for controller in manifest['controllers']:
                folder = args.out/f'{condition}_{controller}'/case['name']
                command = [sys.executable,str(Path(__file__).with_name('evaluate_model_control.py')),
                           '--mode',controller,'--out',str(folder.resolve()), '--label',case['name'],
                           '--offset-xy',*[str(v) for v in case['offset_xy_m']]]
                if condition == 'opening_pulse':
                    command.append('--stress-open')
                if not (args.resume and (folder/'result.json').exists()):
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode:
                        (args.out/'failed_process.json').write_text(json.dumps({
                            'condition': condition, 'controller': controller, 'case': case,
                            'stdout': result.stdout, 'stderr': result.stderr}, indent=2))
                        raise SystemExit(result.stdout+result.stderr)
                result = json.loads((folder/'result.json').read_text())
                assert result['checkpoint_sha256'] == manifest['policy_sha256']
                assert result['offset_xy_m'] == case['offset_xy_m']
                assert result['controller'] == controller
                assert result['stress_open'] == (condition == 'opening_pulse')
                assert not result['test_used']
                if controller == 'wm':
                    assert result['world_model_sha256'] == manifest['world_model_sha256']
                    assert result['planner_settings'] == manifest['settings']
                for name,value in result['code_sha256'].items():
                    assert value == manifest['code_sha256'][name]
                assert (folder/'rollout.npz').exists() and (folder/'physics.npz').exists()
                result['condition'] = condition
                result['folder'] = str(folder.relative_to(args.out))
                completed.append(result)
                (args.out/'completed.json').write_text(json.dumps(completed, indent=2))
                print(f"{condition} {controller} {case['name']}: task={result['task_success']}, "
                      f"depth={result['quality_pass']}, grip_overrides={result['gripper_overrides']}, "
                      f"pulse={result['injected_step']}", flush=True)
    grouped = {}
    for condition in manifest['conditions']:
        grouped[condition] = {}
        for controller in manifest['controllers']:
            rows = [r for r in completed if r['condition'] == condition and r['controller'] == controller]
            triggered = [r for r in rows if r['injected_step'] is not None]
            grouped[condition][controller] = {
                'episodes': len(rows), 'task_successes': sum(r['task_success'] for r in rows),
                'task_and_contact_depth_passes': sum(r['task_success'] and r['quality_pass'] for r in rows),
                'worst_jaw_peak_n': max(max(r['peak_normal_force_per_jaw_n']) for r in rows),
                'max_cube_penetration_m': max(r['max_cube_penetration_m'] for r in rows),
                'total_gripper_overrides': sum(r['gripper_overrides'] for r in rows),
                'pulse_triggered_episodes': len(triggered),
                'pulse_overridden_episodes': sum(r['injected_opening_overridden'] for r in triggered),
                'triggered_episode_task_successes': sum(r['task_success'] for r in triggered),
                'median_control_steps': float(np.median([r['steps'] for r in rows])),
                'median_episode_planner_p95_ms': float(np.median([r['planner_latency_ms_p50_p95'][1]
                    for r in rows if r['planner_latency_ms_p50_p95'] is not None])) if controller == 'wm' else None,
            }
        bc = [r for r in completed if r['condition'] == condition and r['controller'] == 'bc']
        wm = [r for r in completed if r['condition'] == condition and r['controller'] == 'wm']
        grouped[condition]['paired_task_outcomes'] = {
            'both_pass': sum(b['task_success'] and w['task_success'] for b,w in zip(bc,wm)),
            'only_bc_pass': sum(b['task_success'] and not w['task_success'] for b,w in zip(bc,wm)),
            'only_wm_pass': sum(not b['task_success'] and w['task_success'] for b,w in zip(bc,wm)),
            'both_fail': sum(not b['task_success'] and not w['task_success'] for b,w in zip(bc,wm)),
        }
    summary = {'manifest': manifest, 'summary': grouped, 'episodes': completed,
               'limitations': 'One seed, position changes only. Artificial pulses do not establish natural policy improvement or realistic force control. No retraining or parameter selection from these outcomes.'}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(grouped, indent=2), flush=True)


if __name__ == '__main__':
    main()
