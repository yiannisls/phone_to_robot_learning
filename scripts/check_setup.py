"""Check imports and explicitly configured simulator files before evaluation."""
import importlib
import os
from pathlib import Path


def main():
    for name in ('numpy', 'torch', 'mujoco', 'cv2', 'pyquaternion'):
        module = importlib.import_module(name)
        print(f'{name}: {getattr(module, "__version__", "import OK")}')
    value = os.environ.get('ROBOT_LEARNING_HW3')
    if not value:
        raise SystemExit('Set ROBOT_LEARNING_HW3 to your HW3 directory.')
    root = Path(value).expanduser().resolve()
    for relative in ('hw3/sim_env.py', 'so101_gym/assets/so100_transfer_cube_obstacle_ee.xml'):
        if not (root / relative).is_file():
            raise SystemExit(f'Missing simulator dependency: {root / relative}')
    print('Setup imports and simulator paths OK. Run evaluation to check rendering.')


if __name__ == '__main__':
    main()
