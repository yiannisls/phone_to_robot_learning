"""Compare jaw torque and closing-rate limits on the validation layout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "results/gripper_torque_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = PROJECT_FOLDER / "checkpoints/bc_v1/policy.pt"
    evaluator = PROJECT_FOLDER / "scripts/evaluate_policy.py"
    manifest = {
        "source": "IMG_5542", "split": "validation",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_script_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
        "settings": [{"torque_nm": torque, "close_rate_rad_s": rate}
                     for torque in (3.5, 1.0, 0.3) for rate in (2.0, 0.5)],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "evaluate_policy.py").write_text(evaluator.read_text())
    results = []
    for settings in manifest["settings"]:
        torque, rate = settings["torque_nm"], settings["close_rate_rad_s"]
        name = f"torque_{torque}_rate_{rate}".replace(".", "p")
        folder = args.out / name
        command = [
            sys.executable, str(evaluator), "--checkpoint", str(checkpoint),
            "--out", str(folder.resolve()), "--label", name, "--video",
            "--jaw-torque-limit", str(torque), "--gripper-close-rate", str(rate),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise SystemExit(completed.stdout + completed.stderr)
        result = json.loads((folder / "result.json").read_text())
        assert result["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        assert result["evaluation_script_sha256"] == manifest["evaluation_script_sha256"]
        with np.load(folder / "physics.npz", allow_pickle=False) as physics:
            forces = physics["samples"][:, 1:3].max(axis=1)
            result["force_p95_n"] = float(np.percentile(forces, 95))
            result["time_above_100_n_s"] = float((forces > 100).sum() * 0.002)
        results.append({"name": name, **settings, **result})
        (args.out / "summary.json").write_text(json.dumps({
            **manifest, "episodes": results,
            "force_p95_definition": "95th percentile over all physics steps of the larger jaw force",
        }, indent=2))
        print(f"{name}: success={result['task_success']}, "
              f"peak={max(result['peak_normal_force_per_jaw_n']):.1f} N, "
              f"p95={result['force_p95_n']:.1f} N", flush=True)


if __name__ == "__main__":
    main()
