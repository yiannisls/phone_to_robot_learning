"""Compare learned state predictions with saved simulator outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from world_model import WorldModel, load_episodes, make_state, predict_next

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def errors(prediction, actual):
    difference = (prediction - actual).numpy()
    return {
        "cube_mae_m": float(np.abs(difference[..., 6:9]).mean()),
        "cube_rmse_m": float(np.sqrt(np.square(difference[..., 6:9]).mean())),
        "hand_mae_m": float(np.abs(difference[..., 25:28]).mean()),
        "joint_angle_mae_rad": float(np.abs(difference[..., :6]).mean()),
        "gripper_angle_mae_rad": float(np.abs(difference[..., 5]).mean()),
        "samples": len(difference),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_FOLDER / "data/demonstrations_v2")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_FOLDER / "checkpoints/world_model_v1/world_model.pt")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or PROJECT_FOLDER / f"results/world_model_v1_{args.split}"
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = WorldModel()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    episodes, _ = load_episodes(args.data, args.split, checkpoint["profile"])
    out.mkdir(parents=True, exist_ok=False)

    results = []
    with torch.no_grad():
        for path, episode in episodes:
            state = torch.from_numpy(make_state(episode))
            actions = torch.from_numpy(episode["actions"])
            prediction = predict_next(model, checkpoint, state[:-1], actions)
            # Shuffle actions while preserving initial states to test sensitivity.
            order = torch.randperm(len(actions), generator=torch.Generator().manual_seed(7))
            shuffled = predict_next(model, checkpoint, state[:-1], actions[order])
            result = {
                "name": path.stem, "split": args.split,
                "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "one_step": errors(prediction, state[1:]),
                "persistence_one_step": errors(state[:-1], state[1:]),
                "shuffled_action_one_step": errors(shuffled, state[1:]),
                "one_step_by_phase": {},
            }
            for phase in np.unique(episode["phase"]):
                mask = episode["phase"] == phase
                result["one_step_by_phase"][str(phase)] = errors(prediction[mask], state[1:][mask])

            saved = {"actual_states": state.numpy(), "actions": actions.numpy(),
                     "one_step_predictions": prediction.numpy()}
            for horizon in (5, 10):
                starts = torch.arange(len(actions) - horizon + 1)
                rollout = state[starts]
                # Use predicted states throughout; only actions come from the file.
                for step in range(horizon):
                    rollout = predict_next(model, checkpoint, rollout, actions[starts + step])
                actual = state[starts + horizon]
                result[f"open_loop_{horizon}_step"] = errors(rollout, actual)
                result[f"persistence_{horizon}_step"] = errors(state[starts], actual)
                saved[f"{horizon}_step_start_indices"] = starts.numpy()
                saved[f"{horizon}_step_predictions"] = rollout.numpy()
            assert all(np.isfinite(array).all() for array in saved.values())
            np.savez_compressed(out / f"{path.stem}_predictions.npz", **saved)
            results.append(result)

    report = {
        "checkpoint_epoch": checkpoint["epoch"], "profile": checkpoint["profile"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episodes": results,
        "limitations": "One source per evaluation split; recorded teacher actions; no alternative-action validation or control improvement claimed.",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
