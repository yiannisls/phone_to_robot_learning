"""Learn to imitate the freshly generated robot demonstrations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy import HORIZON, Policy, make_features

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
SOURCE_SPLITS = {
    "IMG_5539": "train", "IMG_5540": "train", "IMG_5541": "train",
    "IMG_5542": "validation", "IMG_5543": "test",
}


def load_examples(folder, split):
    features, positions, grippers, used_files = [], [], [], []
    for path in sorted(folder.glob("*.npz")):
        # Identify the source before opening the file: test data stays unread.
        source = path.stem.split("_aug")[0]
        if SOURCE_SPLITS[source] != split:
            continue
        with np.load(path, allow_pickle=False) as episode:
            assert episode["source_video"].item() == source
            assert episode["split"].item() == split
            if not episode["success"].item():
                continue
            count = len(episode["actions"])
            assert len(episode["state_mocap_target"]) == count + 1
            current = episode["state_mocap_target"][:-1]
            # Near the end, repeat the final target instead of leaving the file.
            future = np.minimum(
                np.arange(count)[:, None] + np.arange(1, HORIZON + 1), count
            )
            features.append(make_features(episode)[:-1])
            positions.append(episode["state_mocap_target"][future] - current[:, None])
            grippers.append((episode["actions"][future - 1, 3] > 0.4).astype(np.float32))
            used_files.append(path)
    if not used_files:
        raise ValueError(f"No successful {split} demonstrations found in {folder}")
    arrays = [torch.from_numpy(np.concatenate(items)) for items in (
        features, positions, grippers
    )]
    if not all(torch.isfinite(array).all() for array in arrays):
        raise ValueError("Non-finite training data")
    return (*arrays, used_files)


def imitation_loss(prediction, positions, grippers):
    movement_loss = nn.functional.mse_loss(prediction[..., :3], positions)
    gripper_loss = nn.functional.binary_cross_entropy_with_logits(
        prediction[..., 3], grippers
    )
    return movement_loss + 0.25 * gripper_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_FOLDER / "data/demonstrations_v1")
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "checkpoints/bc_v1")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(7)

    x, y, g, training_files = load_examples(args.data, "train")
    vx, vy, vg, validation_files = load_examples(args.data, "validation")
    # All scaling statistics come from training data only.
    mean = x.mean(0)
    std = x.std(0).clamp_min(0.01)
    position_scale = y.std((0, 1)).clamp_min(0.005)
    x, vx = (x - mean) / std, (vx - mean) / std
    y, vy = y / position_scale, vy / position_scale

    model = Policy(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history, best_checkpoint = [], None
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in torch.randperm(len(x)).split(64):
            # Small measurement noise helps avoid memorizing exact states.
            prediction = model(x[indices] + 0.01 * torch.randn_like(x[indices]))
            loss = imitation_loss(prediction, y[indices], g[indices])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            prediction = model(vx)
            validation_loss = imitation_loss(prediction, vy, vg).item()
            position_error = ((prediction[..., :3] - vy) * position_scale).abs().mean().item()
            gripper_accuracy = ((prediction[..., 3] > 0) == (vg > 0.5)).float().mean().item()
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_position_mae_m": position_error,
            "validation_gripper_accuracy": gripper_accuracy,
        }
        history.append(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_checkpoint = {
                "model": copy.deepcopy(model.state_dict()),
                "feature_mean": mean, "feature_std": std,
                "position_scale": position_scale,
                "input_size": x.shape[1], "horizon": HORIZON, **row,
            }
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(row), flush=True)

    torch.save(best_checkpoint, args.out / "policy.pt")
    (args.out / "training_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        key: value for key, value in best_checkpoint.items()
        if key not in ("model", "feature_mean", "feature_std", "position_scale")
    }
    summary.update({
        "seed": 7, "epochs_trained": args.epochs,
        "training_episodes": len(training_files), "training_transitions": len(x),
        "validation_episodes": len(validation_files), "validation_transitions": len(vx),
        "test_used": False, "normalization": "training data only",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "action_semantics": "future commanded position offsets and requested gripper open/close",
        "software": {"torch": str(torch.__version__), "numpy": np.__version__},
        "data_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in training_files + validation_files},
        "script_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("policy.py", "train_policy.py")},
    })
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Selected checkpoint:", json.dumps({key: summary[key] for key in (
        "epoch", "validation_position_mae_m", "validation_gripper_accuracy"
    )}), flush=True)
    print(f"Saved {args.out / 'policy.pt'}", flush=True)


if __name__ == "__main__":
    main()
