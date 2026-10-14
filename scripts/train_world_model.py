"""Learn state changes from training episodes; select using validation only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from world_model import PHYSICAL_SIZE, SOURCE_SPLITS, WorldModel, load_episodes, make_state

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def training_pairs(episodes):
    inputs, changes = [], []
    for _, episode in episodes:
        state = make_state(episode)
        inputs.append(np.concatenate([state[:-1], episode["actions"]], axis=1))
        changes.append(np.diff(state[:, :PHYSICAL_SIZE], axis=0))
    return (torch.from_numpy(np.concatenate(inputs)),
            torch.from_numpy(np.concatenate(changes)))


def extra_training_pairs(folder, profile):
    """Load simulator branches only after checking their source split and hashes."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest["complete"] or manifest["split"] != "train" or manifest["profile"] != profile:
        raise ValueError("Extra transitions must be complete, training-only and profile-compatible")
    inputs, changes = [], []
    for name, digest in manifest["files"].items():
        if Path(name).name != name or SOURCE_SPLITS[name.split("_aug")[0].removesuffix(".npz")] != "train":
            raise ValueError(f"Non-training source: {name}")
        path = folder / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed extra data: {name}")
        with np.load(path, allow_pickle=False) as branch:
            assert branch["split"].item() == "train"
            assert branch["source_video"].item() == name.split("_aug")[0].removesuffix(".npz")
            state, action = branch["states"], branch["actions"]
            assert state.shape[-1] == 35 and action.shape == (len(state), state.shape[1] - 1, 4)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            inputs.append(np.concatenate([state[:, :-1], action], axis=-1).reshape(-1, 39))
            changes.append(np.diff(state[..., :PHYSICAL_SIZE], axis=1).reshape(-1, PHYSICAL_SIZE))
    x, y = torch.from_numpy(np.concatenate(inputs)), torch.from_numpy(np.concatenate(changes))
    assert len(x) == manifest["transitions"]
    return x, y, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_FOLDER / "data/demonstrations_v2")
    parser.add_argument("--out", type=Path, default=PROJECT_FOLDER / "checkpoints/world_model_v1")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--extra-transitions", type=Path, help="Optional training-only simulator branches")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    training, profile = load_episodes(args.data, "train")
    validation, _ = load_episodes(args.data, "validation", profile)
    x, y = training_pairs(training)
    vx, vy = training_pairs(validation)
    ordinary_training_transitions = len(x)
    extra_manifest = None
    if args.extra_transitions is not None:
        extra_x, extra_y, extra_manifest = extra_training_pairs(args.extra_transitions, profile)
        x, y = torch.cat([x, extra_x]), torch.cat([y, extra_y])
    args.out.mkdir(parents=True, exist_ok=False)

    checkpoint = {
        "input_mean": x.mean(0), "input_std": x.std(0).clamp_min(0.01),
        "delta_scale": y.std(0).clamp_min(0.001), "profile": profile,
        "state_schema": "qpos13_qvel12_hand3_target3_gripper1_goal3",
        "seed": 7,
    }
    x = (x - checkpoint["input_mean"]) / checkpoint["input_std"]
    vx = (vx - checkpoint["input_mean"]) / checkpoint["input_std"]
    y = y / checkpoint["delta_scale"]
    vy = vy / checkpoint["delta_scale"]
    model = WorldModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, history = float("inf"), []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in torch.randperm(len(x)).split(256):
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            validation_loss = nn.functional.mse_loss(model(vx), vy).item()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation_loss": validation_loss}
        history.append(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            checkpoint.update(model=copy.deepcopy(model.state_dict()),
                              epoch=epoch, validation_loss=validation_loss)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(row), flush=True)

    torch.save(checkpoint, args.out / "world_model.pt")
    (args.out / "training_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "selected_epoch": checkpoint["epoch"], "epochs_trained": args.epochs,
        "validation_loss": best_loss, "seed": 7, "profile": profile,
        "training_episodes": len(training), "training_transitions": len(x),
        "ordinary_training_transitions": ordinary_training_transitions,
        "extra_training_transitions": len(x) - ordinary_training_transitions,
        "extra_data_manifest": extra_manifest,
        "validation_episodes": len(validation), "validation_transitions": len(vx),
        "failed_training_episodes_included": sum(not bool(ep["success"]) for _, ep in training),
        "test_used": False, "normalization": "training data only",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_schema": checkpoint["state_schema"],
        "selection": "validation normalized physical-delta MSE",
        "software": {"torch": str(torch.__version__), "numpy": np.__version__},
        "data_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path, _ in training + validation},
        "script_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("world_model.py", "train_world_model.py")},
    }
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Selected epoch {checkpoint['epoch']}; saved {args.out / 'world_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
