"""Fine-tune SmolVLA on the reconstructed human demonstrations.

Uses the original episode split and training-only normalization.
The first run is a short pilot; loss is not a robot success rate.

Usage:
    .venv-vla/bin/python scripts/train_vla.py \
        --dataset work/human-vla-dataset-20261015-01 \
        --out work/human-vla-pilot
"""

import argparse
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from convert_human_episode import checksum, read_visual_dataset
from test_vla_memory import (
    BACKBONE, BACKBONE_REVISION, POLICY, POLICY_REVISION,
    load_policy, memory_usage,
)

# Start small on the 8 GB GPU. Use --max-train-batches 0 for a full epoch.
EPOCHS = 1
BATCH_SIZE = 1
LR = 1e-4
MAX_TRAIN_BATCHES = 200
VALIDATION_SAMPLES = 8  # Evenly spaced windows from each validation episode.


class HumanChunkDataset(Dataset):
    """Pair one image and robot state with the following action chunk."""

    def __init__(self, root, manifest, normalizer, tokenizer, camera, chunk_size):
        self.root = root
        self.entries = manifest["episodes"]
        self.camera = camera
        self.chunk_size = chunk_size
        self.samples = [
            (entry["episode"], step)
            for entry in self.entries for step in range(entry["actions"])
        ]
        self.state_mean = np.array(normalizer["stats"]["observation.state"]["mean"])
        self.state_std = np.array(normalizer["stats"]["observation.state"]["std"])
        self.action_mean = np.array(normalizer["stats"]["action"]["mean"])
        self.action_std = np.array(normalizer["stats"]["action"]["std"])
        tokenizer.padding_side = "right"
        self.language = {}
        for instruction in sorted({entry["instruction"] for entry in self.entries}):
            self.language[instruction] = tokenizer(
                instruction.strip() + "\n", padding="max_length",
                truncation=True, max_length=48, return_tensors="pt",
            )

    @lru_cache(maxsize=2)
    def read_episode(self, episode):
        # Keep only two recordings in RAM instead of loading every image at once.
        path = self.root / self.entries[episode]["folder"] / "episode.npz"
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in
                    ["images", "robot_state", "actions", "instruction"]}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_id, step = self.samples[index]
        episode = self.read_episode(episode_id)
        state = (episode["robot_state"][step] - self.state_mean) / self.state_std
        actions = episode["actions"][step:step + self.chunk_size]
        actions = (actions - self.action_mean) / self.action_std
        valid = len(actions)

        # The last action fills the array, but padded rows do not enter the loss.
        chunk = np.repeat(actions[-1:], self.chunk_size, axis=0)
        chunk[:valid] = actions
        language = self.language[episode["instruction"].item()]
        batch = {
            self.camera: torch.from_numpy(episode["images"][step]).permute(2, 0, 1).float() / 255,
            "observation.state": torch.tensor(state, dtype=torch.float32),
            "observation.language.tokens": language["input_ids"][0],
            "observation.language.attention_mask": language["attention_mask"][0].bool(),
            "action": torch.tensor(chunk, dtype=torch.float32),
            "action_is_pad": torch.arange(self.chunk_size) >= valid,
        }
        return batch, episode_id, step


def train_one_epoch(model, loader, optimizer, device, max_batches, log):
    model.train()
    total_loss = 0.0
    n_actions = 0
    n_batches = 0

    for batch, episode_ids, steps in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()

        valid_actions = int((~batch["action_is_pad"]).sum())
        total_loss += loss.item() * valid_actions
        n_actions += valid_actions
        n_batches += 1
        log.write(json.dumps({
            "batch": n_batches, "episodes": episode_ids.tolist(),
            "steps": steps.tolist(), "loss": loss.item(),
        }) + "\n")
        log.flush()
        if n_batches % 25 == 0:
            print(f"  Batch {n_batches}: loss {loss.item():.6f}", flush=True)
        if max_batches and n_batches >= max_batches:
            break

    return {"loss": total_loss / n_actions, "batches": n_batches}


@torch.no_grad()
def evaluate(model, loader, device, seed):
    model.eval()
    total_loss = 0.0
    n_actions = 0

    # SmolVLA's training loss samples noise. Reuse it for each evaluation.
    # Restore the training RNG afterward so evaluation cannot change training.
    with torch.random.fork_rng(devices=[device.index or 0]):
        for index, (batch, _, _) in enumerate(loader):
            torch.manual_seed(seed + index)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Evaluation loss is not finite.")
            valid_actions = int((~batch["action_is_pad"]).sum())
            total_loss += loss.item() * valid_actions
            n_actions += valid_actions

    return total_loss / n_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=MAX_TRAIN_BATCHES)
    parser.add_argument("--cache", type=Path, default=Path("work/vla-cache/hub"))
    args = parser.parse_args()
    if args.epochs < 1 or args.max_train_batches < 0:
        parser.error("Use positive epochs and a nonnegative batch limit.")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires CUDA with BF16 support.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)

    # Load the original human dataset and its saved normalization.
    root = args.dataset.expanduser().resolve()
    manifest, normalizer = read_visual_dataset(root)
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(root / manifest["episodes"][0]["folder"] / "episode.npz") as data:
        example = {key: data[key] for key in ["images", "robot_state"]}

    # Load pretrained SmolVLA and train only its rank-four LoRA adapters.
    torch.cuda.reset_peak_memory_stats()
    model, config, camera = load_policy(args.cache, example)
    del example
    tokenizer = model.get_base_model().model.vlm_with_expert.processor.tokenizer
    if config.tokenizer_max_length != 48:
        raise ValueError("This dataset setup expects the pinned 48-token configuration.")
    dataset = HumanChunkDataset(
        root, manifest, normalizer, tokenizer, camera, config.chunk_size
    )

    # Split complete episodes, keeping the original validation IDs.
    train_indices = [
        i for i, (episode, _) in enumerate(dataset.samples)
        if episode in manifest["training_episode_ids"]
    ]
    val_indices = []
    for episode in manifest["validation_episode_ids"]:
        indices = [i for i, (e, _) in enumerate(dataset.samples) if e == episode]
        positions = np.unique(np.linspace(0, len(indices) - 1, VALIDATION_SAMPLES, dtype=int))
        val_indices.extend(indices[int(position)] for position in positions)
    probe_rng = np.random.default_rng(args.seed + 1)
    train_probe_indices = probe_rng.choice(train_indices, size=24, replace=False).tolist()

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
    train_probe_loader = DataLoader(Subset(dataset, train_probe_indices), batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training examples: {len(train_indices)}", flush=True)
    print(f"Validation windows: {len(val_indices)} from 3 episodes", flush=True)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}", flush=True)
    probes = {
        "train": [dataset.samples[i] for i in train_probe_indices],
        "validation": [dataset.samples[i] for i in val_indices],
    }
    (args.out / "loss_probes.json").write_text(json.dumps(probes, indent=2) + "\n")
    (args.out / "normalization.json").write_text(json.dumps(normalizer, indent=2) + "\n")
    initial_loss = {
        "train": evaluate(model, train_probe_loader, device, args.seed + 1000),
        "validation": evaluate(model, val_loader, device, args.seed + 1000),
    }
    print(f"Initial fixed-window losses: {initial_loss}", flush=True)

    # Train, then compare the same fixed windows after each epoch.
    history = []
    for epoch in range(1, args.epochs + 1):
        with (args.out / f"epoch_{epoch:03d}.jsonl").open("w") as log:
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, args.max_train_batches, log
            )
        train_probe_loss = evaluate(model, train_probe_loader, device, args.seed + 1000)
        val_loss = evaluate(model, val_loader, device, args.seed + 1000)
        history.append({
            "epoch": epoch, "training_loss": train_metrics["loss"],
            "updates": train_metrics["batches"],
            "fixed_train_loss": train_probe_loss, "validation_loss": val_loss,
        })
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"Epoch {epoch}/{args.epochs} | updates {train_metrics['batches']} | "
            f"train {train_metrics['loss']:.6f} | val {val_loss:.6f}", flush=True,
        )

    # Save the final adapter. The pinned pretrained base is loaded separately.
    adapter_path = args.out / "adapter"
    model.peft_config["default"].base_model_name_or_path = POLICY
    model.peft_config["default"].revision = POLICY_REVISION
    model.save_pretrained(adapter_path, save_embedding_layers=False)
    saved_config = deepcopy(config)
    saved_config.vlm_model_name = BACKBONE
    saved_config.pretrained_path = Path(POLICY)
    saved_config.save_pretrained(adapter_path)
    report = {
        "status": "completed", "experiment": "human_only_smolvla_lora_pilot",
        "policy": POLICY, "policy_revision": POLICY_REVISION,
        "backbone_revision": BACKBONE_REVISION,
        "seed": args.seed, "epochs": args.epochs, "learning_rate": args.lr,
        "batch_size": BATCH_SIZE, "lora_rank": 4,
        "max_train_batches_per_epoch": args.max_train_batches,
        "completed_updates": sum(row["updates"] for row in history),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "initial_loss": initial_loss, "history": history,
        "gpu": torch.cuda.get_device_name(device), "memory": memory_usage(),
        "versions": {name: version(name) for name in ["torch", "lerobot", "transformers", "peft"]},
        "manifest_sha256": checksum(root / "manifest.json"),
        "normalization_sha256": checksum(root / "normalization.json"),
        "script_sha256": checksum(Path(__file__)),
        "data_loader_sha256": checksum(Path(__file__).with_name("convert_human_episode.py")),
        "model_loader_sha256": checksum(Path(__file__).with_name("test_vla_memory.py")),
        "adapter_sha256": checksum(adapter_path / "adapter_model.safetensors"),
        "phone_data_included": False, "dagger_included": False,
        "images_reconstructed": True, "robot_success_measured": False,
        "checkpoint_selection": "Final epoch; no best-validation selection",
        "limitation": "Fixed-window flow-matching loss; contact physics remains unvalidated.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Final adapter: {adapter_path}", flush=True)


if __name__ == "__main__":
    main()
