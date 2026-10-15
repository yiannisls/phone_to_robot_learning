"""Check whether pretrained SmolVLA can train with LoRA on this GPU."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from generate_demo import SOURCE_SPLITS

POLICY = "lerobot/smolvla_base"
POLICY_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def memory_usage():
    """Report PyTorch memory, including its reusable reserved memory."""
    torch.cuda.synchronize()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def read_episode(folder):
    """Read a candidate for an interface test, without adopting it for training."""
    summary = json.loads((folder / "summary.json").read_text())
    path = folder / "episode.npz"
    keys = [
        "images", "robot_state", "actions", "time_s", "phase",
        "instruction", "source_video", "split",
    ]
    with np.load(path, allow_pickle=False) as saved:
        episode = {key: saved[key].copy() for key in keys}

    family = episode["source_video"].item()
    if SOURCE_SPLITS.get(family) != "train":
        raise ValueError("Interface-test statistics must use a training family.")
    if episode["split"].item() != "train" or summary["source_family"] != family:
        raise ValueError("Episode and report have inconsistent source labels.")
    if summary["split"] != "train":
        raise ValueError("The report is not assigned to training.")
    if not summary["task_success"] or not summary["quality_pass"]:
        raise ValueError("Use a candidate that passed the task and contact checks.")
    if summary["max_target_penetration_m"] > 0.002:
        raise ValueError("The candidate exceeds the contact-depth limit.")
    if summary["max_distractor_displacement_m"] >= 0.01:
        raise ValueError("The candidate moved a distractor too far.")

    count = len(episode["actions"])
    if count == 0 or episode["actions"].shape != (count, 4):
        raise ValueError("Expected a nonempty sequence of four-value actions.")
    if episode["robot_state"].shape != (count + 1, 13):
        raise ValueError("Expected thirteen-value states and a final observation.")
    images = episode["images"]
    if images.ndim != 4 or images.shape[0] != count + 1 or images.shape[-1] != 3:
        raise ValueError("Expected one RGB image per observation.")
    if images.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB images.")
    if episode["time_s"].shape != (count + 1,) or len(episode["phase"]) != count:
        raise ValueError("Episode timing or phase labels are misaligned.")
    if not np.allclose(np.diff(episode["time_s"]), summary["control_dt_s"]):
        raise ValueError("Observation timestamps are not evenly spaced.")
    if summary["actions"] != count or summary["observations"] != count + 1:
        raise ValueError("Report counts do not match the episode.")
    if episode["instruction"].item() != summary["instruction"]:
        raise ValueError("Episode and report instructions differ.")

    stats = {}
    for key, values in [
        ("observation.state", episode["robot_state"][:-1]),
        ("action", episode["actions"]),
    ]:
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {key}.")
        stats[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
        }
    normalization = {
        "purpose": "Temporary one-episode interface test; refit on the training corpus.",
        "source_family": family,
        "split": "train",
        "transitions": count,
        "minimum_std": 1e-6,
        "stats": stats,
    }
    metadata = {
        "source_family": family,
        "split": "train",
        "visual_episode_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate_added_to_training": False,
        "task_performance_measured": False,
    }
    return episode, normalization, metadata


def load_policy(cache, episode=None):
    # The policy checkpoint already contains the backbone weights.
    # Download only the backbone configuration and tokenizer separately.
    policy_folder = snapshot_download(
        POLICY,
        revision=POLICY_REVISION,
        cache_dir=cache,
        allow_patterns=["config.json", "model.safetensors"],
    )
    backbone_folder = snapshot_download(
        BACKBONE,
        revision=BACKBONE_REVISION,
        cache_dir=cache,
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
    )

    config = SmolVLAConfig.from_pretrained(policy_folder)
    config.device = "cpu"
    config.vlm_model_name = backbone_folder
    config.load_vlm_weights = False

    # Keep one camera. This changes the inputs, not the weight shapes.
    camera = next(iter(config.image_features))
    config.input_features = {
        key: feature
        for key, feature in config.input_features.items()
        if key not in config.image_features or key == camera
    }
    config.empty_cameras = 0
    if episode is not None:
        height, width = episode["images"].shape[1:3]
        config.input_features["observation.state"] = PolicyFeature(
            FeatureType.STATE, (13,)
        )
        config.input_features[camera] = PolicyFeature(
            FeatureType.VISUAL, (3, height, width)
        )
        config.output_features["action"] = PolicyFeature(
            FeatureType.ACTION, (4,)
        )

    # Strict loading checks that the complete policy weights are present.
    policy = SmolVLAPolicy.from_pretrained(
        policy_folder, config=config, strict=True
    )
    config.pretrained_path = Path(policy_folder)
    config.load_vlm_weights = True
    policy = policy.wrap_with_peft(
        peft_cli_overrides={"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
    )
    return policy.to("cuda"), config, camera


def make_batch(policy, config, camera):
    """Create artificial inputs for a memory check, not task training."""
    tokenizer = policy.get_base_model().model.vlm_with_expert.processor.tokenizer
    language = tokenizer(
        ["Put the green cube in the bin."],
        padding="max_length",
        truncation=True,
        max_length=config.tokenizer_max_length,
        return_tensors="pt",
    )
    state_size = config.input_features["observation.state"].shape[0]
    action_size = config.action_feature.shape[0]
    return {
        camera: torch.rand(1, *config.image_features[camera].shape, device="cuda"),
        "observation.state": torch.zeros(1, state_size, device="cuda"),
        "observation.language.tokens": language["input_ids"].to("cuda"),
        "observation.language.attention_mask": language["attention_mask"].bool().to("cuda"),
        "action": torch.randn(1, config.chunk_size, action_size, device="cuda"),
        "action_is_pad": torch.zeros(1, config.chunk_size, dtype=torch.bool, device="cuda"),
    }


def make_episode_batch(policy, config, camera, episode, normalization, step):
    """Pair the image/state at a step with its following action targets."""
    count = len(episode["actions"])
    if not 0 <= step < count:
        raise ValueError(f"Choose a step from 0 to {count - 1}.")

    stats = normalization["stats"]
    state_mean = np.asarray(stats["observation.state"]["mean"])
    state_std = np.asarray(stats["observation.state"]["std"])
    action_mean = np.asarray(stats["action"]["mean"])
    action_std = np.asarray(stats["action"]["std"])

    state = (episode["robot_state"][step] - state_mean) / state_std
    targets = episode["actions"][step:step + config.chunk_size]
    valid = len(targets)
    targets = (targets - action_mean) / action_std
    padded = np.repeat(targets[-1:], config.chunk_size, axis=0)
    padded[:valid] = targets

    tokenizer = policy.get_base_model().model.vlm_with_expert.processor.tokenizer
    tokenizer.padding_side = "right"
    language = tokenizer(
        [episode["instruction"].item().strip() + "\n"],
        padding="max_length",
        truncation=True,
        max_length=config.tokenizer_max_length,
        return_tensors="pt",
    )
    image = torch.tensor(
        episode["images"][step], dtype=torch.float32, device="cuda"
    ).permute(2, 0, 1).unsqueeze(0) / 255.0

    return {
        camera: image,
        "observation.state": torch.tensor(
            state, dtype=torch.float32, device="cuda"
        ).unsqueeze(0),
        "observation.language.tokens": language["input_ids"].to("cuda"),
        "observation.language.attention_mask": language["attention_mask"].bool().to("cuda"),
        "action": torch.tensor(
            padded, dtype=torch.float32, device="cuda"
        ).unsqueeze(0),
        "action_is_pad": (
            torch.arange(config.chunk_size, device="cuda") >= valid
        ).unsqueeze(0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=Path("work/vla-cache/hub"))
    args = parser.parse_args()
    if args.episode is None and args.step != 0:
        parser.error("--step requires --episode")
    args.out.mkdir(parents=True, exist_ok=False)

    report = {
        "status": "started",
        "purpose": "Real-episode interface check; no task-performance measurement." if args.episode else "Synthetic memory check; no task-performance measurement.",
        "policy": POLICY,
        "policy_revision": POLICY_REVISION,
        "backbone_revision": BACKBONE_REVISION,
        "seed": 42,
        "batch_size": 1,
        "cameras": 1,
        "lora_rank": 4,
        "precision": "Pretrained parameter dtypes with BF16 autocast",
        "versions": {
            name: version(name)
            for name in ["torch", "lerobot", "transformers", "peft"]
        },
    }
    try:
        episode = None
        normalization = None
        if args.episode is not None:
            episode, normalization, metadata = read_episode(args.episode)
            if not 0 <= args.step < len(episode["actions"]):
                raise ValueError("Selected step is outside the episode.")
            report["episode"] = str(args.episode)
            report.update(metadata)
            (args.out / "normalization.json").write_text(
                json.dumps(normalization, indent=2) + "\n"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run with access to the GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This test requires BF16 support.")
        torch.manual_seed(42)
        report["gpu"] = torch.cuda.get_device_name()
        report["total_vram_mib"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        torch.cuda.reset_peak_memory_stats()

        print("Loading pretrained SmolVLA and adding LoRA adapters...", flush=True)
        policy, config, camera = load_policy(args.cache, episode)
        report["after_loading"] = memory_usage()
        report["image_resize"] = config.resize_imgs_with_padding
        report["action_chunk"] = config.chunk_size
        parameters = [p for p in policy.parameters() if p.requires_grad]
        report["trainable_parameters"] = sum(p.numel() for p in parameters)
        report["total_parameters"] = sum(p.numel() for p in policy.parameters())
        report["parameter_dtypes"] = {}
        for parameter in policy.parameters():
            dtype = str(parameter.dtype)
            report["parameter_dtypes"][dtype] = (
                report["parameter_dtypes"].get(dtype, 0) + parameter.numel()
            )
        report["state_size"] = config.input_features["observation.state"].shape[0]
        report["action_size"] = config.action_feature.shape[0]

        policy.train()
        if episode is None:
            batch = make_batch(policy, config, camera)
        else:
            batch = make_episode_batch(
                policy, config, camera, episode, normalization, args.step
            )
            valid = int((~batch["action_is_pad"]).sum().item())
            report["selected_step"] = args.step
            report["selected_time_s"] = float(episode["time_s"][args.step])
            report["selected_phase"] = str(episode["phase"][args.step])
            report["instruction"] = episode["instruction"].item()
            report["valid_actions"] = valid
            report["padded_actions"] = config.chunk_size - valid
            report["input_shapes"] = {key: list(value.shape) for key, value in batch.items()}
            stats = normalization["stats"]
            physical_targets = (
                batch["action"][0, :valid].detach().cpu().numpy()
                * np.asarray(stats["action"]["std"])
                + np.asarray(stats["action"]["mean"])
            )
            expected = episode["actions"][args.step:args.step + valid]
            if not np.allclose(physical_targets, expected, atol=1e-6):
                raise RuntimeError("Action scaling did not round-trip to simulator units.")
            report["action_roundtrip_max_error"] = float(np.abs(physical_targets - expected).max())
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        print("Running forward, backward, and optimizer update...", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("The loss is not finite.")
        report["loss"] = loss.item()
        report["after_forward"] = memory_usage()

        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Gradients are missing or not finite.")
        changed_parameter = next(
            (p for p in parameters if p.grad is not None and p.grad.abs().max().item() > 0),
            None,
        )
        if changed_parameter is None:
            raise RuntimeError("All gradients are zero.")
        before_update = changed_parameter.detach().clone()
        report["after_backward"] = memory_usage()

        optimizer.step()
        if torch.equal(before_update, changed_parameter.detach()):
            raise RuntimeError("The optimizer did not change the checked parameter.")
        report["after_optimizer"] = memory_usage()
        if episode is not None:
            # Check the output interface, without executing predictions in the simulator.
            policy.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = policy.get_base_model().predict_action_chunk(batch)
            if tuple(predicted.shape) != (1, config.chunk_size, 4):
                raise RuntimeError("Unexpected predicted action shape.")
            physical_prediction = (
                predicted.float().cpu().numpy()
                * np.asarray(normalization["stats"]["action"]["std"])
                + np.asarray(normalization["stats"]["action"]["mean"])
            )
            if not np.isfinite(physical_prediction).all():
                raise RuntimeError("Predicted physical actions are not finite.")
            report["predicted_action_shape"] = list(predicted.shape)
            report["predicted_physical_actions_finite"] = True
            report["prediction_executed_in_simulator"] = False
            report["after_prediction"] = memory_usage()
        report["status"] = "passed"
        print(json.dumps(report, indent=2))
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
