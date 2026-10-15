"""Check whether pretrained SmolVLA can train with LoRA on this GPU."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

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


def load_policy(cache):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("work/vla-cache/hub"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    report = {
        "status": "started",
        "purpose": "Synthetic memory check; no task-performance measurement.",
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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run with access to the GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This test requires BF16 support.")
        torch.manual_seed(42)
        report["gpu"] = torch.cuda.get_device_name()
        report["total_vram_mib"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        torch.cuda.reset_peak_memory_stats()

        print("Loading pretrained SmolVLA and adding LoRA adapters...", flush=True)
        policy, config, camera = load_policy(args.cache)
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
        batch = make_batch(policy, config, camera)
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
