# Local VLA setup: hardware check

## Verified on 15 October 2026

| Check | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8,188 MiB |
| Free VRAM at check | 7,794 MiB |
| Driver | 595.99.02 |
| Existing lab PyTorch | 2.14.1+cu130 |
| CUDA tensor calculation | Passed using host GPU access |
| BF16 forward/backward + AdamW update | Passed: three synthetic linear-model steps |
| Isolated VLA environment | LeRobot 0.6.1, Transformers 5.5.4, PEFT 0.21.2; imports passed |
| Free Linux disk space | 13 GB before installation; 7.2 GB afterwards |
| Real SmolVLA forward/backward test | **Pending** |

A restricted process could not access the NVIDIA driver, but the host-access check succeeded. No driver repair was performed or needed for that check. VRAM availability changes with other applications.

## Keep the baseline environment intact

The dependency dry run for `lerobot[smolvla,peft]==0.6.1` would replace packages in the existing lab environment. LeRobot 0.6.1 declares PyTorch >=2.7,<2.12 and NumPy >=2,<2.3, whereas the current lab uses PyTorch 2.14.1 and NumPy 2.5.3. The dry run installed nothing.

The separate `.venv-vla` environment is now installed and excluded from Git. It uses PyTorch 2.11.0+cu130 and torchvision 0.26.0+cu130. Package compatibility, SmolVLA and LoRA imports, CUDA tensor execution, and BF16 support passed. No packages in the baseline environment were replaced.

Recreate this dependency setup from the repository root with Python 3.12 and uv installed:

```bash
uv venv --python python3.12 .venv-vla
uv pip install --python .venv-vla/bin/python --no-cache --link-mode hardlink \
  --torch-backend cu130 -r requirements-vla-lock.txt
uv pip check --python .venv-vla/bin/python
```

`requirements-vla.txt` lists the direct requirements; `requirements-vla-lock.txt` records all 76 installed package versions for this Linux/CUDA setup. This is a version snapshot, not a wheel archive or a guarantee of compatibility with other operating systems or GPU drivers. The CUDA 13.0 packages require a compatible NVIDIA driver. Other platforms need their own dependency resolution.

The install used temporary package caching and hard links to avoid retaining duplicate installed files. Free space fell from approximately 13 GB to 7.2 GB. Model weights, training data, and output checkpoints will require additional space.

[Declared dependencies](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)

## First memory experiment

The starting candidate is pretrained SmolVLA. Begin with one camera, batch size one, frozen vision encoding, and a small LoRA adaptation if supported by the pinned implementation. Measure loading, forward, backward and optimizer-step peaks on the actual GPU. Start from the pretrained weights, not a randomly initialized architecture.

An initial synthetic-batch test establishes only whether that configuration fits and can update parameters. It cannot establish task performance. Repeat with the real synchronized multicube data before committing to a long training run. Add cameras or batch size only after memory headroom is measured. Do not reduce image resolution without checking the model's actual internal resize behavior.

LeRobot's hardware guide estimates 10–16 GB for its small-VLA group at batch size eight and default AdamW. Smaller batches and frozen components may reduce usage, but an 8 GB training fit has not yet been demonstrated here. LoRA is supported for SmolVLA; its memory use depends on configuration.

[Hardware guide](https://huggingface.co/docs/lerobot/main/hardware_guide) · [SmolVLA PEFT support](https://huggingface.co/docs/lerobot/main/peft_training)

No model weights were downloaded, no full training run was started, and the working BC environment was unchanged. Before adding a memory-test script, its filename and complete proposed code will be reviewed in the guided workflow.

The small CUDA training check used a 1024-to-1024 linear layer, batch size two and BF16 autocast. All losses and gradients were finite; losses were 1.3477, 1.2310 and 1.1212. Peak allocated GPU memory was 39.3 MiB. This checks software execution only and is not evidence that SmolVLA fits. No trained weights were saved.

Storage was freed before creating the isolated environment. The next stage is a reviewed memory-test script and a pretrained-model download; neither has run yet.
