# Local VLA setup and memory test

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
| Free Linux disk space | 13 GB before installation; approximately 4.7 GB after the model test |
| Pretrained SmolVLA + LoRA forward/backward/update | **Passed on a synthetic batch** |

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

`requirements-vla.txt` lists the direct requirements; `requirements-vla-lock.txt` records the installed package versions for this Linux/CUDA setup, including MuJoCo 3.14.0 for the visual recorder. This is a version snapshot, not a wheel archive or a guarantee of compatibility with other operating systems or GPU drivers. The CUDA 13.0 packages require a compatible NVIDIA driver. Other platforms need their own dependency resolution.

The install used temporary package caching and hard links to avoid retaining duplicate installed files. Free space fell from approximately 13 GB to 7.2 GB. Model weights, training data, and output checkpoints will require additional space.

[Declared dependencies](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)

## First memory experiment: passed

The test uses pretrained SmolVLA with one camera, batch size one, frozen base parameters, and rank-four LoRA adapters on LeRobot's default SmolVLA targets: action-expert query/value projections and state/action/time projections. There are 185,664 trainable parameters out of 450,231,840 total parameters. The checkpoint is loaded strictly, checking that all expected model weights are present.

The full policy checkpoint contains the backbone weights. The script constructs the architecture from the pinned backbone configuration, then loads the complete pretrained policy. It does not download a second backbone weight file. Model revisions are fixed in the script, and configuration/tokenizer files and weights are cached under `work/vla-cache/hub`.

Run from the project root with the isolated environment:

```bash
.venv-vla/bin/python scripts/test_vla_memory.py \
  --out work/vla-memory-local-01
```

Use a new output folder for every attempt. The script saves `report.json`; it does not save changed weights or a task-trained checkpoint. The first run downloads approximately 907 MB of policy weights plus small backbone metadata/tokenizer files.

| Measurement | Result |
|---|---|
| Input | Random RGB image, zero state, real tokenized instruction, random action targets |
| State / action dimensions | 6 / 6, inherited from the base checkpoint for this memory test |
| Camera input / internal resize | 256 × 256 / 512 × 512 |
| Action chunk / language length | 50 / 48 |
| Parameter dtypes | 301,234,320 BF16; 148,997,520 FP32 |
| Forward precision | BF16 autocast |
| Loss, seed 42 | 2.6188187599; not a task-quality metric |
| Peak PyTorch allocation | 1,283.69 MiB, approximately 1.25 GiB |
| Peak PyTorch reserved memory | 1,362 MiB, approximately 1.33 GiB |
| Verification | Finite loss and gradients, nonzero gradient, checked adapter parameter changed after AdamW |

These peaks are cumulative across loading, forward, backward, and the optimizer step. They describe PyTorch memory, not total device use by the driver and other applications. The six-dimensional interface is inherited from the pretrained model for this synthetic test; the simulator's real observation/action mapping is not established by this result.

[Saved machine-readable report](vla-memory.json). Raw attempts are preserved locally in `work/vla-memory-20261015-01`, `-02`, and `-03`. Attempt 01 loaded the weights but failed while accessing the tokenizer through the LoRA wrapper. Attempt 02 passed after fixing that access. Attempt 03 also passed and records the corrected precision metadata; it produced the same loss and memory peaks as attempt 02.

An initial synthetic-batch test establishes only whether that configuration fits and can update parameters. It cannot establish task performance. Repeat with the real synchronized multicube data before committing to a long training run. Add cameras or batch size only after memory headroom is measured. Do not reduce image resolution without checking the model's actual internal resize behavior.

LeRobot's hardware guide estimates 10–16 GB for its small-VLA group at batch size eight and default AdamW. This rank-four, single-camera synthetic test fits on the RTX 4060 Laptop GPU, but it does not measure full-parameter fine-tuning, longer runs, or task-specific batches. LoRA is supported for SmolVLA; its memory use depends on configuration.

[Hardware guide](https://huggingface.co/docs/lerobot/main/hardware_guide) · [SmolVLA PEFT support](https://huggingface.co/docs/lerobot/main/peft_training)

Pretrained weights have now been downloaded and one synthetic optimizer step has passed. No real-data fine-tuning run was started, no changed weights were saved, and the working BC environment, data splits, checkpoints and evaluation results remain unchanged. The script was previewed in full before creation.

The small CUDA training check used a 1024-to-1024 linear layer, batch size two and BF16 autocast. All losses and gradients were finite; losses were 1.3477, 1.2310 and 1.1212. Peak allocated GPU memory was 39.3 MiB. This checks software execution only and is not evidence that SmolVLA fits. No trained weights were saved.

One synchronized multicube candidate has now been recorded and inspected. Its image/action alignment passed, but placement penetration exceeded the quality threshold; it remains excluded from training. See [the visual-episode report](visual-episode.md). Correct placement and review camera framing before expanding the dataset or fine-tuning. The phone source-family split remains fixed; any new augmentation must keep all descendants of each recording in that recording's split.
