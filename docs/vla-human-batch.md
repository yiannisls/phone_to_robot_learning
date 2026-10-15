# SmolVLA accepts one human teleoperation example

The converted human episode now passes the SmolVLA interface check: reconstructed image, colour instruction, nine measured robot values, and four-value action labels. A normal batch and a batch at the final action both completed an independent forward/backward/AdamW update and returned finite 50 × 4 predictions on the RTX 4060 Laptop GPU. **No changed weights were saved and no prediction was executed on the robot.**

[Measured reports and temporary statistics](vla-human-batch.json) · [Human recording conversion](human-visual-episode.md) · [Earlier phone-data check](vla-real-batch.md)

## What this step checks

`test_vla_memory.py` now chooses a reader from the episode format. The existing phone reader retains its source-family, task and contact checks. The human reader checks `human_visual_v1`, the whole episode checksum, original raw-array checksums, source identity, training assignment, dimensions, nominal timing, colour instruction and action definition. Both use the same normalization calculation after their own source checks have passed.

For human data, the model receives only the image, instruction and `robot_state`: six measured joints, including the jaw, plus measured hand XYZ. Cube/bin coordinates and one-hot colour labels are used to verify the example but are not model inputs. The action label is next-minus-current measured hand XYZ plus the current recorded native gripper command. The human recording does not supply original mocap targets, so the loader does not fabricate them or pretend these labels are phone-style target deltas.

The model configuration takes the episode's validated state width: nine for human data and thirteen for the earlier phone example. Actions remain four-dimensional. SmolVLA's internal padding remains unchanged, so pretrained weight shapes load strictly as before. RGB preprocessing, instruction tokenization, LoRA targets, batch size and numerical checks are unchanged.

A successful check means the model can consume the data and compute a numerical update. It does not show that it uses the colour instruction correctly, learns a useful pickup, or produces executable control commands.

## Measured checks

Every row below starts from the same pinned pretrained model in a fresh process and takes one independent optimizer step. The rows are not consecutive steps of a task-training run.

| Batch | State values | Step / nominal time | Valid / padded actions | Loss | Result |
|---|---:|---|---|---:|---|
| Human example | 9 | 85 / 8.5 s | 50 / 0 | 2.22625923 | Passed |
| Human final action | 9 | 337 / 33.7 s | 1 / 49 | 2.90087152 | Passed |
| Earlier phone example rerun | 13 | 85 / 8.5 s | 50 / 0 | 1.27599049 | Exact historical report match |

Each check verified finite loss/gradients, a nonzero gradient, a checked adapter parameter changing after the update, finite denormalized predictions and output shape **1 × 50 × 4**. The human batches used the actual instruction “Put the red cube in the bin.” Loss values here are numerical checks, not task-quality scores; different examples and target normalizations do not make them comparable policy-performance measurements.

The final batch repeats the last action for tensor padding and marks those 49 entries as padding so they are excluded from the loss. It never reads into another episode. The synthetic `human_teleoperation` phase string is only a source label for report logging, not an inferred grasp/transport phase and not an input to the model.

The configuration uses one camera, batch size one, rank-four LoRA, 185,664 trainable parameters and BF16 autocast. Peak PyTorch allocation was **1,283.82 MiB, about 1.25 GiB**, and peak reserved memory was **1,364 MiB, about 1.33 GiB**, in both human checks. These exclude some driver/application memory and do not establish that full-model fine-tuning or larger batches fit.

Human action normalization followed by inverse scaling recovered the original targets with maximum absolute component error 6.74e-9 for the normal batch and 6.86e-10 for the final batch, in their respective per-component units. Predictions were checked for finiteness but were not clipped, executed or treated as valid robot commands.

## Training-only statistics and regression checks

Temporary state statistics use the 338 action-labelled rows, excluding the final observation. Action statistics use the 338 original actions before chunk padding. Both human runs saved identical mean/std values, with a minimum std of 1e-6. These are one-episode interface statistics; final training normalization must be refitted on the accepted training corpus and frozen for validation/test.

Original human validation IDs **36, 9 and 32** are rejected by this interface-test reader, which fits statistics. Nine negative human checks rejected checksum corruption, copies labelled with each validation ID, an incorrect instruction, an expanded state containing extra values, altered action labels, a validation split label and a DAgger source label. These used temporary modified copies of training episode 0, not actual held-out recordings. The original contact-failing phone candidate was also rejected. DAgger support remains a separate task.

The phone batch's complete JSON report and normalization exactly match the earlier saved run. The human source examples retain their original checksums and remain excluded from the accepted training corpus. All 1,418 frozen baseline artifacts still verify. No original dataset, checkpoint, split or earlier result was modified.

## Reproduce

First create the human example using the [conversion command](human-visual-episode.md). From the project root, with the isolated environment and pinned model cache already available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-vla/bin/python scripts/test_vla_memory.py \
  --episode work/my-human-example --step 85 \
  --out work/my-human-batch-check
```

Expected report fields are `status: passed`, `state_size: 9`, `action_size: 4`, `valid_actions: 50`, and `predicted_action_shape: [1, 50, 4]`. Use a fresh output folder. The offline flags use the existing pinned cache; omit them to allow initial model download as described in [setup](vla-setup.md).

For the boundary case, choose `--step 337` and another output folder. Expect one valid action and 49 padded actions. Each output contains `report.json` and `normalization.json`; no modified weights are saved.

Measured runs are retained under `work/vla-human-batch-20261015-01`, `work/vla-human-batch-end-20261015-01`, and `work/vla-phone-regression-20261015-02`, with exact test-script snapshots. Negative reader-check results are retained in `work/vla-human-reader-verification-20261015-01.json`. These files are local evidence; the small combined report above is included in Git.

## Next: check execution before corpus training

The human reader deliberately reports `contact_quality_evaluated: false`. An interface-only optimizer check does not certify the converted episode's contact physics or make it accepted training data. The phone reader's existing quality requirements remain intact; no human quality-pass flags are invented.

Before converting the full dataset and fitting a policy, replay a declared controller using the human achieved-motion labels under physics. Original position/orientation targets and velocities are missing, and a four-value action cannot express intentional wrist rotations. That execution check must retain its assumptions and outcomes separately from the reconstructed-pose video and from VLA policy evaluation. The seven DAgger segments remain separate until their conversion and split provenance are reviewed.

Presentation explanation: “SmolVLA now accepts one of my human demonstrations. I checked the data alignment, one numerical learning step and the episode boundary on my laptop GPU. This verifies the interface; robot performance still needs closed-loop testing.”
