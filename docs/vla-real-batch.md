# SmolVLA with one recorded robot episode

Pretrained SmolVLA now accepts the actual episode interface: one RGB image, thirteen robot-state values, a language instruction, and four-value action targets. An isolated forward/backward/AdamW update passed on the RTX 4060 Laptop GPU. The model also returned a finite 50 × 4 action chunk. **No task-trained checkpoint was saved and no predicted action was executed in the simulator.**

This checks data plumbing and compute, not manipulation performance or language grounding. The input remains one development candidate from a fixed training layout, excluded from the dataset. It does not become accepted training data merely because this interface test passes.

[Machine-readable results and temporary statistics](vla-real-batch.json) · [Source episode and scripted placement](visual-placement.md) · [Environment setup](vla-setup.md)

## Input and output

| Input | Shape | Preparation |
|---|---|---|
| Camera image | 1 × 3 × 240 × 320 | Recorded RGB uint8 converted to float32 [0, 1]; model resizes/pads internally to 512 × 512 and maps to [-1, 1] |
| Robot state | 1 × 13 | Six joint angles, hand XYZ, commanded target XYZ, applied native gripper command; mean/std normalization |
| Instruction | 1 × 48 tokens plus attention mask | Actual instruction, “Put the green cube in the bin.”, with the trailing newline used by LeRobot's SmolVLA processor |
| Action targets | 1 × 50 × 4 | Three world-frame hand-target deltas per control step plus requested native gripper target; mean/std normalization |
| Action-padding mask | 1 × 50 | Repeated actions beyond the episode end are marked as padding and excluded from the loss |

The model's feature definitions are changed to thirteen state values and four actions; internal state/action padding remains 32. The pretrained layers therefore keep their original weight shapes and load strictly. This proves shape compatibility, not that the pretrained policy already understands our Cartesian command semantics. Its original action convention differs, so task-specific learning and evaluation are still required.

Cube coordinates, target colour one-hot labels, bin coordinates and phase labels are not given to the model. Phase and timing metadata are used to describe the selected sample. The data-generation teacher used simulator cube pose and geometry; that privileged information is not part of this model batch.

A successful input/output check does not demonstrate that predictions depend correctly on the image or instruction. Later evaluation must compare the same scene with different colour instructions and report wrong-object errors separately.

## Temporary normalization and split protection

Statistics use the 251 action-labelled transitions of this `IMG_5539` training candidate. State statistics use `robot_state[:-1]`, excluding the terminal observation with no following action. Action statistics use the 251 original recorded actions, before any 50-step chunk padding. Each feature uses population mean/std, with a minimum std of 1e-6.

The same statistics were saved for both real-data checks. They are temporary one-episode interface statistics, **not final training normalization**. Refit and freeze them using only the complete accepted training corpus before fine-tuning; apply the frozen statistics to validation/test without refitting.

The loader checks source-family membership against the existing split, report/episode consistency, image/state/action dimensions, timestamps, task completion and contact-quality measurements. Negative checks rejected the original contact-failing candidate and temporary copies labelled as the `IMG_5542` validation or `IMG_5543` test family. These checks used copies; no held-out recording was used for fitting statistics or testing the model.

Action normalization/denormalization recovered the input targets with maximum absolute error 4.43e-9 for the carry batch and 2.92e-8 for the episode-end batch, measured across the stored per-component action units. Model predictions were also denormalized and checked for finite values, without clipping, execution, or treating them as valid task commands.

## Measured checks

Each row below loads the same pinned pretrained policy into a fresh process and takes one independent optimizer step. This is not a continuous training run.

| Check | Observation | Valid / padded targets | Outcome |
|---|---|---|---|
| Real carry batch | Step 85, 8.5 s, `recorded_transport` | 50 / 0 | Finite loss/gradients; nonzero gradient; checked adapter parameter changed; output shape 1 × 50 × 4 |
| Real episode-end batch | Step 250, 25.0 s, `settle` | 1 / 49 | Same numerical/interface checks passed; chunk stops at episode boundary |
| Synthetic regression | Original six-state/six-action artificial batch | 50 / 0 | Exact match to the historical synthetic report |

Both real checks used batch size one, one camera, rank-four LoRA with 185,664 trainable parameters, mixed pretrained BF16/FP32 parameter dtypes, and BF16 autocast. Peak PyTorch allocated memory was **1,283.82 MiB (1.25 GiB)**; peak reserved memory was **1,364 MiB (1.33 GiB)**. These are per-process PyTorch measurements, not total device use or a promise that larger batches/full fine-tuning will fit.

The carry loss was 1.2759905 and episode-end loss was 2.0663137. These are seeded numerical checks on different input/target chunks, not comparable policy scores. LeRobot's pinned implementation masks padded targets and reduces the loss over valid entries. Neither value measures robot success, and neither is used to select a model checkpoint.

The original synthetic report remains unchanged at `docs/vla-memory.json`. The new report and temporary statistics are saved in `docs/vla-real-batch.json`; raw run folders contain the individual JSON files and the exact script snapshot. No baseline artifact or split assignment was changed.

## Reproduce

First install the isolated environment and recreate the source candidate from the baseline artifacts and external simulator:

```bash
export ROBOT_LEARNING_HW3=/path/to/hw3_imitation_learning
.venv-vla/bin/python scripts/record_visual_episode.py \
  --placement feedback --source IMG_5539 --color green \
  --out work/real-batch-source
```

Then test a carry batch:

```bash
.venv-vla/bin/python scripts/test_vla_memory.py \
  --episode work/real-batch-source --step 85 \
  --out work/my-real-batch-check
```

Expected output includes `status: passed`, `state_size: 13`, `action_size: 4`, `valid_actions: 50`, `padded_actions: 0`, and `predicted_action_shape: [1, 50, 4]`. The output folder contains `report.json` and `normalization.json`, with no changed model weights. Use a new output folder on every run. For the measured episode's final-action check, use `--step 250` and a separate output folder; if your recording length differs, choose its own final action index.

Omit `--episode` and `--step` to retain the original synthetic test. Recorded losses and memory can vary with hardware/software versions; inspect your own report.

## Next

Camera framing and the first balanced collection pilot are now implemented. The [pilot](visual-pilot.md) recorded 12/18 task completions and a consistent third-slot grasp failure; correct that setup before expanding the corpus, while preserving phone-family boundaries. Freeze the accepted corpus and training-only normalization before any task fine-tuning. Keep the single-cube BC/world-model baseline unchanged, and measure VLA success separately once a saved policy can run closed-loop in simulation.

A separate [human teleoperation interface check](vla-human-batch.md) now passes with nine measured state values and four achieved-motion/gripper labels. Its source-aware reader retains the phone checks above; a rerun reproduced this page's original carry-batch report and normalization exactly. The historical measurements on this page remain unchanged.
