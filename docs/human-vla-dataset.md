# The human dataset for the first VLA experiment

The original **42 human teleoperation episodes** are now converted into synchronized reconstructed images, measured robot states, instructions and action labels. The existing split is preserved: **39 training episodes and 3 validation episodes**. This stage prepares the data; it does not train a policy or establish a success rate.

[Measured verification, manifest and normalization](human-vla-dataset.json) · [How one recording is converted](human-visual-episode.md) · [SmolVLA interface check](vla-human-batch.md)

## What the model will learn from

At time `t`, the model receives a 320 × 240 RGB image from the original `angle` camera, six measured robot joint angles, three measured hand coordinates, and an instruction such as “Put the green cube in the bin.” Its four action targets are the next measured hand displacement in XYZ and the recorded gripper command at `t`.

Images are rendered from the saved robot, cube and bin poses. They are **reconstructed simulator images**, not camera frames saved during teleoperation or phone footage. Cube/bin coordinates and one-hot colour labels are retained for provenance but are not part of the nine-value robot input. Each episode retains its final observation; it has no following action.

| Split | Episodes | Observations/images | Action examples | Red / green / blue episodes |
|---|---:|---:|---:|---|
| Training | 39 | 9,655 | 9,616 | 10 / 14 / 15 |
| Validation | 3 | 787 | 784 | 1 / 1 / 1 |
| Total | 42 | 10,442 | 10,400 | 11 / 15 / 16 |

Validation global episode IDs remain **9, 32 and 36** (zero-based). Source ordering follows the original processed lab manifest, not arbitrary filesystem order. The seven DAgger segments remain separate. No new test split is invented, and the phone train/validation/test families are unchanged. Three validation recordings provide a small development check; they are not a broad generalization benchmark.

## The decision to proceed

We are proceeding with an **experimental supervised fine-tuning corpus**, while retaining the documented contact and action-reconstruction limitations. Training entries are marked `accepted_for_training: true` with that explicit acceptance basis. Validation entries remain false. This is an offline-data decision, not a claim that these episodes pass a physical force or penetration screen.

The original human recordings supply this corpus. The scripted lower-first correction, changed-torque replays, phone-generated visual pilots and seven DAgger segments are not mixed into it. Their historical reports remain unchanged.

The [physics replay](human-physics-replay.md) and [torque comparison](human-jaw-torque.md) showed why later closed-loop evaluation is necessary. Measured hand displacements are not the missing original hand-target commands, original timestamps and velocities were not saved, and four-value actions do not encode intentional wrist rotation. Reconstructed poses alone do not validate contacts. A small fine-tuning run can test learnability of these labels; only simulation rollouts can measure control performance.

## Recreate the dataset

From the project root, using the isolated VLA environment and the external lab checkout with its original raw/processed stores and simulator assets:

```bash
MUJOCO_GL=egl .venv-vla/bin/python scripts/convert_human_episode.py \
  --lab /path/to/so101-behavior-cloning \
  --all --out work/my-human-vla-dataset
```

Use a fresh output folder. Expect progress through `Converted 42/42`, followed by the manifest. The measured dataset is `work/human-vla-dataset-20261015-01` and occupies approximately **94 MiB** including previews/videos.

Each `episode_000` through `episode_041` folder contains `episode.npz`, `summary.json`, `preview.jpg` and `episode.avi`. The dataset root contains:

- `manifest.json`: episode IDs, original splits, relative folders, counts, instructions and checksums.
- `normalization.json`: state/action means and standard deviations fitted on the 9,616 training transitions. Terminal states and all validation rows are excluded; minimum standard deviation is 1e-6.

The manifest is written only after every conversion succeeds. A partially created folder without the final manifest is not a complete dataset. Existing output folders are protected; this command does not resume or overwrite them. `--all` and `--episode` are mutually exclusive. Omitting both still converts episode 0. These files are an intermediate NPZ corpus compatible with our batch-building code, not a native LeRobot dataset directory.

## Verification

All 42 source episodes were independently re-read and their saved arrays/action labels checked. Output checksums, source-array fingerprints, dimensions, finite values, nominal 10 Hz timing and original split assignments pass. State/action normalization was independently recalculated from training rows and matches exactly. Maximum reconstructed hand-position difference across the corpus is **0.0974 mm**; maximum training XYZ label length is **10.030 mm**.

Episode 0 matches the earlier converted example exactly, including RGB images. Training examples for all three colours were visually reviewed. All **1,418 immutable baseline artifacts** still pass checksum verification. No existing dataset, normalizer or checkpoint was replaced.

The measured conversion retains its exact script snapshot and verification report locally. A subsequent CLI-only fix rejects an explicit `--episode 0` combined with `--all`; the conversion calculations are unchanged. The small report records both source hashes. The 94 MiB dataset and external lab data are not included in the older baseline release archive; a public dataset release is still pending.

## Next runnable stage

Add a small SmolVLA fine-tuning entry point using the already tested rank-four LoRA configuration. It should read this manifest, use the saved training-only normalization, keep action chunks inside episode boundaries, mask padded actions, and save the adapter plus configuration and data fingerprints. Report training and validation loss separately. A loss reduction must not be presented as robot success; closed-loop evaluation follows.

Presentation explanation: “I reconstructed images from my 42 recorded robot demonstrations and paired each image with the robot state, colour instruction and following action. I kept three recordings for validation and fitted scaling only on the other 39. The first VLA experiment will test learning from those labels, with the replay limitations documented.”
