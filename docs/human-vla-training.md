# First human-only SmolVLA training run

The first small fine-tuning run is complete. **200 optimizer updates** reduced loss on fixed validation windows from **3.3788 to 1.9726**. A saved LoRA adapter reloads in a fresh process and reproduces that result exactly. The subsequent [first simulation rollout](human-vla-rollout.md) failed to pick up the cube.

[Measured report and verification](human-vla-training.json) · [Prepared human dataset](human-vla-dataset.md) · [Training script](../scripts/train_vla.py)

## Read the code in the same order as the lab trainer

The script follows the familiar PyTorch training structure:

1. Hyperparameters at the top: one epoch, batch size one, learning rate 1e-4, and a 200-batch pilot limit.
2. `HumanChunkDataset`: reads a recording, normalizes the state/action values, and pairs the current image and instruction with the next 50 action labels. The last action fills incomplete chunks, and a mask excludes those padded rows from the loss. Two recordings are cached in RAM.
3. `train_one_epoch`: moves a batch to the GPU, computes SmolVLA's loss, calls `backward`, clips the gradient norm at 1 and updates the adapter with AdamW. It logs every sampled episode/step and prints progress every 25 batches.
4. `evaluate`: disables gradients and measures loss on fixed examples. SmolVLA samples noise for its flow-matching objective, so evaluation resets that noise consistently and restores the training RNG afterward.
5. `main`: loads data/model, creates episode-based subsets and DataLoaders, runs training/evaluation, and saves the final adapter and report.

`read_visual_dataset` in the existing converter verifies the corpus checksums, original split and training-only normalization. `load_policy` in the existing interface-check script loads the pinned pretrained model and adds rank-four LoRA. The training script does not duplicate that model setup. The independent adapter-reload verification is kept outside the learning loop.

This is the same overall organization as the lab trainer, with SmolVLA-specific batches, mixed precision and adapter saving. It uses SmolVLA's own flow-matching loss rather than the ACT reconstruction/KL objective.

## Data and training scope

The available corpus contains 39 training episodes with 9,616 action-labelled states and three validation episodes with 784. Validation IDs remain **9, 32 and 36**. Normalization is the frozen training-only corpus normalizer. Inputs are reconstructed RGB images, nine measured robot values and the recorded colour instruction; actions are three achieved hand displacements plus the native gripper command.

The pilot processes **200 distinct training start positions across 38 training episodes**, sampled by a shuffled DataLoader. This is one capped epoch, not a complete pass through all 9,616 examples. Action chunks can overlap, so 200 windows do not represent 10,000 independent actions. Only **185,664 LoRA parameters** are trainable; the pretrained base is frozen. Batch size is one, LoRA rank four/alpha eight, BF16 autocast is enabled, and the seed is 42. The final adapter is saved; there is no best-validation checkpoint selection or hyperparameter sweep.

The frozen flow-matching loss probes contain 24 training windows and 24 validation windows. Validation samples eight evenly spaced starts from each of the three episodes, including the final action. Means are weighted by the number of valid action rows, so short final chunks are not treated as full chunks. The same examples and noise seeds are used before and after training. This is **not the loss over all 784 validation starts**, and it is not an action error in metres or a task-success percentage.

| Fixed-window loss | Before training | After 200 updates |
|---|---:|---:|
| Training: 24 windows | 2.6393 | 1.7006 |
| Validation: 24 windows across 3 episodes | 3.3788 | 1.9726 |

The mean loss observed while optimizing was 2.3657; it measures a changing model on different sampled training windows and should not be confused with the fixed-window comparison above. Peak PyTorch GPU allocation was **1,293.4 MiB**, with **1,366 MiB reserved**, on the RTX 4060 Laptop GPU. These allocator measurements exclude other applications.

## Where the phone recordings fit

**This run contains human teleoperation only.** No phone-derived examples, DAgger segments, scripted placement corrections or changed-torque replays were added. Phone recordings remain the source of the completed state-BC/world-model baseline.

The later VLA comparison should keep this human-only reference, align the phone-derived robot observations/actions, then compare human-only and combined training under the same evaluation conditions. The phone data would contribute the recorded transport shape with scripted robot grasp/release phases labelled explicitly. Direct training on original phone-camera images has not been implemented, and human achieved-motion labels must not be silently mixed with target-command labels.

## Run the pilot

Use the prepared corpus and isolated VLA environment from the project root:

```bash
.venv-vla/bin/python scripts/train_vla.py \
  --dataset work/human-vla-dataset-20261015-01 \
  --epochs 1 --max-train-batches 200 \
  --out work/my-human-vla-pilot
```

Use a new output folder. Expect the training-example count, 24 validation windows, 185,664 trainable parameters, initial losses, progress every 25 batches, and a final adapter path. The measured run used cached pinned model files with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; those variables are optional when downloading the pinned files initially. The raw lab checkout is not needed for training once this prepared corpus is available.

`--max-train-batches 0` allows a complete training epoch; `--epochs` controls the number of passes. Those longer runs have not been performed or scored here. Changing them is a new experiment, not a reproduction of this pilot.

The output contains `adapter/adapter_model.safetensors`, adapter configuration, portable policy configuration, `normalization.json`, `loss_probes.json`, per-epoch sample logs, `history.json` and `report.json`. The **753,728-byte adapter** requires the separately loaded pinned base model; it is not a standalone 450-million-parameter model. Optimizer state and training-resume support are not provided by this first script.

## Checks and limitations

The new DataLoader batches match the previous interface for state, language, actions and masks. CPU image scaling differs from the earlier GPU scaling by at most 5.96e-8, within the declared 1e-7 tolerance. Normal, final-action and next-episode boundaries were checked. The loader rejects changed split assignments and paths outside the dataset. Logged training IDs contain no validation episodes; all 200 sampled starts are distinct.

A fresh process loaded the pinned base and restored all adapter tensors exactly. Seventy-four adapter tensors differed from initialization, and both fixed-window losses reproduced exactly. The restored model produced finite `(1, 50, 4)` action predictions, converted back to physical label units. Those verification predictions were not executed; the separate [development rollout](human-vla-rollout.md) subsequently tested the saved adapter through physics. The 1,418 immutable baseline artifacts remain unchanged.

The small measured report is tracked in Git. The adapter, logs, exact script snapshots and independent verification are retained locally at `work/human-vla-pilot-20261015-01`; they are not part of the older public baseline archive. Public packaging is still pending.

Reconstructed images, achieved-motion action labels, missing original command/velocity records and unvalidated contact physics remain limitations. Reduced loss supports continuing the experiment; it does not establish colour grounding or successful grasp/placement. The subsequent first rollout failed to pick up the cube. A longer predeclared training experiment is the next candidate; a held-out VLA success-rate benchmark remains pending.

Presentation explanation: “I fine-tuned a small set of SmolVLA adapter weights using my human demonstrations. On fixed held-out examples the learning loss decreased, and the saved adapter reloads correctly. The first simulation test did not pick up the cube, so lower loss has not yet translated into successful control.”

Follow-up: the [2,000-update experiment](human-vla-2000-updates.md) lowered fixed validation loss to 1.2959 but still failed to pick up the cube on the same development start. The original pilot remains preserved.
