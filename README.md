# Phone recordings to robot learning

Start with [the project map](docs/project-map.md) and [the runnable walkthrough](docs/walkthrough.md). Clean-environment validation passed; see [verification evidence](docs/reproduction.md). The checkpoint/data archive is prepared locally; GitHub publication is pending authentication.

This project uses five personally recorded manipulation videos to teach a
simulated SO-100 arm to move a cube into a bin. An ArUco marker on the object
provides the recorded transport motion. Approach, grasp, placement, release and
retreat are scripted when generating demonstrations. A neural network then
learns to imitate the resulting robot state/action sequences.

The current policy is **state-based behavioral cloning**, not a VLA. It does not
take camera images or language as input. A small action-conditioned state world model is now trained
on dataset v2; it now also compares gripper choices inside the control loop.

## World model in the control loop

BC v2 supplies movement while the learned world model compares gripper choices
half a second ahead. On 20 paired positions, both controllers complete 20/20
ordinary tasks. With one deliberately early opening request, BC completes 9/20;
BC plus the world model completes 20/20 and rejects every pulse.

The revised selector makes no ordinary interventions and exactly restores the
unperturbed BC trajectories under this specific artificial fault. Grip forces
remain high and unchanged. A simple rule could also address these pulses;
superiority over such a rule has not been established. The revised results are
a development retest on the same positions, following a post-release gate fix.

See [methods, failures and checks](docs/world_model_control.md), the
[comparison video](results/world_model_control_comparison.mp4), and
[aggregate measurements](results/world_model_control.json). To try the revised
selector with your homework environment active:

```bash
python scripts/evaluate_model_control.py --mode wm --video --out work/wm-validation
```

## Latest policy and recording experiment

BC v2, trained on the preserved phone-derived dataset at 1 rad/s, completes
20/20 original development positions and 19/20 wider positions. BC v1 completes
18/20 of those same wider positions. Force peaks remain a limitation: BC v2 has
larger peaks on the original suite, so both checkpoints are retained.

Thirteen agent-operated recordings are saved as NPZ and lab-compatible Zarr under
`data/teleoperation/agent_v1/`. They contain 2,025 actions and 101,250 force samples,
remain unassigned to splits, and have not entered training. They are separate
from future human keyboard recordings. See [the experiment and reproduction
commands](docs/agent_gripper_policy.md) for outcomes, checks and remaining failures.

## Original BC v1 results

| Check | Result |
| --- | --- |
| Scripted dataset generation | 65/65 task and contact-depth passes |
| Training data | 63 episodes, 9,532 state/action transitions |
| Validation data | 1 episode, 149 state/action transitions |
| Test data | 1 episode; excluded from policy training and checkpoint selection |
| Selected checkpoint | Epoch 100 of 300 |
| Validation imitation position error | 1.17 mm mean absolute error across predicted targets |
| Validation imitation gripper accuracy | 99.3% across predicted commands |
| Learned-policy validation rollout | Success, 166 steps at 10 Hz |
| New starting-position development evaluation | 20/20 task and contact-depth passes; seed 20261007 |
| Development evaluation maximum cube penetration | 1.492 mm |
| Development evaluation worst jaw force peak | 889.5 N |
| Validation rollout maximum cube penetration | 0.644 mm |
| Validation rollout peak normal forces | Fixed jaw 777 N; moving jaw 740 N |

The 20 development layouts use new starting positions inside the training
augmentation range. They vary cube position only, and do not establish performance
with different objects, friction, cameras or obstacles. The held-out recording
has not yet been evaluated by the learned policy. Contact-depth checks do not
establish realistic gripper forces; large force peaks remain a limitation.
Earlier project results are not results of this rebuild.

## Run with the existing homework environment

The simulator currently depends on an independent HW3 checkout containing
`hw3/sim_env.py` and `so101_gym/assets/so100_transfer_cube_obstacle_ee.xml`.
This rebuild is not yet self-contained. The measured run used MuJoCo 3.14.0,
PyTorch 2.14.1+cu130, NumPy 2.5.3 and OpenCV 5.0.0; training ran on CPU.

Set these variables to your existing homework checkout and Python environment:

```bash
export ROBOT_LEARNING_HW3=/path/to/hw3_imitation_learning
export ROBOT_PYTHON="$ROBOT_LEARNING_HW3/.venv/bin/python"
```

Place `IMG_5539.mp4` through `IMG_5543.mp4` in `data/raw/`, then run from
this project's root directory:

```bash
"$ROBOT_PYTHON" scripts/prepare_recordings.py
"$ROBOT_PYTHON" scripts/generate_dataset.py
"$ROBOT_PYTHON" scripts/train_policy.py --epochs 300
"$ROBOT_PYTHON" scripts/evaluate_policy.py --video
```

Preparation uses the manually supplied times in `prepare_recordings.py`; it does
not detect movement boundaries automatically. It skips already prepared transport
curves. Generation resumes missing episodes, but refuses to replace an existing
dataset report. Training and evaluation refuse to overwrite their output folders.

To repeat evaluation with an existing checkpoint, choose a new output folder:

```bash
"$ROBOT_PYTHON" scripts/evaluate_policy.py --video --out results/my-validation-run
```

To repeat training, select a new checkpoint folder with `--out`, and pass its
`policy.pt` to evaluation with `--checkpoint`.

Run the saved-seed development evaluation in a new output folder:

```bash
"$ROBOT_PYTHON" scripts/evaluate_layouts.py --layouts 20 --seed 20261007 --video --out results/my-layout-evaluation
```

If interrupted, repeat that command with `--resume`. Completed cases are reused
only when their positions, checkpoint and evaluation code match. The positions
are saved in `layouts.json` before execution; results are in `summary.json`.

## Follow the code

- `prepare_recordings.py`: run marker tracking, time selection, pose estimation
  and transport conversion for all five videos.
- `generate_demo.py`: execute one adapted path and save robot states/actions.
- `generate_dataset.py`: generate base episodes and 20 starting-position
  variations for each of the three training sources.
- `policy.py`: define the 28 measured-state features and the neural network.
- `train_policy.py`: learn eight future target offsets and gripper decisions;
  select a checkpoint using validation imitation loss.
- `evaluate_policy.py`: execute four learned targets, measure again, and repeat.
  The controller limits each target movement to 12 mm and gripper closing to
  2 rad/s. It records the rollout, video and physics contact diagnostics.

- `evaluate_layouts.py`: run the fixed checkpoint on reproducible starting
  positions and collect task, contact-depth and force measurements.

Source families stay separate: IMG_5539/5540/5541 are training, IMG_5542 is
validation, and IMG_5543 is test. Normalization uses training data only. The policy
does not receive trajectory time, phase labels or future measurements as inputs.

Design assumptions and historical context are recorded in `docs/design_notes.md`.
Dataset provenance is in `results/dataset_v1.json`; checkpoint provenance is in
`checkpoints/bc_v1/training_summary.json`. The learned validation output is in
`results/bc_v1_validation/`. Code was developed with AI assistance in a guided
rebuild; it should not be presented as entirely unaided work.

The development evaluation is saved in `results/bc_v1_layouts/`. See
`docs/baseline_results.md` for its scope and the gripper-force diagnosis.

## Gripper experiments

Closing at 1.0 rad/s retained 20/20 successes on the same development layouts
and reduced the worst force peak from 889.5 N to 499.3 N (43.9%). Closing at
0.5 rad/s achieved 17/20; lowering the actuator torque also caused failures
with this checkpoint. Full results, including failed trials, are documented in
[gripper_experiments.md](docs/gripper_experiments.md).

Use `--gripper-close-rate 1.0` with either evaluation script to reproduce the
selected candidate. Defaults and the existing dataset remain at 2.0 rad/s for
baseline reproduction. Dataset v2 now uses the selected rate, and the first world model
has been trained and evaluated on that execution profile. High sustained grip forces remain a limitation.

## World model

Dataset v2 contains 65 new demonstrations at 1.0 rad/s, with the same source
split. An 8,540-parameter network learns physical state changes from current
state and action. Its checkpoint was selected at epoch 108 using validation.
On the test source, cube coordinate MAE is 0.160 mm at 0.1 seconds, 0.908 mm
at 0.5 seconds and 2.456 mm at 1 second. The unchanged-state baseline gives
0.806, 4.125 and 8.449 mm respectively. These preserve the first model's prediction results; the separate gripper-control
experiment is documented in [world_model_control.md](docs/world_model_control.md).

```bash
"$ROBOT_PYTHON" scripts/generate_dataset.py --version v2 --gripper-close-rate 1.0
"$ROBOT_PYTHON" scripts/train_world_model.py --epochs 150
"$ROBOT_PYTHON" scripts/evaluate_world_model.py --split validation
"$ROBOT_PYTHON" scripts/evaluate_world_model.py --split test
```

Use new output folders when repeating training or evaluation. Full methods,
plots and limitations are in [world_model_results.md](docs/world_model_results.md).
Alternative-action validation is now complete: the first model struggles with
small movement changes and unexpected gripper requests. At six selected validation
moments, 0.5-second cube MAE is 1.441 mm for recorded commands and 10.323 mm
when the first gripper request is flipped. Its predicted effect of changing an
action is worse than ignoring that change at this horizon. See
[world_model_action_probes.md](docs/world_model_action_probes.md) for the experiment
and its limitations. This motivated the second-model experiment below.

```bash
"$ROBOT_PYTHON" scripts/probe_world_model.py
```

## World model after action variations

A second checkpoint uses 18,900 additional simulator transitions from the three
training source families. In the same validation action probes, 0.5-second cube
MAE for a gripper change fell from 10.323 mm to 1.147 mm. Ordinary recorded-motion
MAE increased from 0.856 mm to 1.153 mm at this horizon. Predicting small position
changes still needs work. Gripper-control results are now documented separately
in [world_model_control.md](docs/world_model_control.md).

See [world_model_v2_results.md](docs/world_model_v2_results.md) for reproduction,
comparison plots and limitations. The subsequent control experiment is described
in [world_model_control.md](docs/world_model_control.md). The first model,
its results and both original demonstration datasets are preserved. The second
model has not been evaluated on the test source.

## Additional recordings and two-marker tracking

Ten additional phone recordings are imported under `data/raw/new_batch_20261007/`.
Four passed the sampled-marker review; six are retained for further review. The
four accepted clips now have separate object and table tracks in all 311 sampled
frames, despite both markers decoding as ID 0. Initial table positions were
manually identified, and tracking rejects ambiguous assignments.

See [two_marker_tracking.md](docs/two_marker_tracking.md) for the command, results
and limitations, and [new_recordings_review.md](docs/new_recordings_review.md)
for the import decisions. These clips remain unassigned to splits and have not
entered training. Relative pixel tracks need further processing before they can
provide simulated robot demonstrations.

## Gripper teleoperation recorder

A keyboard recorder now saves 10 Hz robot states/actions and 500 Hz jaw contact
loads, motor torque and object motion. With the homework environment active:

```bash
python scripts/record_gripper.py --out data/teleoperation/grasp-001 --zarr --video
```

The window starts paused; click it and press Space. Escape saves and exits.
Controls, file formats and replay instructions are in
[gripper_recording.md](docs/gripper_recording.md). New recordings keep an unassigned
split and are not automatically added to training. The recorder was checked using
scripted training-demonstration actions and an exact replay; manual grasps still
need to be collected. Large simulated force peaks remain a limitation.
