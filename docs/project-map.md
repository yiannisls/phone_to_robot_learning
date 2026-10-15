# Project map

Read this project in the order below. A demonstration is a sequence of robot states and actions; a policy learns to imitate those actions. A world model learns the resulting state changes.

| Stage | Scripts | Input → output |
|---|---|---|
| Extract phone motion | `track_marker.py`, `select_movement.py`, `estimate_pose.py`, `make_transport.py` | Personal MP4 → marker corners → selected interval → estimated pose → transport CSV |
| Prepare five recordings | `prepare_recordings.py` | Manually selected time intervals → five transport curves |
| Generate robot demonstrations | `generate_demo.py`, `generate_dataset.py` | Transport curves plus scripted approach/grasp/place/release → versioned NPZ episodes |
| Learn BC | `policy.py`, `train_policy.py` | Training episodes → eight-action predictions and saved weights |
| Learn dynamics | `world_model.py`, `collect_action_variations.py`, `train_world_model.py` | State/action/next-state examples → learned transition model |
| Evaluate control | `gripper_planner.py`, `evaluate_model_control.py`, `compare_model_control.py` | Frozen models → saved rollouts, decisions, metrics |

The main entry point is `evaluate_model_control.py`. `record_gripper.py` supplies its simulator construction and force logger; `generate_demo.py` supplies observations. These are runtime dependencies even when no recording or dataset generation is requested.

## What each model does

BC receives 28 measured state features and predicts eight future movement targets and gripper choices. Four movement commands execute before replanning. The state world model receives 35 state features plus four action values and predicts 28 physical state changes; known command bookkeeping is updated analytically. It has no images, language, or force-prediction output.

The gripper selector compares the usual BC request, opening, and closing over five steps (0.5 seconds). A hand-written task score selects an alternative only if its predicted improvement exceeds the margin. BC still supplies movement. See `world_model_control.md` for the activation rules and limitations.

## Data boundaries

IMG_5539, IMG_5540 and IMG_5541 are training families; IMG_5542 is validation; IMG_5543 is test. All augmented descendants retain their source family. There are 63 training episodes, one validation episode and one test episode in each original dataset version. Training normalization uses training examples only. Additional phone clips and automatically operated recordings have not entered these models.

Phone recordings supply the transport curve. Calibration is approximate; start/end intervals are manually chosen. Approach, grasp, placement, release and retreat are scripted during demonstration generation. They are not measured robot actions from the phone.

## Which files are needed?

To run the saved BC or BC + world-model controller, keep these six files together in `scripts/`:

- `evaluate_model_control.py`: entry point for one rollout.
- `policy.py`: behavioral-cloning network and observations.
- `gripper_planner.py`: compares predicted gripper outcomes.
- `world_model.py`: transition network and prediction helpers.
- `record_gripper.py`: simulator setup, execution and contact logger.
- `generate_demo.py`: observation helper imported by the logger.

`compare_model_control.py` runs the paired multi-layout comparison. Keep the extraction, generation and training scripts in the table above to rebuild the pipeline. `collect_action_variations.py` imports helpers from `probe_world_model.py`, so the probe remains a training-data dependency.

Other retained tools have specific roles: `evaluate_world_model.py` measures forecast error; `track_two_markers.py` processes the additional recordings; `collect_agent_grasps.py` and `replay_gripper.py` reproduce the separate gripper-recording experiment; `compare_demonstrations.py` checks human/phone data compatibility. `check_setup.py` and `release_artifacts.py` support setup and release verification.

## Removed after the experiments

The cleanup removed nine standalone scripts: `plot_marker.py`, `plot_release_comparison.py`, `make_readme_figures.py`, `inspect_videos.py`, `diagnose_lab_policy.py`, `summarize_lab_failures.py`, `evaluate_policy.py`, `evaluate_layouts.py`, and `test_gripper_settings.py`. The first six supported completed visualization/diagnostic work; the final three form the historical evaluation path superseded for current control runs. None is imported or launched by the retained scripts. Figures, reports, data and checkpoints remain.

The code is recoverable from commit `e686ed4`. For example, inspect a removed plotting script without changing the current checkout:

```bash
git show e686ed4:scripts/make_readme_figures.py
```

Historical reproduction commands refer to that earlier source tree. Current evaluation uses `evaluate_model_control.py` and `compare_model_control.py`; the historical controllers are not claimed to be identical.

## Combining sources

See [the single-episode compatibility audit](data-compatibility.md) before mixing phone-derived demonstrations with human teleoperation. Their action labels, bin positions and execution profiles differ. `compare_demonstrations.py` checks one training episode from each and renders recorded poses without physics stepping.

## Cleanup verification

After the nine removals, BC validation completed in 131 steps with task and contact-depth passes. Model-assisted validation under the artificial opening request also completed in 131 steps, with exactly one override. All 1,418 preserved artifact checksums passed. These checks use the existing MuJoCo 3.14 profile and validation source; they are not new generalization results.

## Current integration pilot

`test_phone_in_lab.py` tests generated demonstrations in the lab scene. `--approach human` uses a recorded human training prefix followed by phone-derived transport and scripted placement. It performs no learning and keeps all outputs excluded from training. See [pilot results](phone-lab-pilot.md).

## Separate VLA experiment

`test_vla_memory.py` checks whether pretrained SmolVLA can complete one LoRA training step on the local GPU. Inputs are synthetic: one RGB image, a robot-state vector, a tokenized instruction and action targets. Its output is a JSON report of memory use and numerical checks. It downloads pinned pretrained weights, requires `.venv-vla`, and does not train or evaluate the phone BC/world-model controllers. See [setup and measured results](vla-setup.md).

`record_visual_episode.py` replays one successful phone-derived training demonstration in a fixed multicube arrangement and records aligned RGB images, robot inputs, actions, diagnostics and an instruction. It checks timing, command alignment, contact depth and distractor movement. It saves candidate data under a new output folder; it does not add episodes to a training dataset. See [the inspected episode and its placement failure](visual-episode.md).

The visual recorder also supports `--placement feedback`: a separately labelled teacher takes over near the bin, centres and lowers the cube, then releases gradually. The source prefix is checked and its retained transport count is reported. See [the corrected candidate and all development attempts](visual-placement.md). Original `--placement replay` behavior remains the default.

`test_vla_memory.py --episode <candidate-folder> --step <index>` also checks a real image/state/instruction/action batch, including temporary training-only normalization, episode-end padding and predicted action shape. It refuses held-out families and contact-failing candidates. No model predictions are executed and changed weights are not saved. See [the real-batch checks](vla-real-batch.md).

The visual recorder supports `--framing workspace` for a fixed, closer camera view. Default `--framing scene` preserves the simulator camera. The report stores the actual camera pose and field of view. See [the measured framing comparison](camera-framing.md) and [the next collection specification](visual-collection-plan.md); balanced target-slot recording is now implemented, with the first pilot reporting 12/18 task completions and no training inclusions.

`collect_visual_pilot.py` records the predeclared six colour arrangements × three target instructions, retains all attempts/logs and checks initial-scene equality across instructions. `record_visual_episode.py --slot-colors ...` retargets scripted pickup and the processed phone transport, inserts commands to respect the 12 mm movement limit, and reports camera coverage. See [the pilot outcomes and failed grasp](visual-pilot.md). BC/WM and VLA future evaluation must follow [the comparison boundaries](controller-comparison.md); a multicube world-model adapter is not implemented.

`--layout front` explicitly moves the third pickup slot 9 cm toward the front. Both visual scripts default to `--layout original`. Reports now include physics-rate robot self-penetration; review eligibility requires none detected. The second collection completed 18/18 scripted transfers in this smaller workspace; the original third-slot failure remains. See [workspace choice, measurements and reproduction](visual-workspace.md). No task-trained VLA exists yet.

`convert_human_episode.py` reads one original multicube human episode from an explicitly configured lab checkout, preserves its existing split, reconstructs images without physics stepping and saves a nine-value measured robot input with four-value achieved-motion/gripper labels. It verifies labels against the original processed store and leaves all candidates outside training. DAgger conversion and mixing with phone commands are not implemented. The [source-aware interface check](vla-human-batch.md) now handles this human example with nine state values while retaining thirteen for phone examples; both the final-action mask and exact phone-report regression passed. See [the complete human episode conversion](human-visual-episode.md).
