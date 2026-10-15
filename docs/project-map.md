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

## Supporting experiments

Keep these for investigation rather than the first walkthrough: `inspect_videos.py`, `plot_marker.py`, `track_two_markers.py`, `evaluate_layouts.py`, `test_gripper_settings.py`, `probe_world_model.py`, `evaluate_world_model.py`, `collect_agent_grasps.py`, `replay_gripper.py`, and the older `evaluate_policy.py`. Historical reports and outputs retain their original names and provenance. No files need moving to understand the main workflow.

## Combining sources

See [the single-episode compatibility audit](data-compatibility.md) before mixing phone-derived demonstrations with human teleoperation. Their action labels, bin positions and execution profiles differ. `compare_demonstrations.py` checks one training episode from each and renders recorded poses without physics stepping.
