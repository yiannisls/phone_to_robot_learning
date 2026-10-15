# Human and phone demonstration compatibility

Compared original curated human training episode 1 against phone-derived training source IMG_5539 (dataset v2). No datasets were merged, no policy was trained, and existing results were not changed.

| Property | Human teleoperation | Phone-derived robot demonstration |
|---|---|---|
| Recorded states | 680 | 150 |
| Duration | 67.9 s | 14.9 s |
| Control interval | 0.1 s | 0.1 s |
| XYZ action label | Difference between measured hand positions | Difference between commanded mocap targets |
| Gripper commands in this example | Approximately 0 to 0.7 rad | -0.174 to 1.0 rad |
| Closing-rate handling | No explicit closing-rate limiter in recorder | Limited to 1 rad/s |
| Native goal XYZ | (-0.20, 0.70, 0.021) m | (-0.23, 0.57, 0.021) m |
| Original obstacles | Enabled | Disabled |
| Origin | Personally collected keyboard teleoperation | Phone-measured transport plus scripted robot manipulation phases |

The human processed XYZ labels were checked against consecutive raw hand positions. Phone XYZ actions were checked against consecutive commanded targets. On this phone episode, the mean norm of commanded-target delta minus measured-hand delta is 0.808 mm per step. This is a label mismatch, not a policy performance measurement.

Both sets of joint/cube poses can be rendered in the same robot scene: maximum reconstructed hand-position error was 0.086 mm for the human recording and 0.084 mm for the phone recording. These checks used the sampled video poses. They do not demonstrate that the commands reproduce the trajectories under physics.

## Common-scene visual replay

The common scene keeps the lab bin at (-0.20, 0.70, 0.001) m and disables both obstacles for visibility. Recorded joint and cube poses are loaded, then mj_forward computes geometry. No physics stepping is performed. The phone cube therefore ends outside the common bin, as expected from its different original goal; this is not a new task failure.

The human raw store does not include full original mocap targets, velocities or camera frames. Loading recorded poses can provide reconstructed visual observations, but is not exact dynamic replay or proof of visual-policy success. Native hand orientation is retained through the saved joints. Camera images generated this way must be identified as reconstruction.

## Consequence for the next experiment

Do not concatenate existing action arrays. A combined-data experiment needs a single task profile, observation definition, action-target convention and gripper executor. Use the lab scene as the target task when comparing against its frozen 97%/82% policy. Retarget phone transport to that task and generate new versioned demonstrations; check obstacle clearance with actual physics before accepting them. Keep original source-family splits. The old phone checkpoints and their 20-layout scores remain historical evidence for their original scene.

A candidate common label convention is future achieved hand poses plus native gripper commands, which can be extracted from both recordings. This requires a new policy/executor experiment; it is not a conversion that preserves either existing policy's behavior. Source balance and differing episode durations also need explicit treatment rather than letting the longer demonstration dominate by row count. These are prerequisites for a later combined-data experiment, not changes made in this audit.

## What the 100% result means

BC and BC + world model both succeeded 20/20 in ordinary phone-project development trials. Under one artificial early-opening request, BC succeeded 9/20 and model-assisted control 20/20. The model-assisted controller has not been evaluated on the human lab's shifted-obstacle distribution. These experiments do not establish that it outperforms the 82% lab policy.

## Reproduce

From the phone project root, using the documented dependencies and an external lab checkout containing the data:

```bash
python scripts/compare_demonstrations.py \
  --lab /absolute/path/to/so101-behavior-cloning \
  --phone data/demonstrations_v2/IMG_5539.npz \
  --out work/data-compatibility
```

Zarr stores are opened read-only. Output must be a new directory. This checks training data only. The original human episode remains in the lab directory and has not been copied into this project's training set. See the adjacent JSON report for measurements and episode array hashes. The raw results and two pose-replay videos are saved locally in `work/data-compatibility-20261015`.
