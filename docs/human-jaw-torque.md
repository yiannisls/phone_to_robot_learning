# Does a lower jaw torque limit fix the contacts?

**Not in this training case.** The 1 Nm replay loses the cube; 2 Nm completes the transfer but produces a larger pickup contact peak. Keep the existing 3.5 Nm default while investigating pickup alignment and contact geometry. None of these candidates is accepted into a VLA training corpus.

[Measured reports and trace checks](human-jaw-torque.json) · [Previous lower-first experiment](human-lower-first.md)

## One episode, three settings

These are repeated diagnostic replays of original training episode 0, not three independent demonstrations or a learned-policy success rate. All use the same recorded action prefix, initial configuration, fixed hand orientation, explicit handoff at action 322, and one instant opening request after scripted lowering. Changing torque changes the physical trajectory, so the resulting states and feedback placement durations need not match.

| Jaw motor limit | Transfer completed | Maximum target penetration | Maximum single target-contact force | Duration |
|---|---|---|---|---|
| Default ±3.5 Nm | Yes | 1.005 mm | 97.92 N | 46.0 s |
| ±1.0 Nm | No | 1.141 mm | 141.19 N | 32.2 s |
| ±2.0 Nm | Yes | 1.028 mm | 220.82 N | 45.7 s |

All three pass the existing depth/self-contact/distractor-motion screen. The 1 Nm case demonstrates why that screen must be considered separately from task completion: the red cube slips during transport and rests on the blue cube. At the end its only logged target contact is with `blue_box`. The placement readiness check rejects the handoff and no release is attempted.

The 2 Nm setting was tried after the 1 Nm failure as one intermediate development setting. It is not a held-out or predeclared benchmark result. Both completed teacher runs exceed the proposed 40-second common policy-evaluation allowance. All attempts are retained.

## What the new trace tells us

The jaw actuator drives a hinge with unit gear ratio; the script checks that configuration before interpreting its output as Nm. With `--contact-trace`, `jaw_trace.npz` records the requested jaw angle, pre-step measured angle and speed, actuator torque, and joint actuator torque at every 2 ms physics step. The row timestamp is the time after that step. The existing contact trace uses the same timestamps. These are simulator measurements, not real gripper force measurements.

In the default replay, the approximately 98 N contact peak occurs against the fixed jaw pad during release, while the jaw motor torque is only **0.026 Nm**. Contact force therefore cannot be read directly from the moving jaw motor limit. The arm is also constrained to a hand target, and the cube can contact the fixed jaw and floor. The precise contribution of these constraints has not been isolated.

The default release has a step-maximum contact force above 50 N for **16 ms**. That threshold describes the event duration; it is not an acceptance limit. During scripted lowering, median jaw torque magnitude is **3.16 Nm**, with a median step-maximum contact force of **28.81 N**. This is not only a brief release spike.

At 2 Nm, the release peak falls to **42.81 N**, but pickup reaches **220.82 N** at 16.470 s against `fixed_jaw_pad_1`. At 1 Nm, pickup reaches **141.19 N** at 16.882 s against the same pad, and the cube is later lost. Lower torque changes jaw tracking and contact geometry; these observations do not isolate the cause of each peak or establish a physically validated setting.

## Reproduce

From the project root, use a fresh output directory for each run:

```bash
MUJOCO_GL=egl .venv-vla/bin/python scripts/replay_human_episode.py \
  --lab /path/to/so101-behavior-cloning \
  --episode 0 --placement lower-first --handoff-step 322 \
  --contact-trace --out work/my-jaw-default
```

Repeat with `--jaw-torque-limit 1 --out work/my-jaw-1nm` or `--jaw-torque-limit 2 --out work/my-jaw-2nm`, replacing the original `--out` argument. Expect the outcomes in the table under the recorded MuJoCo 3.14.0 configuration. Inspect the report on another platform/version.

The option changes the in-memory jaw actuator force range for that run. It does not edit the external simulator XML or change contact parameters, arm control, instant opening, or evaluation thresholds. Omitting it preserves the default. Zero, negative, infinite and NaN limits are rejected before loading the data.

## Verification and boundaries

The user's `work/my-lower-first-replay` report and arrays exactly match the retained corrected run. Adding passive jaw logging also reproduces its rollout, physics and contact arrays exactly. Each new trace is finite, aligned with physics samples and commands, and respects its configured motor limit; actuator and joint actuator torques match for the checked unit-gear transmission.

Source-array fingerprints agree across all three runs. All **1,418 immutable baseline artifacts** pass checksum verification. Original datasets, held-out families, normalization, checkpoints and historical results are unchanged. The default configuration remains unchanged; no model was trained or evaluated here.

New runs and script snapshots are in `work/human-jaw-torque-20261015-01/{default,limit-1nm,limit-2nm}`. The small measured report is tracked in Git; the larger traces and videos remain local and are not included in the older baseline release archive.

The next useful diagnosis is pickup alignment and contact geometry, followed by checks on more training episodes. A torque reduction alone is not supported as the fix.

Presentation explanation: “I tested whether reducing the gripper motor torque fixed the large contact forces. One setting lost the cube, and another increased the pickup peak. I retained these failures and kept the baseline unchanged rather than claiming the force problem was solved.”
