# Additional phone-only controller evaluation

## Protocol

After stopping teleoperation integration, the active experiment returns to the original phone-derived BC v2 and state world model v2. Neither model, controller settings nor simulator execution profile was changed for this comparison. No human or automatically operated teleoperation data entered training.

One hundred starting-position offsets were sampled with seed 20261015, X in [-40, 40] mm and Y in [-30, 40] mm. The full positions list and control/source hashes were saved before the remaining 320 runs; the first twenty layouts had already been evaluated as described below. This is the same range as the earlier wider comparison, with the first twenty newly sampled positions and eighty further positions. It does not introduce different obstacles, objects, camera views, friction or masses.

Each position is used for BC and BC + world model under ordinary execution and under a single artificial early-opening request: 400 runs total. The fault is triggered only after a cube is lifted above 70 mm outside the bin, with both BC and the applied gripper request closed. Reports must therefore distinguish fault-trigger counts from all requested stress runs: a controller that never lifts may fail before any fault is injected.

The baseline environment uses MuJoCo 3.14.0, 10 Hz control, 500 Hz physics, a 1 rad/s closing rate and +/-3.5 N m jaw actuator torque limit. The training source-family split and held-out recording remain unchanged. This evaluates new simulator layouts, not the held-out phone recording. No weights were retrained and no parameters were selected from these results.

## Expansion and scheduling

The first twenty positions were evaluated before extending the suite to 100. All 80 earlier controller/condition runs are reused after checking layout positions, model identities, execution-code hashes and settings. They are counted once. The remaining 320 runs use the next eighty positions from the same seeded generator. No case was selected or removed according to outcome. The original twenty-layout results are retained independently.

The existing runner now supports `--workers`; four independent subprocesses were used. Before scaling up, eight parallel runs exactly reproduced their earlier states, actions and physics traces. Model and execution code are unchanged; only scheduling and decimal offset formatting changed. CPU timing measurements under parallel load should not be treated as single-controller latency benchmarks.

## Reproduce

With the documented environment active and ROBOT_LEARNING_HW3 set, run from the repository root using a new output directory:

```bash
python scripts/compare_model_control.py \
  --layouts 100 --seed 20261015 --workers 4 --out work/phone-only-confirmation
```

The existing runner saves the sampled positions, checkpoint/code hashes, per-episode states and actions, contact measurements, decisions and aggregate metrics. Raw outputs from this run are retained locally in `work/phone-only-100-layouts-20261015`. See the adjacent JSON report for the frozen protocol, compact episode outcomes and summary. Historical results are preserved separately.

## Interpretation

The purpose is to check whether the earlier phone-only result repeats on additional layouts. A perfect score on a finite layout suite is not a guarantee of general success. The ordinary comparison measures natural task performance; the separate artificial opening experiment measures protection from one specified disturbance. The selector does not change the initial grasp or plan arm motion. Depth checks do not establish realistic gripping forces.

## Results

| Condition, 100 positions | BC tasks | BC task + depth pass | BC + WM tasks | BC + WM task + depth pass |
|---|---:|---:|---:|---:|
| Ordinary | 99/100 | 99/100 | 99/100 | 99/100 |
| Artificial early opening | 49/100 | 0/100 | 99/100 | 99/100 |

The one natural failure is layout_25, offset (-39.10, +38.20) mm, outside the training augmentation range on both axes. The cube remains on the table and both controllers time out at 240 commands. The world-model selector makes no initial-grasp intervention. This also prevents fault injection for that layout: 99 stress runs per controller actually receive the opening request, and the model-assisted controller rejects all 99.

No ordinary gripper overrides occurred. Every model-assisted ordinary and protected stress state/action trace exactly matches its unperturbed BC reference, including the shared missed-grasp failure. The model therefore preserves the nominal trajectory under this fault; it does not improve natural task success on this suite. A simple rule remains a relevant unbenchmarked alternative for this specific error.

Worst normal/protected cube penetration is 1.794 mm, versus 5.754 mm for BC under the pulse. Worst jaw force remains 933.0 N; passing depth checks does not establish realistic or safe contact forces. Parallel-load inference timings are not presented as isolated-controller benchmarks.

All 400 condition/controller/layout outcomes are unique. Both checkpoints and all prior datasets/results remain unchanged. The complete prior 20-layout suite is retained with its 20/20 ordinary and 12/20 BC-pulse results; it is a prefix of this expanded suite and must not be added to it as independent observations.
