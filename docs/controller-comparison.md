# Comparing the state controller with the VLA

Yes: compare the complete systems on the same manipulation task, while reporting their different inputs and training histories. Do not compare the existing single-cube 99/100 result directly with a future language-conditioned multicube score and call the difference an improvement.

Behavioural cloning is a training method: learning actions from demonstrations. A VLA can also be trained by imitation. The comparison here is **state-based BC plus a learned gripper selector** versus **fine-tuned SmolVLA using images, text and robot measurements**.

## What each system knows

| System | Inputs | Additional help |
|---|---|---|
| Existing BC | 28 state features including exact selected-cube and bin positions | Target identity supplied by the simulator/task |
| Existing BC + world model | BC inputs plus 35-value dynamics state with exact robot/cube state | Learned short forecasts; hand-coded gripper candidates, task score and gates |
| Planned SmolVLA | Camera image and instruction; 9 measured robot values in the human interface, 13 state/command values in the phone prototype | External pretrained weights, then task imitation training |

The VLA must infer the requested object from appearance and language. The state controller is given the requested object's exact state; it is a privileged-state reference, not a test of visual recognition or language grounding. A lower VLA task score would not by itself mean that adding vision/language is useless. A higher score would not isolate the architecture as the cause when inputs, pretraining and demonstrations differ.

The placement helper in the visual-data recorder is a scripted teacher for collecting demonstrations. It must not silently execute during a rollout labelled “VLA alone.” If a helper is used at deployment, report that hybrid as a separate system and disclose its simulator-state inputs.

## Keep historical results intact

The frozen phone baseline achieved 99/100 ordinary successes for both BC and BC + WM. Under an artificial early-opening request, BC achieved 49/100 and BC + WM 99/100. Those used the original single-cube scene, execution settings and saved protocol. They remain historical results, not entries measured in a new multicube table.

The current VLA work includes a [200-update human-only fine-tuning pilot](human-vla-training.md) with a saved adapter and reduced fixed-window validation loss. The [first development rollout](human-vla-rollout.md) failed to pick up the cube on one training start. No held-out VLA success-rate benchmark exists yet. This development executor uses native instant gripper targets, unlike the closing-rate limit proposed below. The 18-attempt visual pilot measures demonstration collection, not policy success.

## Proposed matched evaluation

Before policy training/testing, freeze the layout/colour/instruction/seed list and shared execution settings. Run every listed case with each controller, preserving failures rather than resampling successful cases.

- Use the same robot, multicube scene, bin geometry, object properties, initial physical state, MuJoCo version and success criteria.
- Use 10 Hz control, the existing 1 rad/s closing limit and +/-3.5 Nm jaw torque limits, and a maximum 12 mm target movement per control step.
- For this new multicube suite, propose 400 control steps (40 seconds) for every system. The slower demonstration teacher motivates a new common allowance; the historical 240-step baseline results cannot be reused as measurements under it.
- Use the same movement replanning interval: predict again after at most four executed commands. Log the world-model selector's additional state observations at each gripper decision.
- Report correct requested-cube placement, wrong-object movement, missed grasp, drop/release failure, timeout, out-of-bounds behavior, task-only success and contact-quality success. Apply the same 2 mm penetration and 10 mm distractor-displacement checks.
- Test an identical initial scene under all three colour instructions. Report instruction consistency separately; the state reference receives the oracle-selected target and does not earn a language-understanding claim.
- Keep ordinary execution and the artificial early-opening experiment in separate tables. In the fault test, use the same declared injection rule and report whether it actually triggered; a policy that never lifts the object has not successfully resisted the fault.
- Report sample counts, paired outcomes and inference latency, with hardware and observation/replanning rates. The VLA's pretraining, larger parameter count and GPU use are material differences.

These are proposed evaluation settings, not an implemented benchmark. The final case list must be recorded before evaluating trained policies; the collection pilot is not that test set.

## Adaptation is required

The BC feature builder already refers to a selected target cube, but the current evaluation script constructs the single-cube scene. The world model concatenates single-cube qpos/qvel into a fixed 35-value input. Passing the multicube simulator's full arrays would be incompatible.

A multicube comparison therefore needs an explicit target-state adapter, a common executor and validation of the world model's predictions under the new contacts and layouts. Matching array dimensions alone does not establish predictive accuracy. A transferred frozen model must be labelled as transfer from the old scene. If BC/WM are retrained on the new accepted corpus for a controlled data comparison, save them as new models; do not replace the historical checkpoints. No additional DAgger work is required by this comparison plan.

Preserve source-family boundaries: IMG_5539/5540/5541 training, IMG_5542 validation, IMG_5543 test. Fit normalization on training transitions only. For the most controlled data comparison, give both learned controllers the same accepted demonstration episodes and clearly disclose the VLA's external pretraining. A frozen old baseline trained on different demonstrations is a useful reference, but not a controlled architecture ablation.

The first human-only adapter and development rollout are recorded. Implement the matched executor and target-state adapter as a separate, reviewable stage before publishing any BC + WM versus VLA score table.
