# Design notes for the final report

## Data and reproduction

The five phone recordings are personally collected data. They supply object
transport paths. The original project converted those paths into simulated robot
demonstrations, combining recorded transport with scripted approach, grasp,
release, and retreat. This rebuild will regenerate demonstrations and training
variants from the recordings using new, readable scripts, followed by policy and
world-model training. Learning will use the newly generated data.

The new extraction pipeline starts with marker centers and now saves all four
corners for pose estimation. IMG_5539 has a newly reviewed movement interval of
2.5 to 7.3 seconds in this rebuild. Record the remaining intervals before
generating demonstrations. Results from this rebuild will be measured separately
because extraction and implementation choices can differ from the original.

Keep source families together: IMG_5539, IMG_5540, and IMG_5541 are training;
IMG_5542 is validation; IMG_5543 is test. Generated variants and recovery examples
inherit their source's split. Training normalization uses training observations
only. Repeating previously evaluated layouts is reproduction, not a fresh test.

## Conversion assumptions

- ArUco dictionary DICT_4X4_50, object marker ID 0, measured black-square width 30 mm.
- Original pose extraction uses provisional camera intrinsics: focal length
  1794.146 pixels, principal point (540, 960), image size 1080 by 1920, zero
  distortion. Camera calibration has not been validated.
- Manual transport intervals, smoothing, alignment to the simulated start-to-bin
  direction, and retiming to 6.5 seconds at 10 Hz.
- Lift is scaled to 10 cm; lateral deviation is scaled by 0.4. Starting positions
  use a chosen 0.2 mm per pixel scale and are clipped to +/-25 mm.
- These transformations adapt recorded motion to the simulation; they are not
  validated metric reconstruction of the physical scene.
- The simulator uses a fixed bin, a 4 cm cube, and a course-provided Cartesian
  controller. The course simulator and assets require an independent checkout.

## Learning and evaluation

The remaining transport intervals come from earlier manual review. New source
tracking, pose estimates, transport curves and simulation replays are generated
by this rebuild. Training augmentation uses 20 seeded starting positions per
training source. Validation and test sources receive no augmentation. Base layout
offsets are recomputed from initial marker centers using the documented pixel scale.

The new transport script uses a five-sample moving average. It reduces lateral
deviation by a factor of 0.4, removes the start-to-end height difference, scales
the lift to 10 cm, and retimes the movement to 6.5 seconds. It saves 66 target
positions defining 65 intervals of 0.1 seconds.

The new generator starts with the previously corrected grasp offset
(-25, 0, 40) mm relative to the cube, a 2 rad/s closing limit, and measured-state
placement followed by release and 14 cm retreat. These settings are inherited
from earlier inspected experiments, not newly selected with a fresh holdout.

The new action format stores Cartesian commanded-target increments and the
requested native gripper target. Observations store the actual applied gripper
command after gradual closing. This differs from the original dataset's fourth
action component (the applied command). All new learners must use this documented
interface consistently. Every episode records T actions and T+1 observations,
source and transport hashes, split, success and cube contact-depth checks.

The original expanded behavioral-cloning baseline achieved 14/20 on the saved
layout suite. Later grasp and release teaching changes achieved 20/20 saved and
20/20 fresh layouts. The previous corrected-grasp policy achieved 19/20 on the
same fresh layouts. These are historical results until separately reproduced by
the new code; they must not be attributed automatically to this rebuild.

The world model predicts state changes from current state and action. It is
separate from behavioral cloning and is not currently used by the policy for
planning. The original baseline demonstrations and later smooth-closing
demonstrations have different execution profiles; evaluate the model on a
consistent profile before attempting control with it.

Large simulated gripper contact forces and generalization beyond the tested scene
remain limitations. Record unsuccessful experiments and AI assistance alongside
the final design choices.

The tracker uses an adaptive threshold window maximum of 73 pixels. This was
selected using training recording IMG_5540 to recover blurred marker detections
within its movement interval. Missing detections remain explicit; they are not
filled with invented positions.

Behavioral cloning uses 28 current-state measurements and predicts eight future
commanded-position offsets with gripper open/close decisions. Scaling uses
training data only; the checkpoint is selected by validation imitation loss.
Test data is not loaded during training. This loss measures imitation and does
not establish success in simulation.

The rebuild's policy uses 24,352 parameters. Validation selected epoch 100,
with 1.17 mm target-position MAE and 99.3% gripper accuracy. Its validation
rollout succeeded in 166 steps. Maximum cube penetration was 0.644 mm; peak
normal forces were 777 N at the fixed jaw and 740 N at the moving jaw.
These results cover one validation layout. The test recording has not been
evaluated by the learned policy.
