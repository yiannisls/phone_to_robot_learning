# Record gripper teleoperation in MuJoCo

`record_gripper.py` lets you move the hand with the keyboard, control the gripper,
mark stable holds or slips, and save the resulting robot states and contact loads.
The simulator scene matches the existing demonstration setup: obstacles disabled,
bin moved to (-0.23, 0.57, 0.001), fixed hand orientation, 50 g cube, and the
selected 1 rad/s closing rate with a +/-3.5 N m jaw actuator limit.

## Your first recording

From the project root, with your homework Python environment active:

```bash
export ROBOT_LEARNING_HW3=/absolute/path/to/hw3_imitation_learning
python scripts/record_gripper.py --out data/teleoperation/grasp-001 --zarr --video
```

The window starts paused. Click inside it so it receives keyboard input, then
press Space. Recording and simulation advance together at 10 Hz. Paused wall-clock
time is excluded. The default time limit is 120 seconds of simulated time.

| Key | Action |
| --- | --- |
| A / D | Move along negative / positive simulation x |
| S / W | Move along negative / positive simulation y |
| F / R | Move down / up |
| O / C | Request full opening / closing |
| [ / ] | Decrease / increase the requested jaw angle by 0.05 rad |
| Space | Start or pause |
| B | Mark a stable hold |
| N | Mark slipping or a dropped object |
| T | Mark a successful transfer |
| G | Mark a failed attempt |
| Escape | Save the episode and quit |

Movement keys change the target by 5 mm per received key event. Holding a key
depends on the operating system's key-repeat behavior. The top view helps you
see the simulation axes. Movement and gripper keys operate while unpaused;
annotation keys also work while paused. Closing is limited to 1 rad/s; opening
is immediate. Cartesian commands retain the existing workspace limits and a
maximum 12 mm step. Hand rotation stays fixed.

For the first attempt, open the gripper, align it with the cube, lower slowly,
close, and lift a little. Mark B if it holds or N if it slips. Escape saves both
successful and failed attempts. A complete transfer into the bin is not required
for a useful grasp-force recording. Each invocation records one episode; use a
new output folder for each attempt.

## Saved data

| File | Contents |
| --- | --- |
| episode.npz | T actions and T+1 robot/cube states at 10 Hz; exact initial integration state |
| physics.npz | 500 Hz samples with named columns and units |
| events.json | Your labels with simulation time and state index |
| summary.json | Settings, provenance, force peaks and measured outcome quantities |
| episode.zarr | Lab-compatible current/actions/next arrays and physics samples, when --zarr is supplied |
| recording.avi | Two camera views and measurements, when --video is supplied |

The physics samples include requested and applied gripper targets, measured jaw
angle/velocity, jaw actuator torque, each jaw's summed normal cube-contact load,
cube contact penetration, cube position/velocity, hand position, and relative
cube/hand position. The measured actuator uses a unit rotational gear, so its
actuator-force output is torque in N m. Normal contact loads are in newtons;
they are separate from motor torque and do not represent resultant vector force
or all friction forces.

The recorder uses MuJoCo's mj_contactForce for each cube/jaw contact and sums
positive normal loads separately for Fixed_Jaw and Moving_Jaw. Samples are read
at each physics-step boundary after mj_step, using its contact solution and
simulator observation conventions. The display also shows each control interval's
force peaks so a short spike is visible even if it ends before the next display.

User labels are annotations, not automatic proof of slipping or success. Relative
cube/hand motion is recorded for later analysis; it can also change during intended
placement or release. Dataset split remains unassigned. These recordings are not
automatically included in BC or world-model training.

## Replay a recording

```bash
python scripts/replay_gripper.py data/teleoperation/grasp-001 --out work/grasp-001-replay
```

Replay rebuilds the same scene, restores the initial integration state including
controls and solver warm start, executes the recorded actions, and compares every
qpos/qvel value and physics sample. verification.json reports whether differences
stay within the strict tolerances. Existing folders are protected. Replay requires
the recorded MuJoCo version and the same simulator assets/settings.

## Checks completed before saving this recorder

The recorder was exercised with the 149 actions from the training-family
IMG_5539 demonstration in dataset v2. This is a scripted software check, not a
new human teleoperated demonstration. It saved 150 states and 7,450 force samples.
Replay reproduced qpos, qvel and all physics samples exactly in this run.

Other checks confirmed 2 ms sample spacing, 10 Hz state/action alignment, the
closing-rate and torque limits, zero jaw/cube loads before contact and after
release, and event/state timestamp alignment. A three-step desktop check opened
and closed the OpenCV window successfully. The first restricted display attempt
could not connect; the desktop-access check succeeded. No manual keyboard grasp
has been collected yet.

The scripted check's maximum normal loads were 459.14 N and 459.65 N per jaw;
maximum cube penetration was 1.422 mm. These retain the existing simulation-force
limitation and are not a new improvement in grip settings. Validation outputs
are in results/gripper_recorder_verification/, copied from the temporary checked
runs. Their origin is explicitly labelled scripted_recorder_check or replay.

## Comparing grip settings

Start with the defaults to establish one manual attempt. Later compare recordings
with the same cube, starting layout and grasp motion, changing one setting at a
time. The recorder exposes --gripper-close-rate, --jaw-torque-limit, --offset-xy,
--move-mm and --max-seconds. Include holds and slips, and compare sustained contact
loads as well as peaks. Unconstrained teleoperation alone cannot identify a minimum
required grip force; controlled replay comparisons are the next experiment.

Force requirements depend on the simulated mass, friction and contact parameters.
These data describe this simulation and do not establish physical-world force
requirements. The recorder does not automatically choose force limits or change
the trained policy.

## Follow the code

create_environment sets up the existing scene. physics_sample reads one force and
motion sample. step applies one 0.1-second command and records its 50 physics
substeps. save writes the episode and its metadata. main handles the keyboard and
display. replay_gripper.py reuses the same command executor for verification.

For a noninteractive software check, --commands accepts a saved T-by-4 action
array, and --headless suppresses the window. This path is labelled scripted and
must not be presented as a human recording.

## Zarr and agent-operated data

Add `--zarr` to save an `episode.zarr` store as well as the NPZ files. The lab
loader can read its named arrays; the store also preserves exact next states,
initial integration state, full velocities and force traces. Commands are
mocap-target increments plus a requested native gripper target, which may differ
from the action convention of an older lab dataset. The full format and the
13 checked agent-operated recordings are documented in
[agent_gripper_policy.md](agent_gripper_policy.md).

Those recordings are labelled as agent-operated. They remain outside training;
your later keyboard recordings will be labelled `human_keyboard_teleoperation`.
All output folders are protected against overwriting.
