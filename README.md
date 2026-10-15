# Phone to Robot Learning

**Teaching a simulated robot to move an object using personally recorded phone videos.**

The internship challenge asks applicants to collect their own data and use it to drive a robot in simulation. I started with five phone videos of a simple manipulation task: pick up a marked object, carry it, and place it in a container. The aim is to transfer useful information from that human movement to a robot, then test what the robot actually learns.

![A real phone recording beside a learned robot rollout](docs/assets/phone-to-robot.png)

*Left: one of my original recordings, with an ArUco marker attached to the object. Right: the learned policy moving a cube with the simulated SO-100/101 arm. These are separate examples, not synchronized frames.*

**Current result:** both frozen phone-derived controllers complete **99/100 transfers** in the expanded position evaluation. A learned world model also helps reject a deliberately early gripper-opening command. Under an artificial early-opening request, BC completes **49/100** and BC + world model **99/100**.

[See the result](#3-the-result--testing-the-robot) · [Understand the pipeline](#2-the-middle--turning-movement-into-robot-demonstrations) · [Next: multicube VLA](#4-next--choose-the-right-cube-from-a-language-instruction) · [Run it](#run-the-project) · [Read the code](docs/project-map.md)

## 1. The start — collect a small, personal dataset

The starting point is an ordinary phone recording, rather than a robot demonstration. A black-and-white **ArUco marker** makes the object's movement easier to track: its four corners can be detected in each usable frame.

For each video, I select the interval covering the manipulation. The code tracks the marker, estimates its pose, smooths the motion, and adapts the path to the simulator's scale and direction.

![The adapted lift extracted from one phone recording](docs/assets/phone-transport.png)

*Example from IMG_5539. This is the path after smoothing, scaling and retiming for the robot. It is an approximation based on provisional camera calibration.*

The five recordings stay separated by source: three for training, one for validation and one reserved for testing. Starting-position variations are generated only from the three training recordings. **More simulated variations do not mean more independent real recordings.**

## 2. The middle — turning movement into robot demonstrations

A phone video shows how the object travels. It does not provide the robot commands needed to approach, grasp, or release it. To create the first robot dataset, the recorded transport path is combined with scripted approach, grasp, placement, release and retreat phases.

```mermaid
flowchart LR
    A[My phone recordings] --> B[Track the marked object]
    B --> C[Adapt its transport path]
    C --> D[Generate robot demonstrations]
    S[Scripted grasp and release phases] --> D
    D --> E[Learn actions with BC]
    D --> F[Learn state changes with a world model]
    E --> G[Move the simulated robot]
    F --> H[Compare possible gripper outcomes]
    H --> G
```

### Learning the actions: behavioral cloning

**Behavioral cloning (BC)** learns by example. Given the current robot and object measurements, a small neural network predicts upcoming movement and gripper commands. During execution, the robot follows a few commands, receives updated measurements, and predicts again.

The demonstrated trajectories teach the policy; the policy then chooses its own actions during evaluation. The current controller uses simulator state measurements. The next experiment adds camera images and colour instructions in a separate multicube task.

### Looking ahead: the world model

The **world model** learns to answer: “Given this state and this action, what state comes next?” It is trained on recorded simulator transitions, including training-only branches that try alternative actions.

During control, BC proposes the movement. The world model predicts what might happen under three gripper choices: follow BC, open, or close. A task-specific scoring rule compares those predictions over half a second, and the robot executes the first selected command before observing again.

The transition predictions are learned. The candidate choices, scoring rule and activation conditions are hand-written design decisions. This controller currently assists the gripper; BC supplies the arm movement.

## 3. The result — testing the robot

To test whether looking ahead changes behavior, one experiment deliberately requests opening while the cube is being carried. This is a controlled error injection, rather than a naturally occurring failure in every rollout.

![Saved simulation replay: early opening with BC versus BC plus the world model](docs/assets/gripper-comparison.gif)

*Left: BC receives the opening request and drops the cube. Right: the model-assisted controller rejects the request and completes the transfer. The animation replays saved simulator states from one paired example.*

![Success counts for ordinary execution and the artificial early-opening experiment](docs/assets/control-results-100.png)

| Phone-project experiment, 100 layouts | BC | BC + world model |
|---|---:|---:|
| Ordinary execution | 99/100 | 99/100 |
| One artificial early-opening request | 49/100 | 99/100 |

These are **400 controller runs across 100 layouts and two conditions**, with frozen checkpoints and controller settings. The suite was expanded from 20 to 100 positions; the first 20 are retained and counted once. Both controllers miss the initial grasp on one layout. In the 99 lifted stress trials, the model-assisted controller rejects every injected opening request. No ordinary interventions occur.

[100-layout protocol and outcomes](docs/phone-only-confirmation.md). Earlier development results remain available in the historical reports.

This establishes a useful result for one specific disturbance. It does not establish general robustness: the same cube and scene were used, a simple rule could also address this opening error, and simulated grip forces remain unrealistically high. The ordinary experiment shows no success-rate improvement from adding the world model.

[Control method and limitations](docs/world_model_control.md) · [Reproduction evidence](docs/reproduction.md)

## What worked, and what still needs work

- **Personal footage contributes to control.** Its transport paths become part of the demonstrations used to train the robot policy.
- **Alternative-action examples help the world model.** In selected validation probes, its half-second cube prediction error after a changed gripper request fell from 10.32 mm to 1.15 mm. This is prediction error, not a task-success score.
- **Grasp and release need robot-specific information.** The phone dataset currently relies on scripted examples for these phases; the contribution of the footage is the transport movement.
- **Physics and generalization remain limitations.** Large contact-force peaks, approximate camera calibration, a small real dataset and a narrow task prevent broader claims.

[Prediction experiments](docs/world_model_v2_results.md) · [Gripper investigation](docs/gripper_experiments.md)

## 4. Next — choose the right cube from a language instruction

The next experiment uses the HW3 multicube scene: red, green and blue cubes with shuffled positions and a varying bin position. The task is to follow an instruction such as **“Put the green cube in the bin.”** This makes the requested object part of the task, rather than always moving the same cube.

**Status: data and model integration. SmolVLA passes isolated synthetic and real-episode training/interface checks. No task-trained VLA checkpoint or simulation evaluation is available yet.** The 99/100 result above belongs to the state-based, single-cube experiment and does not predict performance here.

| What the VLA receives | What it should learn |
|---|---|
| Simulator camera views | Locate the cubes, robot and bin |
| “Put the green cube in the bin” | Select the requested colour |
| Robot measurements, such as joints and gripper opening | Choose movement and gripper commands |

```mermaid
flowchart LR
    A[Camera images] --> D[Pretrained VLA, then fine-tuning]
    B[Instruction: put the green cube in the bin] --> D
    C[Robot measurements] --> D
    D --> E[Movement and gripper actions]
    E --> F[Observe again in simulation]
```

### Keep the personal phone data central

The planned dataset will adapt the recorded transport paths to a selected cube and the bin, with the robot-specific approach, grasp and release phases documented separately. Successful demonstrations must be verified in the multicube scene before use. Existing phone footage supplies motion; it does not contain robot action labels or examples of selecting coloured cubes from language.

For each accepted robot demonstration, save synchronized images, robot measurements, action labels and a colour instruction. The existing state-only recordings do not already provide that visual dataset. Rendered simulator images will be labelled as simulated or reconstructed, rather than as direct phone-camera visual transfer.

### Which model?

**LeRobot is the training and dataset framework, not a separate VLA model.** The first practical candidate is [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla), a 450-million-parameter pretrained model. [π₀.₅](https://huggingface.co/docs/lerobot/main/pi05) is another candidate if compute and integration time allow. Both require adapting the data and action interface to this simulator. SmolVLA has passed synthetic and real-episode training/interface checks here; neither has been evaluated on the multicube task.

**Local hardware check:** pretrained SmolVLA passes isolated forward/backward/optimizer checks on both synthetic data and our recorded 13-state/4-action episode, using one camera, batch size one and rank-four LoRA on the RTX 4060 Laptop GPU (8 GB). Peak PyTorch allocation was 1.25 GiB (1.33 GiB reserved). This verifies the data interface and a small training configuration, not robot performance. [Real-batch checks](docs/vla-real-batch.md) · [Setup](docs/vla-setup.md)

The [fixed workspace camera](docs/camera-framing.md) gives larger cube images while preserving the physical trajectory. The [first 18-attempt collection pilot](docs/visual-pilot.md) completed 12 transfers; all six failures came from the third pickup slot, where the grasp collides with the arm and fails to track its target. Every attempt is retained. A [second pilot in a smaller workspace](docs/visual-workspace.md) moved that slot 9 cm toward the front and completed **18/18 scripted transfers**, with no detected robot self-penetration. This changes the collection geometry; the original position remains unresolved. Neither pilot is a VLA score, and no pilot episode has entered training.

### How the experiment will be evaluated

Compare BC + world model and SmolVLA only under a declared common task/execution protocol. The state controller receives exact target/bin coordinates; the VLA receives images and text. Keep the original single-cube results separate from future multicube scores. [Comparison protocol and required adaptations](docs/controller-comparison.md).

Use held-out layouts with balanced red, green and blue targets. Test the same layout with different colour instructions to check whether actions follow the instruction. Report correct-cube placement, wrong-cube selection, and grasp/release failures separately. Training, validation and test phone families remain separate, including generated descendants; normalization uses training examples only.

| Stage | Status |
|---|---|
| Phone-derived BC + world model, 100-position evaluation | Complete: 99/100 ordinary transfers for both |
| Inspect one synchronized multicube visual demonstration | Original placement failed; scripted placement correction passed the 2 mm check on one training layout; not yet added to training |
| Generate and review balanced colour-conditioned demonstrations | Original layout: 12/18; smaller workspace: 18/18 scripted completions; zero training inclusions |
| Inspect existing human multicube demonstrations | 42 teleoperation episodes + 7 DAgger segments verified; one episode converted; nine-state/four-action SmolVLA batch checks pass; not yet adopted |
| Convert the verified dataset and fine-tune a pretrained VLA | Corpus training planned; real 13-state/4-action batch and episode-end padding checks pass |
| Evaluate language selection and manipulation on held-out layouts | Planned |
| Resolve contact-physics limitations | Open |

The [first visual recording](docs/visual-episode.md) exposed a placement contact problem. A [scripted correction](docs/visual-placement.md) reduced maximum target penetration from 3.69 mm to 0.97 mm on that same training layout. It preserves a phone-transport prefix, then uses simulator feedback for placement. The candidate is not yet added to training, and this is not a VLA performance result.

The existing BC/world-model checkpoints and results remain the completed baseline. The current world model is not yet integrated with a VLA.

### Human teleoperation and earlier integration

A separate [BC/DAgger lab](https://github.com/yiannisls/so101-behavioral-cloning) achieved 97/100 original and 82/100 shifted-obstacle successes. Those are a different task setup and are not results of this phone-data controller. An integration pilot combined one human approach with phone-derived transport and scripted release; it completed a transfer but failed the contact-depth check. This integration is deferred and has not entered policy training.

The lab also contains **42 multicube teleoperation episodes and seven DAgger correction segments**. A [single-episode converter](docs/human-visual-episode.md) now reconstructs synchronized images from the saved robot/object poses and preserves the original action labels. Its nine-value measured robot input differs from the earlier phone interface. A [one-batch SmolVLA check](docs/vla-human-batch.md) now passes with nine measured state values and four action values, including final-action padding. These are independent temporary optimizer updates; no task-trained checkpoint or VLA task score is available. A [physics replay of the human labels](docs/human-physics-replay.md) completed one red transfer but failed the 2 mm contact-depth screen (5.05 mm measured); the example remains excluded from training.

[Archived integration pilot](docs/phone-lab-pilot.md) · [Data compatibility findings](docs/data-compatibility.md)

## Run the project

Start with the [guided walkthrough](docs/walkthrough.md). It covers installation, the separately obtained HW3 simulator, artifact verification, evaluation and training in that order. The saved-model archive is prepared locally; its public GitHub Release upload is pending. [Artifact publication instructions](docs/publishing.md)

Once the environment, simulator and saved artifacts are available, run from the repository root:

```bash
python scripts/evaluate_model_control.py --mode bc --video --out work/first-bc
python scripts/evaluate_model_control.py --mode wm --stress-open --video --out work/first-wm-opening
```

Each new output folder contains measurements and a rollout video. Choose new folder names when repeating an experiment; existing runs are protected from overwriting.

For the implementation, follow the [project map](docs/project-map.md). Detailed assumptions, source-family boundaries and historical experiments remain in the [design notes](docs/design_notes.md) and linked experiment reports. Simulator provenance and redistribution details are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).
