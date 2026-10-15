# Sources and data provenance

The simulation interface and scene are obtained separately from the ETH Zürich Robot Learning HW3 assignment: https://github.com/tenvinc/ethz-course-2026/tree/main/hw3_imitation_learning . Set ROBOT_LEARNING_HW3 to your local assignment directory. Course code is not included in this repository's artifact archive. The inspected course source does not provide a top-level license; this project does not assign it a new license. Preserve the robot model's own Apache-2.0 notices when obtaining its assets.

The applicant personally recorded the original five phone videos. Marker tracking estimates transport motion; robot approach, grasp, placement and retreat are scripted during demonstration generation. Automatically operated simulator recordings are labeled separately from human teleoperation. The later human BC/DAgger lab is a separate experiment and is not training data for the checkpoints distributed here.

No blanket license is applied to material with unresolved ownership or redistribution terms. This file records provenance, not a claim that all dependencies can be redistributed.

The separate VLA memory experiment uses [Hugging Face LeRobot](https://github.com/huggingface/lerobot/tree/v0.6.1), [SmolVLA base](https://huggingface.co/lerobot/smolvla_base), and the configuration/tokenizer of [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct). LeRobot is Apache-2.0 licensed; both model cards declare Apache-2.0. Pretrained weights are downloaded from their original repositories into an ignored local cache and are not included in this Git repository or the existing baseline artifact archive. Their authorship remains with the original contributors.
