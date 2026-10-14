import argparse
import csv
import hashlib
import json
import numpy as np

from generate_demo import generate_demo, PROJECT_FOLDER, SOURCE_SPLITS

parser = argparse.ArgumentParser(description="Generate a versioned demonstration dataset.")
parser.add_argument("--version", default="v1")
parser.add_argument("--gripper-close-rate", type=float, default=2.0)
args = parser.parse_args()
if not args.version.isidentifier():
    parser.error("Use a simple version name, such as v2.")
if not np.isfinite(args.gripper_close_rate) or args.gripper_close_rate <= 0:
    parser.error("Closing rate must be finite and positive.")
settings = {"version": args.version, "gripper_close_rate": args.gripper_close_rate}
data_folder = PROJECT_FOLDER / "data" / f"demonstrations_{args.version}"
report_path = PROJECT_FOLDER / "results" / f"dataset_{args.version}.json"
if report_path.exists():
    raise SystemExit("This dataset report already exists; preserve it for comparison.")


def first_marker_center(source):
    path = PROJECT_FOLDER / "data" / "processed" / f"{source}_marker_corners.csv"
    with path.open() as file:
        for row in csv.DictReader(file):
            if row["detected"] == "1":
                return np.array([float(row["x_pixels"]), float(row["y_pixels"])])
    raise ValueError(f"No marker detected in {source}.")


# Approximate the different base layouts from their initial image positions.
reference_center = first_marker_center("IMG_5539")
base_offsets = {}
for source in SOURCE_SPLITS:
    dx, dy = first_marker_center(source) - reference_center
    base_offsets[source] = np.clip([-dy * 0.0002, -dx * 0.0002], -0.025, 0.025)

# One base demonstration per recording, with its original source split.
for source in SOURCE_SPLITS:
    output = data_folder / f"{source}.npz"
    if not output.exists():
        generate_demo(source=source, offset_xy=base_offsets[source], video=False, **settings)

# Twenty new starting positions per training recording.
seed = 19
rng = np.random.default_rng(seed)
for source, split in SOURCE_SPLITS.items():
    if split != "train":
        continue

    for index in range(20):
        name = f"{source}_aug{index:02d}"
        offset = rng.uniform([-0.025, -0.015], [0.025, 0.025])
        if not (data_folder / f"{name}.npz").exists():
            generate_demo(source=source, name=name, offset_xy=offset, video=False, **settings)

episodes = []
for path in sorted(data_folder.glob("*.npz")):
    with np.load(path, allow_pickle=False) as episode:
        source = str(episode["source_video"].item())
        split = str(episode["split"].item())
        assert float(episode["gripper_close_rate_rad_s"]) == args.gripper_close_rate
        assert episode["dataset_version"].item() == args.version
        assert split == SOURCE_SPLITS[source]
        raw_path = PROJECT_FOLDER / "data" / "raw" / f"{source}.mp4"
        transport_path = PROJECT_FOLDER / "data" / "processed" / f"{source}_transport.csv"
        assert str(episode["source_sha256"].item()) == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        assert str(episode["transport_sha256"].item()) == hashlib.sha256(transport_path.read_bytes()).hexdigest()

        finite = all(np.isfinite(episode[key]).all() for key in (
            "actions", "state_joints", "state_ee_full", "state_cube", "qpos", "qvel"
        ))
        episodes.append({
            "name": path.stem,
            "source": source,
            "split": split,
            "transitions": len(episode["actions"]),
            "task_success": bool(episode["task_success"]),
            "quality_pass": bool(episode["quality_pass"]),
            "success": bool(episode["success"]),
            "numeric_data_finite": finite,
            "max_cube_penetration_m": float(episode["max_cube_penetration_m"]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

report = {
    "seed": seed, "dataset_version": args.version,
    "gripper_close_rate_rad_s": args.gripper_close_rate,
    "variants_per_training_source": 20,
    "base_offset_rule": "0.2 mm per pixel; x=-image_y, y=-image_x; clipped to +/-25 mm",
    "base_offsets_xy_m": {source: offset.tolist() for source, offset in base_offsets.items()},
    "augmentation_offset_range_xy_m": [[-0.025, -0.015], [0.025, 0.025]],
    "source_splits": SOURCE_SPLITS,
    "original_augmented_data_used": False,
    "episodes": episodes,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open("x") as file:
    json.dump(report, file, indent=2)

for split in ("train", "validation", "test"):
    cases = [episode for episode in episodes if episode["split"] == split]
    accepted = sum(episode["success"] for episode in cases)
    print(f"{split}: {len(cases)} episodes, {accepted} passed task and quality checks.")

print(f"Saved report: {report_path}")
