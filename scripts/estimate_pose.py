import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="IMG_5539")
args = parser.parse_args()

# Physical marker size and provisional camera settings.
marker_size_m = 0.03
camera_matrix = np.array([
    [1794.146, 0, 540],
    [0, 1794.146, 960],
    [0, 0, 1],
], dtype=np.float64)
distortion = np.zeros(5)

# Coordinates of the marker's four corners on its own flat surface.
half = marker_size_m / 2
marker_points = np.array([
    [-half, half, 0],
    [half, half, 0],
    [half, -half, 0],
    [-half, -half, 0],
], dtype=np.float32)

project_folder = Path(__file__).resolve().parents[1]
data_folder = project_folder / "data" / "processed"
reference_path = data_folder / f"{args.source}_marker_corners.csv"
input_path = data_folder / f"{args.source}_movement_corners.csv"
output_path = data_folder / f"{args.source}_movement_3d.csv"


def estimate_pose(row):
    image_points = np.array([
        [float(row[f"corner_{i}_x"]), float(row[f"corner_{i}_y"])]
        for i in range(4)
    ], dtype=np.float32)

    success, rotation, position = cv2.solvePnP(
        marker_points, image_points, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise SystemExit(f"Pose estimation failed at frame {row['frame']}.")

    return rotation, position.reshape(3)


# Use the first detected marker as the reference.
# Comparing later poses with it assumes the camera stays fixed.
with reference_path.open() as file:
    reference_row = next(
        (row for row in csv.DictReader(file) if row["detected"] == "1"),
        None
    )

if reference_row is None:
    raise SystemExit("No detected marker available for the reference.")

reference_rotation, reference_position = estimate_pose(reference_row)
reference_axes = cv2.Rodrigues(reference_rotation)[0]

with input_path.open() as source, output_path.open("w", newline="") as destination:
    writer = csv.writer(destination)
    writer.writerow(["time_seconds", "movement_seconds", "x_m", "y_m", "z_m"])

    for row in csv.DictReader(source):
        if row["detected"] != "1":
            raise SystemExit(f"Missing marker at frame {row['frame']}.")

        _, position = estimate_pose(row)

        # Express displacement along the initial marker's axes.
        relative_position = reference_axes.T @ (position - reference_position)

        writer.writerow([
            row["time_seconds"], row["movement_seconds"], *relative_position
        ])

print(f"Saved {output_path}")
print("Positions are in metres, using provisional camera calibration.")
