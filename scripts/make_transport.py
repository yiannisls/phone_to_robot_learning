import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="IMG_5539")
args = parser.parse_args()

project_folder = Path(__file__).resolve().parents[1]
data_folder = project_folder / "data" / "processed"
input_path = data_folder / f"{args.source}_movement_3d.csv"
output_path = data_folder / f"{args.source}_transport.csv"

data = np.genfromtxt(input_path, delimiter=",", names=True)
source_time = data["movement_seconds"]
positions = np.column_stack([data["x_m"], data["y_m"], data["z_m"]])

# Reduce small fluctuations using a five-sample moving average.
smoothed = np.empty_like(positions)
for axis in range(3):
    padded = np.pad(positions[:, axis], (2, 2), mode="edge")
    smoothed[:, axis] = np.convolve(padded, np.ones(5) / 5, mode="valid")

# Describe the selected movement relative to its starting position.
movement = smoothed - smoothed[0]

# Separate travel towards the destination from sideways movement.
displacement = movement[-1, :2]
distance = np.linalg.norm(displacement)
if distance < 0.01:
    raise SystemExit("The estimated horizontal movement is too small.")

direction = displacement / distance
side_direction = np.array([-direction[1], direction[0]])

progress = movement[:, :2] @ direction / distance
lateral = 0.4 * (movement[:, :2] @ side_direction)

# Adapt the lift so both ends are at the same height.
height = movement[:, 2] - np.linspace(0, movement[-1, 2], len(movement))
height = np.maximum(height, 0)

if height.max() < 0.005:
    raise SystemExit("The estimated lift is too small.")

height = height / height.max() * 0.10

# Preserve the movement's timing pattern while stretching it to 6.5 seconds.
duration = 6.5
robot_time = np.linspace(0, duration, 66)
source_progress_time = source_time / source_time[-1]
robot_progress_time = robot_time / duration

transport = np.column_stack([
    robot_time,
    np.interp(robot_progress_time, source_progress_time, progress),
    np.interp(robot_progress_time, source_progress_time, lateral),
    np.interp(robot_progress_time, source_progress_time, height),
])

np.savetxt(
    output_path, transport, delimiter=",",
    header="time_seconds,progress,lateral_m,height_m", comments=""
)

print(f"Saved {len(transport)} transport points to {output_path}")
print("Duration: 6.5 seconds; interval between points: 0.1 seconds.")
