import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="IMG_5539")
parser.add_argument("--start", type=float, default=2.5)
parser.add_argument("--end", type=float, default=7.3)
args = parser.parse_args()
if args.end <= args.start:
    parser.error("--end must be after --start")
start_time = args.start
end_time = args.end

data_folder = Path(__file__).resolve().parents[1] / "data" / "processed"
input_path = data_folder / f"{args.source}_marker_corners.csv"
output_path = data_folder / f"{args.source}_movement_corners.csv"

selected_rows = []

with input_path.open() as file:
    reader = csv.DictReader(file)
    columns = reader.fieldnames + ["movement_seconds"]

    for row in reader:
        time = float(row["time_seconds"])

        if start_time <= time <= end_time:
            # Time measured from the beginning of the selected movement.
            row["movement_seconds"] = round(time - start_time, 3)
            selected_rows.append(row)

if not selected_rows:
    raise SystemExit("No samples found inside the selected interval.")

with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(selected_rows)

print(f"Selected {len(selected_rows)} samples.")
print(f"Movement duration: {end_time - start_time:.1f} seconds.")
print(f"Saved to {output_path}.")
