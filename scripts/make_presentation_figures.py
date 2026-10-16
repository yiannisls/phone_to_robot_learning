"""Make README figures from the original recording and saved measurements.

Run from the project root:
    python scripts/make_presentation_figures.py --out work/readme-figures

This only reads existing data. It does not regenerate demonstrations or results.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "IMG_5539"
FRAME = 105  # 3.5 seconds: the hand and marker are both visible.
NAVY = "#192f4e"
TEAL = "#008c95"
GREY = "#7a8fa6"
AXIS_COLORS = ["#ed4545", "#31a35b", "#2987ed"]


def read_frame(path, frame):
    video = cv2.VideoCapture(str(path))
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = video.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame} from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        video.release()


def marker_axes(row):
    # Match estimate_pose.py: a 3 cm marker and provisional camera settings.
    camera = np.array([[1794.146, 0, 540], [0, 1794.146, 960], [0, 0, 1.]])
    half = 0.03 / 2
    corners_3d = np.array([
        [-half, half, 0], [half, half, 0],
        [half, -half, 0], [-half, -half, 0],
    ], dtype=np.float32)
    corners = np.array([
        [float(row[f"corner_{i}_x"]), float(row[f"corner_{i}_y"])]
        for i in range(4)
    ], dtype=np.float32)
    ok, rotation, position = cv2.solvePnP(
        corners_3d, corners, camera, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("Marker pose estimation failed.")
    points = np.array([[0, 0, 0], [.035, 0, 0], [0, .035, 0], [0, 0, .035]])
    pixels, _ = cv2.projectPoints(points, rotation, position, camera, np.zeros(5))
    fitted, _ = cv2.projectPoints(corners_3d, rotation, position, camera, np.zeros(5))
    error = np.linalg.norm(fitted.reshape(4, 2) - corners, axis=1).mean()
    return pixels.reshape(4, 2), float(error)


def draw_axes(ax, points):
    origin = points[0]
    ax.scatter(*origin, s=30, color="white", edgecolor=NAVY, zorder=5)
    for endpoint, name, color in zip(points[1:], "XYZ", AXIS_COLORS):
        ax.annotate("", xy=endpoint, xytext=origin,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 3})
        ax.annotate(name, xy=endpoint, xytext=(7, 7), textcoords="offset points",
                    color=color, fontsize=17, weight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})


def save_figure(fig, path):
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def make_marker_figure(image, axes, out):
    fig, panels = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(top=.78, bottom=.10, left=.04, right=.96, wspace=.12)
    fig.text(.04, .94, "The marker gives the object a reference frame", size=22, weight="bold", color=NAVY)
    fig.text(.04, .88, "TF: estimated marker position and orientation relative to the camera", size=13, color=NAVY)
    for ax in panels:
        ax.imshow(image)
        draw_axes(ax, axes)
        ax.axis("off")
    panels[0].set_xlim(0, 1080)
    panels[0].set_ylim(1450, 450)
    panels[0].set_title("My recording", loc="left", pad=12, fontsize=14)
    x, y = axes[0]
    panels[1].set_xlim(x - 220, x + 220)
    panels[1].set_ylim(y + 220, y - 220)
    panels[1].set_title("The same marker, enlarged", loc="left", pad=12, fontsize=14)
    fig.text(.04, .035, "Four detected corners define the frame. The coloured arrows show its X, Y and Z axes.", size=12, color=NAVY)
    save_figure(fig, out / "phone-marker-frame.png")


def make_path_figure(image, movement, transport, out):
    fig, (photo, chart) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(top=.78, bottom=.14, left=.04, right=.96, wspace=.25)
    fig.text(.04, .94, "From my movement to a robot path", size=22, weight="bold", color=NAVY)
    fig.text(.04, .88, "One recording: track the object, then smooth and adapt its movement", size=13, color=NAVY)
    photo.imshow(image)
    x, y = movement["x_pixels"], movement["y_pixels"]
    photo.plot(x, y, color="white", lw=5, alpha=.85)
    photo.plot(x, y, color=TEAL, lw=2.5)
    photo.scatter(x, y, s=8, color=TEAL)
    for index, name, shift in [(0, "Start", (8, -28)), (-1, "Finish", (5, 24))]:
        photo.scatter(x[index], y[index], s=60, color=TEAL, edgecolor="white", zorder=4)
        photo.annotate(name, (x[index], y[index]), xytext=shift, textcoords="offset points",
                       fontsize=12, color=NAVY, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95})
    photo.set_xlim(0, 1080)
    photo.set_ylim(1450, 450)
    photo.axis("off")
    photo.set_title("Tracked object path · 2.5–7.3 seconds", loc="left", fontsize=13, pad=12)
    time, height = transport["time_seconds"], 100 * transport["height_m"]
    chart.plot(time, height, color=TEAL, lw=3)
    chart.fill_between(time, height, color=TEAL, alpha=.10)
    chart.set(xlabel="Robot transport time (seconds)", ylabel="Adapted lift (cm)", ylim=(0, 11))
    chart.set_title("Adapted robot lift", loc="left", fontsize=13, pad=12)
    chart.grid(axis="y", color="#e1e6ec")
    chart.spines[["top", "right"]].set_visible(False)
    chart.set_box_aspect(.8)
    fig.text(.04, .035, "The image path follows the marker. The lift curve comes from the saved, adapted transport data.", size=12, color=NAVY)
    save_figure(fig, out / "phone-motion-path.png")


def make_results_figure(report, out):
    fig, panels = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(top=.76, bottom=.22, left=.09, right=.97, wspace=.15)
    fig.text(.05, .94, "What changes when the robot looks ahead?", size=22, weight="bold", color=NAVY)
    fig.text(.05, .86, "100 layouts per controller and condition · 400 simulation runs", size=13, color=NAVY)
    for ax, condition, title in zip(panels, ["normal", "opening_pulse"], ["Normal transfer", "Deliberate early opening"]):
        entries = [report["summary"][condition][mode] for mode in ["bc", "wm"]]
        values = [100 * row["task_successes"] / row["episodes"] for row in entries]
        bars = ax.bar(["BC", "BC + world model"], values, color=[GREY, TEAL], width=.6)
        ax.bar_label(bars, labels=[f"{value:g}%" for value in values], padding=5, size=17, weight="bold", color=NAVY)
        ax.set_title(title, fontsize=15, pad=16, color=NAVY)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", color="#e1e6ec")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    panels[0].set_ylabel("Task success", fontsize=13)
    save_figure(fig, out / "control-success-percent.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": NAVY})
    raw = args.root / "data/raw" / f"{SOURCE}.mp4"
    processed = args.root / "data/processed"
    corners_path = processed / f"{SOURCE}_marker_corners.csv"
    with corners_path.open() as file:
        row = next(row for row in csv.DictReader(file) if int(row["frame"]) == FRAME and row["detected"] == "1")
    image = read_frame(raw, FRAME)
    axes, error = marker_axes(row)
    movement_path = processed / f"{SOURCE}_movement_corners.csv"
    transport_path = processed / f"{SOURCE}_transport.csv"
    movement = np.genfromtxt(movement_path, delimiter=",", names=True)
    if not np.all(movement["detected"] == 1):
        raise ValueError("The selected movement contains missing detections.")
    transport = np.genfromtxt(transport_path, delimiter=",", names=True)
    report_path = args.root / "docs/phone-only-confirmation.json"
    report = json.loads(report_path.read_text())
    make_marker_figure(image, axes, args.out)
    make_path_figure(image, movement, transport, args.out)
    make_results_figure(report, args.out)
    sources = {
        "source_recording": SOURCE, "frame": FRAME, "frame_time_seconds": 3.5,
        "marker_size_m": .03, "axis_length_m": .035,
        "mean_corner_reprojection_error_pixels": error,
        "axes_projected_pixels": axes.tolist(),
        "pose_method": "Same solvePnP settings as estimate_pose.py; provisional camera calibration",
        "trajectory": "Saved marker-centre image coordinates; hand is visible but not tracked",
        "transport": "Saved smoothed and retimed transport, not raw measured height",
        "sha256": {},
    }
    for path in [raw, corners_path, movement_path, transport_path, report_path]:
        sources["sha256"][str(path.relative_to(args.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / "presentation-sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Saved three figures and source records to {args.out}")


if __name__ == "__main__":
    main()
