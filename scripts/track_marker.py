import csv
from pathlib import Path
import sys
import cv2

# Get the video filename from the command we use to run this script.
if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/track_marker.py data/raw/IMG_5539.mp4")
video_path = Path(sys.argv[1])

video = cv2.VideoCapture(str(video_path))
if not video.isOpened():
    raise SystemExit(f"Could not open {video_path}")

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    video.release()
    raise SystemExit("The video has no usable frame-rate metadata.")
sample_every = max(1, round(fps / 10))

# The marker dictionary and ID used in this project.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

parameters = cv2.aruco.DetectorParameters()
# Try larger threshold windows to detect the marker during motion blur.
parameters.adaptiveThreshWinSizeMax = 73
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / f"{video_path.stem}_marker_corners.csv"

frame_number = 0
sampled_count = 0
detected_count = 0

with output_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "frame", "time_seconds", "x_pixels", "y_pixels", "detected",
        "corner_0_x", "corner_0_y", "corner_1_x", "corner_1_y",
        "corner_2_x", "corner_2_y", "corner_3_x", "corner_3_y"
    ])

    while True:
        success, image = video.read()
        if not success:
            break

        # Only inspect about 10 frames per second.
        if frame_number % sample_every == 0:
            sampled_count += 1
            corners, ids, _ = detector.detectMarkers(image)
            time_seconds = frame_number / fps

            if ids is not None and 0 in ids:
                marker_index = list(ids.flatten()).index(0)
                marker_corners = corners[marker_index][0]
                center = marker_corners.mean(axis=0)
                # Flatten four (x, y) pairs into eight CSV values.
                corner_values = marker_corners.flatten().tolist()
                # Save the first detection so we can inspect it.
                if detected_count == 0:
                    cv2.aruco.drawDetectedMarkers(image, corners, ids)
                    cv2.circle(image, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)
                    cv2.imwrite(str(output_path.with_suffix(".jpg")), image)
                writer.writerow([
                    frame_number, round(time_seconds, 3),
                    round(center[0], 1), round(center[1], 1), 1
                ] + corner_values)
                detected_count += 1
            else:
                writer.writerow([
                    frame_number, round(time_seconds, 3), "", "", 0
                ] + [""] * 8)

        frame_number += 1

video.release()

print(f"Read {frame_number} video frames.")
print(f"Found marker ID 0 in {detected_count} of {sampled_count} sampled frames.")
print(f"Saved its pixel positions to {output_path}.")
