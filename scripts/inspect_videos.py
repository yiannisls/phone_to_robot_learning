from pathlib import Path
import cv2

# Find the project's data/raw folder, starting from this script's location.
project_folder = Path(__file__).resolve().parents[1]
video_folder = project_folder / "data" / "raw"

# Check each MP4 recording.
for video_path in sorted(video_folder.glob("*.mp4")):
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"Could not open: {video_path.name}")
        continue

    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    seconds = frames / fps if fps > 0 else 0

    print(
        f"{video_path.name}: "
        f"{fps:.2f} frames/second, "
        f"{int(width)} x {int(height)} pixels per frame, "
        f"{seconds:.1f} seconds"
    )

    video.release()
