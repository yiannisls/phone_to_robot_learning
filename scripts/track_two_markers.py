"""Track object and table markers with duplicate IDs using explicit initialization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def assign_pair(corners, previous, reference_seed):
    """Return [table, object] corners, or a reason we cannot assign them."""
    if len(corners) != 2:
        return None, "need_exactly_two_markers"
    centers = np.array([marker.mean(axis=0) for marker in corners])
    if np.linalg.norm(centers[0] - centers[1]) < 100:
        return None, "markers_too_close"

    if previous is None:
        distances = np.linalg.norm(centers - reference_seed, axis=1)
        reference = int(np.argmin(distances))
        if distances[reference] > 100 or abs(distances[0] - distances[1]) < 20:
            return None, "reference_seed_ambiguous"
        order = [reference, 1 - reference]
    else:
        orders = ([0, 1], [1, 0])
        costs = [np.linalg.norm(centers[order] - previous, axis=1).sum() for order in orders]
        if abs(costs[0] - costs[1]) < 20:
            return None, "assignment_ambiguous"
        order = orders[int(np.argmin(costs))]
        jumps = np.linalg.norm(centers[order] - previous, axis=1)
        if jumps[0] > 50 or jumps[1] > 250:
            return None, "movement_too_large"
    return np.array([corners[i] for i in order]), "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--reference-center", type=float, nargs=2, required=True,
                        metavar=("X", "Y"), help="Table marker centre in the first frame, in pixels")
    parser.add_argument("--out", type=Path, required=True, help="New output folder")
    parser.add_argument("--marker-id", type=int, default=0)
    args = parser.parse_args()
    reference_seed = np.array(args.reference_center)
    if not np.isfinite(reference_seed).all():
        parser.error("Reference centre must be finite")
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        video.release()
        raise SystemExit("Invalid frame rate")
    width, height = int(video.get(3)), int(video.get(4))
    sample_every = max(1, round(fps / 10))
    args.out.mkdir(parents=True, exist_ok=False)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMax = 73
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)

    # The first columns describe the OBJECT and match the original CSV schema.
    columns = ["frame", "time_seconds", "x_pixels", "y_pixels", "detected"]
    columns += [f"corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")]
    columns += ["reference_x_pixels", "reference_y_pixels"]
    columns += [f"reference_corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")]
    columns += ["relative_x_pixels", "relative_y_pixels", "assignment_status"]
    previous, rows, previews = None, [], []
    frame = 0
    # Frame samples every second are displayed for checking identities manually.
    try:
        while True:
            success, image = video.read()
            if not success:
                break
            if frame % sample_every == 0:
                corners, ids, _ = detector.detectMarkers(image)
                matching = [] if ids is None else [c[0] for c, i in zip(corners, ids.ravel())
                                                   if int(i) == args.marker_id]
                pair, status = assign_pair(matching, previous, reference_seed)
                row = {key: "" for key in columns}
                row.update(frame=frame, time_seconds=frame / fps,
                           detected=int(pair is not None), assignment_status=status)
                if pair is not None:
                    centers = pair.mean(axis=1)
                    previous = centers.copy()
                    row.update(x_pixels=float(centers[1, 0]), y_pixels=float(centers[1, 1]),
                               reference_x_pixels=float(centers[0, 0]),
                               reference_y_pixels=float(centers[0, 1]),
                               relative_x_pixels=float(centers[1, 0] - centers[0, 0]),
                               relative_y_pixels=float(centers[1, 1] - centers[0, 1]))
                    for i in range(4):
                        for j, axis in enumerate(("x", "y")):
                            row[f"corner_{i}_{axis}"] = float(pair[1, i, j])
                            row[f"reference_corner_{i}_{axis}"] = float(pair[0, i, j])
                    for role, marker, color in zip(("TABLE", "OBJECT"), pair,
                                                   ((255, 100, 0), (0, 180, 0))):
                        cv2.polylines(image, [marker.astype(np.int32)], True, color, 5)
                        centre = marker.mean(axis=0).astype(int)
                        cv2.putText(image, role, tuple(centre + [0, -30]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                else:
                    # Require reinitialization rather than carrying identity across a gap.
                    previous = None
                rows.append(row)
                if frame % round(fps) == 0:
                    tile = cv2.resize(image, (216, 384))
                    cv2.rectangle(tile, (0, 0), (216, 28), (255, 255, 255), -1)
                    cv2.putText(tile, f"{frame / fps:.1f}s: {status}", (4, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                    previews.append(tile)
            frame += 1
    finally:
        video.release()

    with (args.out / "markers.csv").open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if previews:
        cv2.imwrite(str(args.out / "identities.jpg"), np.concatenate(previews, axis=1))
    valid = [row for row in rows if row["detected"]]
    reference = np.array([[row["reference_x_pixels"], row["reference_y_pixels"]] for row in valid])
    summary = {
        "video": str(args.video.resolve()), "fps": fps, "frame_size_pixels": [width, height],
        "decoded_frames": frame, "duration_s": frame / fps,
        "sample_every_frames": sample_every, "samples": len(rows), "valid_pairs": len(valid),
        "reference_seed_pixels": reference_seed.tolist(), "marker_id": args.marker_id,
        "status_counts": {status: sum(row["assignment_status"] == status for row in rows)
                          for status in sorted({row["assignment_status"] for row in rows})},
        "reference_coordinate_range_pixels": np.ptp(reference, axis=0).tolist() if len(reference) else None,
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assumptions": "Manually identified table seed; small inter-frame camera movement; exactly two markers of the selected ID; pixel gates suit this 1080x1920 batch.",
        "limitations": "Relative pixels remove image translation only, not rotation, perspective or depth. No interpolated detections, pose calibration, movement interval, split assignment or training here.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.video.name}: {len(valid)}/{len(rows)} unambiguous marker pairs; {args.out}", flush=True)


if __name__ == "__main__":
    main()
