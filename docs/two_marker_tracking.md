# Tracking the new videos with duplicate marker IDs

The new tracker distinguishes the object and table markers in all 311 samples
from the four accepted videos. Both markers decode as ID 0, so identity comes
from an explicitly identified starting table position and subsequent image motion.

| Recording | Valid pairs | Initial table centre, pixels |
| --- | ---: | --- |
| IMG_5565 | 90/90 | 241.25, 1336.5 |
| IMG_5571 | 67/67 | 249.75, 1313.0 |
| IMG_5572 | 82/82 | 251.5, 1310.0 |
| IMG_5573 | 72/72 | 252.25, 1307.75 |

These reference centres were read from the first-frame detector outputs and
identified as the table marker by visual review. They are annotations for this
batch, not a universal rule that the left marker is always the table marker.

## How the code assigns identity

The first assignment chooses the marker nearest the supplied table centre.
Subsequent assignments compare the two possible ways to match detections to the
previous table and object positions. The order returned by OpenCV does not determine
identity. Missing/extra detections, close markers, nearly tied assignments, and
implausibly large jumps are flagged. A flagged sample contains no invented corners;
tracking restarts from the reference seed afterward.

The pixel thresholds assume 1080 by 1920 videos sampled around 10 Hz and small
camera movement: minimum marker separation 100 pixels, assignment margin 20 pixels,
maximum table movement 50 pixels, maximum object movement 250 pixels per sample.
These gates must be reviewed for other resolutions, motion speeds or scenes.
Future recordings should use distinct IDs to reduce dependence on these assumptions.

The CSV's original corner/centre columns describe the object. Additional columns
describe the table and the object-minus-table image position. Relative pixels
compensate for image translation approximately. They do not correct camera rotation,
perspective or depth, and are not calibrated metre-scale robot trajectories.

## Verification

The table label stayed on the table and the object label stayed on the SSD in
the saved one-second preview strips. All 311 sampled pairs passed the assignment
checks. Synthetic checks shuffled detector order while moving the object and
confirmed consistent labels; missing, overlapping and implausible-jump cases
were rejected.

The table image centre varies by roughly 3–8.5 pixels per coordinate across
these clips. This includes detection noise and possible camera movement; it is
not a calibrated camera-motion estimate. Marker assignments and quality counts
apply to the sampled frames, not every 30 fps video frame.

## Run it

From the project root, with the existing Python environment active:

```bash
python scripts/track_two_markers.py data/raw/new_batch_20261007/accepted/IMG_5565.mp4 --reference-center 241.25 1336.5 --out data/processed/new_batch_20261007/dual_marker_v1/IMG_5565
```

For the other videos, use the table centres listed above and corresponding filenames.
Existing output folders are protected; choose a new `--out` folder for repeat runs.

Each output folder contains `markers.csv`, a labelled `identities.jpg` preview,
and `summary.json` with frame metadata, quality counts, assumptions and code/video
hashes. The four validated temporary runs were copied into the project after review.

## Next steps

Assign the new recording families to splits before training, select the transport
intervals, and reconstruct movement with the table reference. Camera calibration
and the physical black-square sizes remain unverified for the new recordings.
The new tracks have not entered policy or world-model training. Six other imported
clips still need review of detection gaps or marker overlap.

For the gripper experiment, add a MuJoCo teleoperation recorder that synchronizes
commands, measured jaw state, contact forces and object motion. Force peaks should
be saved at the physics rate. These data can support a controlled experiment on
closing rate and actuator limits, including failed/slipping grasps.
