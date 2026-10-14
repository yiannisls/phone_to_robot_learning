# New recordings: quality review, 7 October 2026

Four recordings passed the strict sampled-marker check. Six are preserved for
review rather than discarded. All ten decode completely at 30 fps and 1080 × 1920.
None has entered training or changed the existing source split.

| Video | Duration | Both markers at 10 Hz | Decision |
| --- | ---: | ---: | --- |
| IMG_5557.mp4 | 8.17 s | 75/82 | review_required |
| IMG_5565.mp4 | 9.00 s | 90/90 | accepted |
| IMG_5566.mp4 | 6.97 s | 69/70 | review_required |
| IMG_5567.mp4 | 7.10 s | 67/71 | review_required |
| IMG_5568.mp4 | 6.47 s | 64/65 | review_required |
| IMG_5569.mp4 | 7.77 s | 71/78 | review_required |
| IMG_5570.mp4 | 6.70 s | 64/67 | review_required |
| IMG_5571.mp4 | 6.63 s | 67/67 | accepted |
| IMG_5572.mp4 | 8.13 s | 82/82 | accepted |
| IMG_5573.mp4 | 7.20 s | 72/72 | accepted |

## Marker identity needs attention

Both the table marker and the object marker decode as ID 0. The current tracker
uses the first detection with ID 0; that ordering is not a reliable identity.
Before extracting motion, identify the stationary reference and moving marker
separately, and flag ambiguous overlap instead of silently switching identities.
For future filming use distinct IDs (for example table 1, object 0). These clips
do not all need to be re-recorded.

## Clips retained for review

- **IMG_5557.mp4:** The object passes over the table marker, temporarily hiding it; location-only marker association is ambiguous during this overlap. Review the reference gap before pose reconstruction.
- **IMG_5566.mp4:** One object-marker detection gap at 5.2 s, near placement/release.
- **IMG_5567.mp4:** Object-marker gaps at 5.2–5.4 s and 5.8 s near placement/release; hand partly covers the marker.
- **IMG_5568.mp4:** One object-marker detection gap at 5.2 s, near placement/release.
- **IMG_5569.mp4:** Object-marker gaps at 4.9–5.4 s and 5.6 s near placement/release; longest missing interval in this batch.
- **IMG_5570.mp4:** Object-marker gaps at 4.3–4.4 s and 5.2 s near placement/release.

IMG_5557 has a particularly clear overlap of the two markers. Most other gaps
occur around placement or hand withdrawal. Some may fall outside the eventual
transport crop, but no movement intervals have been selected here. Short gaps
must not be silently interpolated and treated as measured data.

The four accepted clips have both markers detected in all 10 Hz samples and
complete transfers in the visual contact sheets. This is a recording-quality
decision, not a claim of calibrated trajectories or readiness for the old tracker.

## Saved files

Videos are under `data/raw/new_batch_20261007/accepted/` and
`data/raw/new_batch_20261007/review_required/`. The batch manifest records source
hashes, metadata, quality decisions and `split: unassigned`. Each clip is one
source family; future derivatives must inherit its assigned split. The old
three-training / one-validation / one-test source split remains unchanged.

Raw detector observations, contact sheets and review code are in
`results/video_review_20261007/`. Duplicate marker IDs are retained as separate
detections in the raw observations; they are not collapsed into a single marker.
The review code uses the attachment paths recorded in the manifest.

Before using this batch: assign recording families to splits, resolve marker
identity, select movement intervals and then generate a separate dataset version.
The existing batch preparation script does not automatically include this folder.

## Teleoperated gripper data in MuJoCo

The user confirmed that proposed teleoperation is in our MuJoCo simulation.
A useful collection should synchronize commanded gripper target, applied target,
measured jaw position/velocity, actuator effort, per-jaw cube contact forces,
cube pose/velocity, and slip/drop outcomes. Log at physics rate for force peaks
and retain the 10 Hz policy transitions too. Actuator torque and jaw contact
force are different measurements and should be labelled separately.

Use the same object mass, friction and simulator settings when comparing trials.
Teleoperation can supply varied grasps; a controlled closing-rate and actuator-limit
experiment is better for comparing settings than unconstrained demonstrations alone.
Record failed/slipping trials as well as stable holds. These measurements tune
the simulated gripper; they do not validate physical-world force requirements.
No teleoperation collector or new force experiment was run during this import.
