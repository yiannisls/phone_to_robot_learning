# Next visual-data collection pilot

The first 18-attempt pilot has now been implemented and recorded: **12 task completions, six pickup failures at slot 2, and zero training inclusions**. See [the complete pilot and failure analysis](visual-pilot.md). The specification below records the matrix that was declared before collection; it is not a claim of a completed training dataset.

## First pilot: 18 candidates

Use one training source family, `IMG_5539`, and the three current cube locations as spatial slots: [-0.35, 0.35], [-0.23, 0.40], and [-0.35, 0.52] metres in XY. Keep the bin at [-0.23, 0.57] XY and use the fixed workspace camera. These locations define this small pilot; they are not the full shuffled HW3 distribution.

Enumerate all six ways to assign red, green and blue to these slots. For each arrangement, record three attempts: “Put the red cube in the bin,” “Put the green cube in the bin,” and “Put the blue cube in the bin.” This gives **6 colour arrangements × 3 instructions = 18 attempted episodes**.

For any one arrangement, the initial scene, camera and robot state must be the same across its three instructions. Only the target instruction and the teacher's subsequent motion should differ. Check that equality rather than allowing the selected target always to occupy the original pickup position. Robot inputs remain thirteen values; target one-hot labels and cube/bin coordinates stay outside the VLA input.

## Keep the phone data's role explicit

The original recorder mode replays a saved demonstration at its original pickup position. Selecting another colour in that mode puts that colour at that same position; that alone is not a balanced language dataset. The implemented `--slot-colors` mode instead assigns physical slots independently of the instruction and retargets the motion to the requested cube.

The implemented pilot retargets the phone-derived movement to the chosen slot and bin. It records interpolation needed to keep commands within 12 mm; the timing can stretch nonuniformly. Keep the measured transport shape/progress and label scripted approach, grasp, placement, release and retreat separately. Record any replaced transport tail, the retained fraction and all coordinate/timing transforms. A fully scripted transport must not be labelled as phone-derived motion. The original source files remain immutable.

Each attempt needs an episode ID, source family, arrangement ID, target colour, slot coordinates, bin position, instruction, camera settings, actual states/actions/timestamps, motion provenance, script/dependency fingerprints, task result and quality measurements. New outputs belong in their own folder, outside frozen baseline datasets.

## Quality and review

Record every prescribed attempt and retain failures. Use the existing task checks, maximum target penetration <=2 mm, and distractor displacement <10 mm. Also check finite data, command/image/state alignment, camera coverage, phone-motion contribution and consistent metadata. Contact-depth acceptance does not validate grip forces; that limitation stays visible.

Inspect examples for every target colour and pickup slot before freezing the pilot manifest. A helper stall, failed grasp, wrong-object movement, excessive penetration or clipped required workspace is a reported collection failure. Do not silently retry until only successes remain or present the accepted subset as a success rate. The collector's success rate and a learned policy's success rate are different measurements.

The 18 count is an attempt count, not a promise of 18 accepted demonstrations. The pilot is not ready for training: it lacks successful demonstrations for the third pickup slot, and robot self-contact needs inspection in addition to target-cube contact depth.

## Expansion after the pilot

After the retargeting and quality checks work, repeat the same matrix for `IMG_5540` and `IMG_5541`, giving **54 training candidates across three personal recording families**. Those are augmented simulation demonstrations, not 54 independent phone recordings. Then introduce declared training-layout variation and review its reachability, framing and collection failures before enlarging the corpus.

Preserve `IMG_5542` as validation and `IMG_5543` as test, including every generated descendant. Define separate validation/test layout groups before task training. Do not use test outcomes to choose the camera, teacher settings, normalization or checkpoint. The current train-only recorder must be extended deliberately before producing held-out visual data; this document does not claim that capability is already present.

Freeze accepted episode IDs, family/layout assignments and file checksums in a versioned manifest. Fit normalization on accepted training transitions only, before chunk padding, and reuse it unchanged for validation/test. Do not carry over the temporary single-episode statistics from the interface test.

A later evaluation must run the saved VLA closed-loop on a predetermined layout/instruction matrix, including identical layouts with different instructions. Report correct-object placement, wrong-object movement, grasp/release errors, counts and failures separately. The existing 99/100 single-cube BC/world-model result is a separate baseline.
