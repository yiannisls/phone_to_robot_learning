# Reproduction verification — 14 October 2026

Verified on Linux with Python 3.12.3, MuJoCo 3.14.0 and CPU policy inference. The exact clean-environment packages are recorded in `requirements-lock.txt`. Hardware/driver portability beyond this machine has not been tested.

- All 1,418 preserved release artifact checksums passed.
- The two original demonstration versions retain 63 training, one validation and one test episode each; every augmented episode retains its source-family split. Test episode metadata was inspected only for integrity, not model selection.
- All 30 recorded simulator dependency fingerprints matched the separately supplied local HW3 files.
- The saved BC validation rollout succeeded in 131 steps and passed the 2 mm penetration check. Peak normal force remained 735.94 N: this is not a force-safety result.
- A fresh local Git clone plus the extracted release archive passed all checksums.
- Dependencies installed successfully into an isolated Python environment. In that environment, world-model control with an injected opening request succeeded in 131 steps, rejecting exactly one request. Its saved states, actions and physics traces exactly matched the unperturbed BC validation run.
- The verifier correctly rejected a deliberately corrupted file and a missing file in a temporary test directory. All source scripts parsed successfully.

The clean clone used the separately supplied local simulator, not an independently downloaded upstream copy. Exact simulator fingerprints are recorded because an upstream revision for that local directory could not be established.

The versioned archive contains historical experiment evidence and the saved comparison video. Newly reproduced outputs are kept under `work/reproducibility-20261014`; aggregate comparison verification is recorded below. No weights were retrained or altered.

GitHub publication is pending authentication. The archive round-trip was tested locally, not downloaded from a public Release.

## Full paired comparison

All 80 runs completed with seed 20261009 on the previously inspected development layouts. Ordinary BC and model-assisted control both succeeded 20/20 with contact-depth passes. Under the injected opening, BC succeeded 9/20 (0/20 contact-depth passes); model-assisted control succeeded 20/20 with depth passes. It made zero ordinary overrides and one override per stress episode. Worst force remained 1,057.48 N. These counts reproduce the historical experiment; they are not new generalization evidence.

See [machine-readable results](reproduction-results.json). Source control code and model hashes remain unchanged. No training was performed.
