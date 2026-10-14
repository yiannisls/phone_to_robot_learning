# Publish the verified milestone

The repository contains source, documentation, plots, small result summaries and artifact checksums. The release archive contains the original five videos, extracted data, preserved demonstration versions, training-only action variations, four checkpoints and historical evaluation evidence. It excludes the external course simulator, later unused phone recordings, and unrelated teleoperation collections.

Before upload, verify the archive and keep the exact release version paired with the source commit:

```bash
python scripts/release_artifacts.py verify
```

Run the checksum command from the archive directory, or use:

```bash
(cd dist && sha256sum -c phone-robot-learning-v0.1.0.tar.gz.sha256)
```

After authenticating GitHub CLI, the intended public repository name is `phone-robot-learning`. Check whether it already exists before creating it. If it does, inspect its contents before setting a remote; do not overwrite an unrelated repository.

```bash
gh auth status
gh repo create phone-robot-learning --public --source . --remote origin --push
gh release create v0.1.0 dist/phone-robot-learning-v0.1.0.tar.gz dist/phone-robot-learning-v0.1.0.tar.gz.sha256 --target main --title 'Phone demonstrations, BC and world-model gripper control' --notes-file docs/reproduction.md
```

After publication, add the actual repository and release links to the README, verify downloads into a fresh clone, and commit that documentation update. Do not claim a public release before these commands succeed. No release currently exists for this local repository.
