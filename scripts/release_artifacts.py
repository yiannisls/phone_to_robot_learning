"""Verify immutable artifacts, or package them for a GitHub Release."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def verify(root, manifest):
    for relative, expected in manifest['sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise SystemExit(f'Unsafe manifest path: {relative}')
        if not path.is_file():
            raise SystemExit(f'Missing artifact: {relative}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Checksum mismatch: {relative}')
    print(f"Verified {len(manifest['sha256'])} artifacts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'pack'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'artifact-manifest.json').read_text())
    verify(args.root, manifest)
    if args.command == 'pack':
        if args.out is None:
            parser.error('pack requires --out pointing to a new .tar.gz file')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('xb') as output:
            with tarfile.open(fileobj=output, mode='w:gz') as archive:
                for relative in sorted(manifest['sha256']):
                    archive.add(args.root / relative, arcname=relative, recursive=False)
        checksum = hashlib.sha256(args.out.read_bytes()).hexdigest()
        args.out.with_suffix(args.out.suffix + '.sha256').write_text(f'{checksum}  {args.out.name}\n')
        print(f'Packed {args.out}')


if __name__ == '__main__':
    main()
