#!/usr/bin/env python3
"""Back up server/client sources, patches and config assets, excluding runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SKIP = {'.git', '__pycache__', '.DS_Store', '.idea', '.venv', '.cache'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Archive already exists; choose a new output name')
    files = set()
    for pattern in ('*.py', '*.md', 'requirements*.txt', '*calibration.json'):
        files.update(ROOT.glob(pattern))
    for folder in ('server', 'proto', 'robot_setup', 'camera/calibration',
                   'SO101_teleop_export_2026-10-23'):
        for path in (ROOT / folder).rglob('*'):
            relative = path.relative_to(ROOT)
            if (path.is_file() and not set(relative.parts) & SKIP
                    and not path.name.startswith('._') and path.name != 'teleop_last_run.json'):
                files.add(path)
    payloads = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(files)}
    sums = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    payloads['SOURCE_SHA256.json'] = (json.dumps(sums, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.output.stem
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(f'{prefix}/{name}', data)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC failure')
        for name, digest in sums.items():
            if hashlib.sha256(archive.read(f'{prefix}/{name}')).hexdigest() != digest:
                raise RuntimeError(f'Archive SHA256 failure: {name}')
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(
        f'{digest}  {args.output.name}\n')
    print(f'{args.output.resolve()}: {len(payloads)} files, {args.output.stat().st_size} bytes')
    print('CRC and source SHA256: PASS')


if __name__ == '__main__':
    main()
