#!/usr/bin/env python3
"""Apply the released patches and backport the client's exact gRPC protocol."""
import argparse
import shutil
from pathlib import Path


def replace_once(path, old, new):
    source = path.read_text()
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError(f'Expected exactly one patch target in {path}: {old[:80]}')
    path.write_text(source.replace(old, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    package = Path(__file__).resolve().parent
    release = root / 'so101-pi05-base/training-docker/patched'
    vlash = root / 'vlash'
    for name, target in (
            ('train.py', 'vlash/train.py'), ('run.py', 'vlash/run.py'),
            ('modeling_pi05.py', 'vlash/policies/pi05/modeling_pi05.py')):
        shutil.copy2(release / name, vlash / target)
    from enable_training_resume import patch as enable_resume
    enable_resume(vlash / 'vlash/train.py')
    for source in (package / 'runtime').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:
            target = vlash / source.relative_to(package / 'runtime')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    config = vlash / 'vlash/configs/__init__.py'
    if 'from vlash.configs.serve_config import ServerConfig' not in config.read_text():
        with config.open('a') as out:
            out.write('\nfrom vlash.configs.serve_config import ServerConfig\n')
    cli = vlash / 'vlash/cli.py'
    replace_once(cli, '    elif command == "run":',
                 '    elif command == "serve":\n'
                 '        from vlash.serve import serve_from_config\n'
                 '        if len(sys.argv) < 3:\n'
                 '            raise SystemExit("Usage: vlash serve <config.yaml> [options]")\n'
                 '        sys.argv = [sys.argv[0], f"--config_path={sys.argv[2]}", *sys.argv[3:]]\n'
                 '        serve_from_config()\n'
                 '    elif command == "run":')
    print('Applied author training/model patches and remote-inference backport:', vlash)


if __name__ == '__main__':
    main()
