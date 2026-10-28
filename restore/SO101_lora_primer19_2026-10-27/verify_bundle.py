#!/usr/bin/env python3
"""Check package integrity without dependencies or device access."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 checked=0
 for line in (ROOT/'SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);path=ROOT/name
  if ROOT not in path.resolve().parents:raise RuntimeError('Unsafe manifest path')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Checksum mismatch: '+name)
  checked+=1
 for path in ROOT.rglob('*.py'):compile(path.read_text(),str(path),'exec')
 for path in ROOT.rglob('*.json'):json.loads(path.read_text())
 cfg=json.loads((ROOT/'robot_setup/robot_config.json').read_text())
 for name in [cfg['joint_calibration'],cfg['camera']['intrinsics'],cfg['kinematics']['file'],cfg['handeye']['transform']]:
  if not (ROOT/name).is_file():raise RuntimeError('Missing profile dependency: '+name)
 print(f'OK: {checked} files; checksums, JSON, Python syntax and profile paths valid. No hardware opened.')

if __name__=='__main__':main()
