"""Verify the exported bundle without accessing hardware."""
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    count=0
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SystemExit('Checksum mismatch: '+name)
        if path.suffix=='.json':json.loads(path.read_text())
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
        count+=1
    print(f'OK: {count} files; checksums, JSON and Python syntax verified. No hardware accessed.')

if __name__=='__main__':main()
