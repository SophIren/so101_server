"""Timestamped camera snapshots and the exact images submitted to the policy."""
import datetime
import json
from pathlib import Path
import time


class CameraLog:
    def __init__(self, directory, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.index = 0
        self.last_observation = None
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata or {}, ensure_ascii=False, indent=2) + '\n')
        print('camera log:', self.directory, flush=True)

    def save(self, frames, stage, step=None, state=None):
        import cv2

        record = {'index': self.index, 'stage': stage, 'step': step,
                  'timestamp': datetime.datetime.now().astimezone().isoformat(),
                  'elapsed_seconds': time.monotonic() - self.started, 'files': {}}
        if state is not None:
            record['state'] = state
        # Separate directories keep the same camera names across snapshots.
        folder = self.directory / f'{self.index:06d}_{stage}'
        folder.mkdir()
        for name, frame in frames.items():
            if frame is None:
                continue
            filename = folder / f'{name}.png'
            if not cv2.imwrite(str(filename), frame):
                raise OSError(f'Could not save camera frame: {filename}')
            record['files'][name] = str(filename.relative_to(self.directory))
        with (self.directory / 'frames.jsonl').open('a') as manifest:
            manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.index += 1
        if stage == 'observation':
            self.last_observation = record['index']
        return record['index']

    def observation(self, observation, step=None):
        frames = {key.removeprefix('observation.images.'): value
                  for key, value in observation.items() if key.startswith('observation.images.')}
        return self.save(frames, 'observation', step,
                         state=observation['observation.state'].tolist())
