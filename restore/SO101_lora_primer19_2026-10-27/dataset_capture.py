"""Camera acquisition and lossless episode journals for teleoperation datasets."""
from collections import deque
import datetime
import json
from pathlib import Path
import queue
import shutil
import threading
import time

import cv2


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class CameraStream:
    def __init__(self, camera, name):
        self.camera, self.name = camera, name
        self.frames = deque(maxlen=12)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name=f'camera-{name}')
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                frame = self.camera.read()
                captured = time.monotonic()
                with self.lock:
                    self.frames.append((captured, frame.copy()))
        except Exception as error:
            self.error = error

    def snapshot(self, at=None, max_age=.3):
        if self.error:
            raise RuntimeError(f'{self.name} camera failed: {self.error}') from self.error
        at = time.monotonic() if at is None else at
        with self.lock:
            candidates = [(stamp, frame) for stamp, frame in self.frames if stamp <= at]
            if not candidates:
                raise RuntimeError(f'{self.name}: no camera frame available before the observation')
            stamp, frame = candidates[-1]
        if at - stamp > max_age:
            raise RuntimeError(f'{self.name}: stale frame ({at - stamp:.3f}s)')
        return stamp, frame.copy()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.camera.release()
        self.thread.join(timeout=1)


class RawEpisode:
    def __init__(self, dataset, task, max_pending=16):
        name = datetime.datetime.now().strftime('episode_%Y%m%d_%H%M%S_%f')
        self.path = Path(dataset) / 'episodes' / name
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.task = task
        self.count = 0
        self.error = None
        self.closed = False
        self.jobs = queue.Queue(maxsize=max_pending)
        self.info = {'format': 'so101-raw-v1', 'task': task, 'status': 'recording',
                     'started': datetime.datetime.now().astimezone().isoformat()}
        write_json(self.path / 'episode.json', self.info)
        self.worker = threading.Thread(target=self._run, daemon=True, name='episode-writer')
        self.worker.start()

    def append(self, row, frames):
        if self.closed:
            raise RuntimeError('Episode is already closed')
        if self.error:
            raise RuntimeError(f'Camera log writer failed: {self.error}') from self.error
        row = dict(row, sample_index=self.count)
        try:
            self.jobs.put_nowait((row, {k: v.copy() for k, v in frames.items()}))
        except queue.Full as error:
            raise RuntimeError('Dataset disk writer cannot keep up; episode must be discarded') from error
        self.count += 1

    def _run(self):
        journal = None
        try:
            journal = (self.path / 'samples.jsonl').open('a')
        except Exception as error:
            self.error = error
        try:
            while True:
                job = self.jobs.get()
                try:
                    if job is None:
                        return
                    if self.error:
                        continue
                    row, frames = job
                    row['images'] = {}
                    for key, image in frames.items():
                        filename = Path(key) / f"{row['sample_index']:06d}.png"
                        (self.path / filename).parent.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(self.path / filename), image):
                            raise OSError(f'Could not save {filename}')
                        row['images'][key] = str(filename)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    journal.flush()
                except Exception as error:
                    self.error = error
                finally:
                    self.jobs.task_done()
        finally:
            if journal is not None:
                journal.close()

    def finish(self, outcome, error=None):
        if self.closed:
            return
        self.closed = True
        self.jobs.put(None)
        self.worker.join()
        problem = str(self.error or error) if self.error or error else None
        self.info.update(status='complete' if problem is None else 'interrupted',
                         outcome=outcome if problem is None else 'interrupted',
                         samples=self.count, error=problem,
                         finished=datetime.datetime.now().astimezone().isoformat())
        write_json(self.path / 'episode.json', self.info)
        if outcome == 'discard' and problem is None:
            shutil.rmtree(self.path)
        print(f'Episode {self.info["outcome"]}: {self.count} samples, {self.path}', flush=True)
        if self.error:
            raise RuntimeError(f'Dataset writer failed: {self.error}') from self.error
