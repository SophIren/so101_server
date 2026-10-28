#!/usr/bin/env python3
"""Review synchronized base/wrist video and the state/action labels of an episode."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--episode', type=int, default=0)
    args = parser.parse_args()
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    chunk = args.episode // info['chunks_size']
    data_path = args.dataset / info['data_path'].format(episode_chunk=chunk, episode_index=args.episode)
    data = pq.read_table(data_path).to_pydict()
    keys = [key for key, feature in info['features'].items() if feature['dtype'] == 'video']
    videos = [cv2.VideoCapture(str(args.dataset / info['video_path'].format(
        episode_chunk=chunk, video_key=key, episode_index=args.episode))) for key in keys]
    try:
        for index in range(len(data['timestamp'])):
            started = time.monotonic()
            frames = []
            for key, video in zip(keys, videos):
                ok, frame = video.read()
                if not ok:
                    raise RuntimeError('Video ended before the action/state labels')
                cv2.putText(frame, key.split('.')[-1], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                frames.append(frame)
            canvas = np.concatenate(frames, axis=1)
            footer = np.zeros((80, canvas.shape[1], 3), np.uint8)
            for row, text in enumerate((f'Frame {index}, t={data["timestamp"][index]:.2f}s',
                                       'state: ' + str(np.round(data['observation.state'][index], 1)),
                                       'action: ' + str(np.round(data['action'][index], 1)))):
                cv2.putText(footer, text, (10, 20 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
            cv2.imshow('Dataset review: q=quit, space=pause', np.concatenate((canvas, footer), axis=0))
            key = cv2.waitKey(max(1, int(1000 * (1 / info['fps'] - (time.monotonic() - started))))) & 255
            if key == ord(' '):
                key = cv2.waitKey(0) & 255
            if key in (ord('q'), 27):
                break
    finally:
        for video in videos:
            video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
