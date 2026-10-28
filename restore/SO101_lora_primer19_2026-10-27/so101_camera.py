"""Camera capture for the SO-101 rig.

The repository was exported from Linux, where cameras are addressed through
/dev/v4l/by-id. On macOS the same camera shows up as a plain AVFoundation index.
Resolution/reject logic is unchanged; only device selection differs.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_config():
    return json.loads((ROOT / 'robot_setup/robot_config.json').read_text())


def open_camera(index, width, height, fps, mjpg=False):
    backend = (cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else
               cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY)
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f'Cannot open camera index {index}')
    if mjpg:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


class Camera:
    """One USB camera; `policy` camera feeds the policy, `wrist` is optional."""

    def __init__(self, index, calibration, width=640, height=480, fps=30, rectify=False):
        self.index = index
        self.calibration = calibration
        self.width, self.height = width, height
        self.camera = open_camera(index, width, height, fps)
        actual = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (width, height):
            print(f'  camera {index}: requested {width}x{height}, got {actual[0]}x{actual[1]}')
        self.maps = None
        if rectify:
            self._build_maps()

    def _build_maps(self):
        cal = self.calibration
        if cal['image_size'] != [self.width, self.height]:
            raise RuntimeError(
                f'Rectification needs {cal["image_size"]}, camera runs '
                f'{self.width}x{self.height}. Calibrate the policy resolution instead.')
        policy = json.loads((ROOT / 'camera/calibration/rectified_view.json').read_text())
        K = np.asarray(cal['camera_matrix'], dtype=np.float64)
        dist = np.asarray(cal['distortion_coefficients'], dtype=np.float64)
        out = np.asarray(policy['camera_matrix'], dtype=np.float64)
        self.maps = cv2.initUndistortRectifyMap(
            K, dist, None, out, (self.width, self.height), cv2.CV_32FC1)

    def read(self):
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError(f'Camera {self.index} stopped delivering frames')
        if self.maps is not None:
            frame = cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        return frame

    def release(self):
        self.camera.release()


def probe_cameras(max_index=4):
    """List indices that deliver a frame. Run before a session to pick devices.

    Returns [(index, width, height, error)]; error is None on success. Opening is
    attempted through AVFoundation first, then through OpenCV's default backend,
    because macOS denies camera access per *application* (TCC), not per device.
    """
    found = []
    for index in range(max_index + 1):
        error = None
        try:
            camera = open_camera(index, 640, 480, 30)
        except Exception as problem:  # noqa: BLE001 - report, do not abort the scan
            found.append((index, 0, 0, str(problem)))
            continue
        try:
            ok = False
            frame = None
            for _ in range(3):
                ok, frame = camera.read()
                if ok:
                    break
            if ok and frame is not None:
                found.append((index, int(frame.shape[1]), int(frame.shape[0]), None))
            else:
                found.append((index, 0, 0, 'opened but delivered no frame'))
        except Exception as problem:  # noqa: BLE001
            found.append((index, 0, 0, str(problem)))
        finally:
            camera.release()
    return found
