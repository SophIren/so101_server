#!/usr/bin/env python3
"""Просмотр USB-камеры: python3 camera_viewer.py (выход: Q или Esc)."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAMERA = json.loads((Path(__file__).resolve().parent / 'robot_setup/robot_config.json').read_text())['camera']['device']


def main():
    parser = argparse.ArgumentParser(description='Просмотр камеры, целевая частота 15 fps')
    parser.add_argument('--device', default=DEFAULT_CAMERA)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--check', action='store_true', help='Проверить частоту без окна')
    parser.add_argument('--calibrated', action='store_true',
                        help='1280×720 с исправлением искажений, C переключает исходный кадр')
    args = parser.parse_args()
    calibration = None
    if args.calibrated:
        calibration = json.loads((Path(__file__).resolve().parent /
            'camera/calibration/camera_intrinsics.json').read_text())
        args.width, args.height = calibration['image_size']
        args.fps = min(args.fps, 10)
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error('Частота и размеры должны быть положительными')

    device = int(args.device) if args.device.isdigit() else args.device
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError(f'Не удалось открыть камеру: {args.device}')
        # MJPEG позволяет передавать 720p без ограничения полосы USB сырым видео.
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FPS, args.fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = camera.get(cv2.CAP_PROP_FPS)
        maps = None
        if calibration:
            if [width, height] != calibration['image_size']:
                raise RuntimeError('Фактическое разрешение не совпадает с калибровкой')
            K = np.asarray(calibration['camera_matrix'], dtype=np.float64)
            distortion = np.asarray(calibration['distortion_coefficients'], dtype=np.float64)
            policy = json.loads((Path(__file__).resolve().parent /
                'camera/calibration/rectified_view.json').read_text())
            if policy['image_size'] != [width, height]:
                raise RuntimeError('Размер исправленного вида не совпадает с кадром')
            output_K = np.asarray(policy['camera_matrix'], dtype=np.float64)
            maps = cv2.initUndistortRectifyMap(K, distortion, None, output_K,
                                             (width, height), cv2.CV_32FC1)
        print(f'Камера: {width}x{height}; режим {reported_fps:g} fps; '
              f'целевая частота показа {args.fps:g} fps', flush=True)

        for _ in range(5):
            ok, _ = camera.read()
            if not ok:
                raise RuntimeError('Камера не передаёт кадры')

        window = 'Robot camera - Q / Esc: exit'
        if not args.check:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, width, height)
        interval = 1 / args.fps
        started = time.monotonic()
        sample_started = started
        sample_frames = 0
        total_frames = 0
        measured_fps = 0.0
        deadline = started
        show_corrected = maps is not None
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Получение изображения прервано')
            if maps is not None and show_corrected:
                frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            total_frames += 1
            sample_frames += 1
            now = time.monotonic()
            if now - sample_started >= 1:
                measured_fps = sample_frames / (now - sample_started)
                sample_started, sample_frames = now, 0

            if not args.check:
                mode = 'CORRECTED VIEW' if show_corrected else 'RAW FULL FRAME'
                cv2.putText(frame, f'{measured_fps:.1f} fps | {mode}',
                            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
                cv2.imshow(window, frame)

            deadline += interval
            delay = max(0, deadline - time.monotonic())
            if args.check:
                time.sleep(delay)
                if total_frames >= 60:
                    elapsed = time.monotonic() - started
                    print(f'Измерено: {total_frames / elapsed:.2f} fps '
                          f'({total_frames} кадров за {elapsed:.2f} с)', flush=True)
                    break
            else:
                key = cv2.waitKey(max(1, round(delay * 1000))) & 255
                if key in (27, ord('q'), ord('Q')):
                    break
                if key in (ord('c'), ord('C')) and maps is not None:
                    show_corrected = not show_corrected
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            # Не накапливаем отставание, если камера выдаёт кадры медленнее.
            if time.monotonic() - deadline > interval:
                deadline = time.monotonic()
    finally:
        camera.release()
        if not args.check:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(f'Ошибка: {error}')
