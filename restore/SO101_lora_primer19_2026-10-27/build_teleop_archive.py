#!/usr/bin/env python3
"""Build a portable, checksummed teleop and dataset-recording bundle."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = 'SO101_teleop_dataset_2026-10-23'
FILES = (
    'teleop.py', 'teleop_loader.py', 'record_dataset.py', 'dataset_capture.py',
    'dataset_tools.py', 'prepare_dataset.py', 'validate_dataset.py', 'preview_dataset.py',
    'make_training_config.py', 'check_training_stack.py', 'check_adapter.py',
    'configure_server_cameras.py', 'so101_camera.py', 'so101_arm.py',
    'serial_protocol.py', 'verified_telemetry.py', 'disable_torque.py',
    'policy_client.py', 'vlash_transport.py', 'run_policy.py', 'camera_log.py',
    'so101_follower_calibration.json', 'requirements.txt', 'requirements-dataset.txt',
    'requirements-policy.txt', 'DATASET_RU.md', 'test_dataset_pipeline.py', 'fake_bus.py',
    'build_teleop_archive.py', 'robot_setup/robot_config.json', 'robot_setup/99-so101.rules',
    'robot_setup/so101_kinematics.urdf', 'robot_setup/handeye_transform.json',
    'camera/calibration/camera_intrinsics.json', 'camera/calibration/camera_intrinsics.npz',
    'camera/calibration/rectified_view.json', 'proto/vlash_transport.proto',
)

VERIFY_SOURCE = '''#!/usr/bin/env python3
"""Verify the archive's files using only the standard library; no hardware."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
count = 0
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SystemExit('Missing/invalid file: ' + name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Checksum mismatch: ' + name)
    if path.suffix == '.json':
        json.loads(path.read_text())
    if path.suffix == '.py':
        compile(path.read_text(), str(path), 'exec')
    count += 1
print(f'OK: {count} files, SHA256, JSON and Python syntax. No hardware accessed.')
'''

README = '''# SO-101: телеоперация и запись датасета на другом компьютере

Начните с этого файла. Полная инструкция записи и подготовки обучения —
`DATASET_RU.md`. Нужен Python 3.10–3.12. Комплект предназначен для тех же
физических leader/follower; их калибровки уже включены. Повторно назначать
ID, нулевые смещения или записывать EEPROM для переноса не нужно.

## Установка на Ubuntu/Debian

Распакуйте ZIP целиком, зайдите в его папку. Если системных компонентов нет:

```bash
sudo apt install python3-venv python3-pip ffmpeg libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dataset.txt
python verify_bundle.py
bash install_usb_access.sh
```

Последняя команда устанавливает udev-права для двух ваших контроллеров.
Переподключите USB после установки правил. Для терминального сеанса без окон
при записи используйте `--no-preview`.

## Установка на macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dataset.txt
python verify_bundle.py
```

Разрешите доступ к камере приложению терминала. Linux-скрипт установки USB
на macOS запускать не нужно. Если в сети корпоративный TLS-инспектор,
можно использовать `uv pip install --system-certs --python .venv/bin/python ...`.

## Телеоперация без камер

Сначала проверка без включения моторов:

```bash
python teleop.py --resume-holding
```

После проверки, с поддержанной рукой и свободной рабочей областью:

```bash
python teleop.py --execute --supported --full-range --resume-holding --seconds 60 --speed 400
```

Leader определяется по серийному номеру `5B79016829`, follower — `5B8E114736`.
На другой ОС номера портов могут отличаться. Если автоопределение не сработает,
добавьте `--leader-port <порт-leader> --follower-port <порт-follower>`.
Нельзя одновременно запускать два процесса, использующих одни и те же порты.

## Запись демонстраций двумя камерами

```bash
python run_policy.py --probe
```

Проверьте снимки в `logs/cameras/`. На исходном компьютере общий вид — камера 1,
камера захвата — 0; на новом компьютере индексы нужно проверить заново.
Запись (пример задачи; замените инструкцию своей):

```bash
python record_dataset.py --dataset datasets/box_raw_v1 \\
  --task "Pick up the box." --base-camera 1 --wrist-camera 0 \\
  --execute --supported --resume-holding --episodes 10
```

После `POSE ALIGNED`: `s` — начать, `k` — сохранить успех,
`f` — сохранить неудачу, `d` — удалить, `q` — завершить сеанс.
Исходные PNG, измерения, цели follower, временные метки и телеметрия
сохраняются в выбранную папку. Новое обучение автоматически не запускается.

Сборка и проверка LeRobot-датасета:

```bash
python prepare_dataset.py datasets/box_raw_v1 \\
  --output datasets/box_lerobot_v1 --repo-id marks/so101-box-v1
python validate_dataset.py datasets/box_lerobot_v1
python preview_dataset.py datasets/box_lerobot_v1 --episode 0
```

Детали эпизодов, две камеры, частота 10 Гц, перенос на сервер и конфигурации
LoRA описаны в `DATASET_RU.md`. Для записи не нужны GPU, сервер или PyTorch.
Файлы клиентского inference приложены для `--probe`; для обычного `run_policy.py`
дополнительно нужны зависимости `requirements-policy.txt` и настроенный сервер.

## Остановка

Ctrl+C, `q` или таймер останавливают телеоп, оставляя follower в удержании.
Поддержите руку и предмет, завершите процесс, затем при необходимости:

```bash
python disable_torque.py --supported
```

## Проверка без оборудования

```bash
python -m unittest -v test_dataset_pipeline
```

ZIP не содержит `.venv`, весов модели, SSH-ключей, текущих логов или записанных
датасетов. Зависимости устанавливаются на новом компьютере. Проверки переноса
выполнены распаковкой в отдельную папку и программными тестами; физический
запуск на втором компьютере не выполнялся.
'''


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Archive already exists: {output}; choose another --output')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='so101-archive-') as temporary:
        stage = Path(temporary) / NAME
        stage.mkdir()
        for name in FILES:
            source, destination = ROOT / name, stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        exported = ROOT / 'SO101_teleop_export_2026-10-23'
        for source in exported.rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc':
                continue
            if source.name in ('.DS_Store', 'teleop_last_run.json'):
                continue
            destination = stage / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for name in ('install_usb_access.sh', '99-so101-teleop.rules'):
            shutil.copy2(exported / name, stage / name)
        (stage / 'README_RU.md').write_text(README)
        (stage / 'DATASET_RU.md').write_text(
            (stage / 'DATASET_RU.md').read_text().replace('cd /Users/m.shkut/Documents/SO101', f'cd {NAME}'))
        (stage / 'verify_bundle.py').write_text(VERIFY_SOURCE)
        versions = {}
        for package in ('numpy', 'scipy', 'opencv-python', 'pyserial', 'pyarrow', 'imageio-ffmpeg'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        (stage / 'bundle_info.json').write_text(json.dumps({
            'created': datetime.datetime.now().astimezone().isoformat(),
            'purpose': 'Calibrated SO101 teleop and two-camera task dataset recording',
            'source_environment_packages': versions, 'hardware_recalibration_required': False,
            'scope': 'Same physical arms/controllers; recalibrate if motors/mechanics change',
        }, ensure_ascii=False, indent=2) + '\n')
        entries = []
        for file in sorted(stage.rglob('*')):
            if file.is_file():
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                entries.append(f'{digest}  {file.relative_to(stage).as_posix()}\n')
        (stage / 'SHA256SUMS').write_text(''.join(entries))
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for file in sorted(stage.rglob('*')):
                if file.is_file():
                    archive.write(file, file.relative_to(stage.parent))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(f'Archive: {output}\nFiles: {len(entries) + 1}\nSize: {output.stat().st_size:,} bytes\nSHA256: {digest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / f'{NAME}.zip')
    build(parser.parse_args().output)
