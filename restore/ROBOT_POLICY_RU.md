# Управление SO-101 через VLASH (pi05)

Для записи собственных демонстраций двумя камерами, подготовки LeRobot-датасета
и конфигурации дообучения используйте `DATASET_RU.md` и `record_dataset.py`.

`run_policy.py` отправляет локальные состояния суставов и кадры на сервер,
получает action chunk и при `--execute --supported` передаёт действия локальному
SO-101 через последовательный порт.

## Протокол

Поддерживается `transport.RemoteInference` из ветки VLASH `dev/remote_inference`:

1. `Ready(Empty)` проверяет готовность сервиса.
2. `LaunchInference(stream Observation)` получает pickle со словарём CPU PyTorch
   tensors. Состояние имеет форму `[1,6]`, изображения — RGB float32
   `[1,3,H,W]` в `[0,1]`, текст — `task: [prompt]`.
3. `GetActionChunk(Empty)` ждёт результат и возвращает pickle с actions `[N,6]`.

Кадры OpenCV преобразуются из BGR в RGB. Observation передаётся частями по
2 МиБ, включая корректный `TRANSFER_END` для последней или единственной части.
Actions с CUDA-сервера десериализуются на CPU, поэтому CUDA на Mac не нужна.
Сервер использует pickle: подключайтесь к своему доверенному серверу.

Схема встроена в `vlash_transport.py` и сохранена в `proto/vlash_transport.proto`.
Если reflection отвечает `UNIMPLEMENTED`, клиент автоматически использует эту
схему. Передавать `--proto` или включать reflection на сервере не требуется.
Также сохранена поддержка явно переданного `.proto`, descriptor set и старого
тестового unary RPC `Infer` с reflection.

Реализация сверена с [исходниками VLASH](https://github.com/mit-han-lab/vlash/tree/093d70b1fecff37b868fd124a6ae8b5691df2a39/vlash/transport)
и [сервером](https://github.com/mit-han-lab/vlash/blob/093d70b1fecff37b868fd124a6ae8b5691df2a39/vlash/serve.py).

## Установка

```bash
cd ~/Documents/SO101
uv pip install --system-certs --python .venv/bin/python \
    -r requirements.txt -r requirements-policy.txt
```

PyTorch нужен для формата данных VLASH; ставить LeRobot или VLASH на Mac не нужно.
`grpcio-tools` используется только при явном `--proto` и в тестах.

## Сервер и туннель

Восстановление новой ВМ, зависимости, хранение и дальнейшее обслуживание LoRA
описаны в [server/README_RU.md](server/README_RU.md). На новой ВМ:

```bash
python3 /opt/so101/restore/server/manage.py start
```

Дождитесь `Server started on port 50051`: строки `Loading policy` ещё не означают
готовность сервера. На Mac в отдельном терминале оставьте работающий туннель:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 50051:127.0.0.1:50051 -l m.shkut proj.support-concept-car.mlc
```

Проверка без открытия камер и последовательного порта:

```bash
.venv/bin/python run_policy.py --check-server
.venv/bin/python run_policy.py --dump-descriptor
.venv/bin/python check_remote_policy.py --timeout 180
```

Последняя команда отправляет синтетические наблюдения в реальную модель и проверяет
конечные actions `(32, 6)` без открытия камер, serial и без движения робота.

Первая команда должна напечатать `VLASH Ready OK at 127.0.0.1:50051`.
`--timeout 120` увеличивает срок ожидания каждого RPC, если первый inference
занимает больше стандартных 60 секунд. Ошибки сервера выводятся вместе с именем
RPC и подробностями gRPC.

## Камеры и пробный запрос

Разрешите камеру приложению терминала в настройках macOS, затем найдите индексы:

```bash
.venv/bin/python run_policy.py --probe
```

Кадры сохраняются в `logs/cameras/`. Выберите камеру сцены и камеру на руке:

```bash
.venv/bin/python run_policy.py --base-camera 0 --wrist-camera 1 \
    --prompt "take the box" --save-frames logs/frames
```

Без `--execute` программа только читает положения, отправляет запрос и печатает
форму чанка и первые/последние actions. Цели моторам не отправляются.
По умолчанию клиент отправляет только две реальные камеры: `base_0_rgb` и
`left_wrist_0_rgb`. Третий слот `right_wrist_0_rgb` не передаётся; на сервере
нужно `policy.empty_cameras: 1`, чтобы модель добавила заглушку с маской False.
Три слота в конфигурации весов сохраняются — это схема модели, а не число
передаваемых изображений. Не удаляйте третий слот из config.json чекпоинта.
Если wrist-камера не открылась, обычный запуск останавливается, чтобы исключить
незаметную подмену изображения. Для одной камеры используйте `--no-wrist` и
сервер с `policy.empty_cameras: 2`.
Старые подстановки доступны явно через `--fill-right-wrist base`, `wrist` или
`black`; это режим совместимости, а не корректное маскирование камеры.
Если индексы камер перепутаны, укажите `--base-camera 1 --wrist-camera 0`.

## Выполнение actions

Во всех обычных запусках (также без `--execute`) автоматически сохраняются
кадры в `logs/policy_<дата>_<время>_<микросекунды>_frames/`. Путь печатается
как `camera log:`. Внутри:

- `000000_observation/` и последующие `*_observation/` — PNG переданных
  изображений (по умолчанию двух реальных камер). Эти кадры
  сохраняются **до запроса**, поэтому остаются и при ошибке inference.
- `*_execution/` — снимки доступных физических камер после каждого шага движения.
- `*_final/` — финальный снимок при остановке сеанса.
- `frames.jsonl` — время, номер шага, состояние и пути к снимкам;
  `metadata.json` — prompt, выбор камер и режим запуска.

JSON-лог действий содержит путь `camera_log`, а каждый шаг — номера снимков
`observation_frame` (вход модели) и `camera_frame` (после действия).
Это снимки по шагам, а не непрерывное видео: во время ожидания inference
промежуточные кадры не записываются. Сохранение PNG добавляет время к шагу.
`--no-camera-log` отключает автоматическую запись. Старый `--save-frames DIR`
дополнительно сохраняет первые кадры под именами `base.png` и `wrist.png`.

Рука должна находиться на устойчивой опоре, рабочая зона — быть свободной.
Короткий запуск с ограничением приращения каждого сустава:

```bash
.venv/bin/python run_policy.py --base-camera 0 --wrist-camera 1 \
    --prompt "take the box" --execute --supported --steps 20 --max-ticks 20
```

Полный первый чанк — без `--steps`. Повторное получение observations и actions:

```bash
.venv/bin/python run_policy.py --base-camera 0 --wrist-camera 1 \
    --prompt "take the box" --execute --supported --replan \
    --max-ticks 20 --max-seconds 30
```

Первый напечатанный чанк используется для движения. При `--replan` следующий
запрос отправляется после выполнения текущего чанка, со свежим измерением
положения и новыми кадрами. Это синхронное управление: во время ожидания сети
рука удерживает последнюю цель. `--hz` задаёт верхнюю частоту цикла; фактическая
частота зависит от ожидания достижения целей и чтения телеметрии.

Сервер сам ограничивает длину чанка через `policy.n_action_steps` (в вашем логе
32, хотя `chunk_size` модели равен 50). Actions должны иметь непустую форму
`[N,6]` и конечные значения. Суставы следуют порядку `shoulder_pan`,
`shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
Пять суставов нормализуются по калиброванному диапазону в `[-100,100]`,
гриппер — в `[0,100]`; цели ограничиваются локальной калибровкой.

`--port-name /dev/tty.usbmodem...` переопределяет автоматический выбор порта.
Лимит `--steps` действует на весь сеанс. `--max-seconds` проверяется перед
каждым действием; уже начатое ожидание inference ограничивается `--timeout`,
ожидание достижения отдельной цели — логикой `Arm.hold_and_check`.

При остановке или ошибке сохраняется лог `logs/policy_*.json`, рука остаётся
с включённым удержанием. Поддержите её и затем отключите torque:

```bash
.venv/bin/python disable_torque.py --supported
```

## Проверки без железа

```bash
.venv/bin/python selftest.py
.venv/bin/python -m unittest -v test_policy_transport test_camera_log
```

Новые тесты проверяют сервер без reflection, передачу большого observation,
RGB/нормализацию/формы tensors, загрузку CUDA pickle на CPU, повторные запросы,
ошибки сервера и исполнение чанков на FakeBus с ограничениями шагов и времени.
Тесты открывают только локальный TCP-порт и не обращаются к камерам или роботу.
Отдельный mock с настоящим streaming-контрактом:

```bash
.venv/bin/python mock_vlash_server.py --port 50052
.venv/bin/python run_policy.py --port 50052 --check-server
```
