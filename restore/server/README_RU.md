# Восстановленный сервер SO101 / pi0.5

Новая ВМ: `ssh -l m.shkut proj.support-concept-car.mlc`.
Рабочий каталог: `/opt/so101`. Локальный клиент сохраняет прежний gRPC-протокол.
Сервер работает с одним клиентом одновременно: протокол VLASH не содержит ID запросов.

## Установка с нуля

Скопировать эту папку `server` на Linux-машину с NVIDIA GPU и выполнить:

```bash
bash server/bootstrap.sh
export HF_HOME=/opt/so101/cache/huggingface
/opt/so101/env/bin/hf auth login
/opt/so101/env/bin/python server/download_assets.py --root /opt/so101
```

Вход HF нужен для официального токенизатора `google/paligemma-3b-pt-224`;
аккаунт должен иметь доступ и принять условия Google на странице модели.
Не вставлять токены в код, конфиги или переписку. Если есть сохранённый токенизатор,
его можно скопировать в `/opt/so101/models/paligemma_tokenizer` и проверить
`AutoTokenizer.from_pretrained` с `local_files_only=True`.

Установщик фиксирует исходники и основные зависимости. Конкретный результат
установки записывается в `/opt/so101/logs/installed-requirements.txt`.
Для другого расположения задать `SO101_ROOT=/путь bash server/bootstrap.sh`;
остальным командам ниже передавать соответствующий `--root`.

Веса: `dongyoonkim/so101-pi05-base`, ревизия
`56325bafcca9ec3155b8bfdf655be7df17e7f09f`. Скрипт загрузки проверяет структуру,
число параметров, конфиг и полноту safetensors авторским проверяющим скриптом.

Результаты выполненных проверок находятся в `verified/verification.json`.
Там отдельно отмечено, проверялась ли реальная модель или только транспорт.
27 октября 2026 года полный путь локальный клиент → SSH → реальная модель → actions
прошёл три запроса с ответом `(32, 6)` без NaN/Inf. На синтетических наблюдениях
первый запрос занял 40,4 с, последующие — 5,7 и 4,6 с, включая передачу данных.

## Запуск на ВМ

Если пакет скопирован как при восстановлении в `/opt/so101/restore`:

```bash
python3 /opt/so101/restore/server/manage.py start
python3 /opt/so101/restore/server/manage.py status
```

`start` печатает путь к логу. Готовность модели подтверждается строкой
`Server started on port 50051` и успешным RPC `Ready`.
Процесс продолжает работать после закрытия SSH. Для остановки:

```bash
python3 /opt/so101/restore/server/manage.py stop
```

Сервер слушает только `127.0.0.1:50051`. Транспорт использует pickle;
его следует оставлять за SSH и подключать только доверенный локальный клиент.

## Подключение с локального компьютера

Туннель в отдельном терминале:

```bash
ssh -N -L 50051:127.0.0.1:50051 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -l m.shkut proj.support-concept-car.mlc
```

В локальном каталоге SO101:

```bash
.venv/bin/python run_policy.py --check-server --timeout 180
.venv/bin/python check_remote_policy.py --timeout 180 --requests 2
```

Вторая команда проверяет **настоящую модель**, отправляя два синтетических изображения
и нулевой state. Она не открывает камеры и serial и не двигает руку. Ожидаемый ответ:
`(32, 6)`, все значения конечны. Синтетические actions не предназначены для исполнения.

Когда робот и камеры подключены, запрос с реальными наблюдениями без движения:

```bash
.venv/bin/python run_policy.py --prompt "take the box" \
  --base-camera 1 --wrist-camera 0 --timeout 180
```

На текущем локальном компьютере base был камерой 1, wrist — 0; на другом их нужно
проверить через `run_policy.py --probe`. Для движения сохраняется прежний явный
режим `--execute --supported` и существующие ограничения локального контроллера.

## Две камеры

Передаются только `observation.images.base_0_rgb` и
`observation.images.left_wrist_0_rgb`. В конфиге весов остаются три слота;
сервер переопределяет `policy.empty_cameras: 1`. Отсутствующий правый wrist
получает маску false. Дублировать base/wrist в третий слот не нужно.

## Последующее дообучение

Установлены авторские патчи `train.py`, `modeling_pi05.py`, `run.py` и LeRobot
`video_utils.py`. В частности, LoRA-чекпоинт сохраняет `normalize_buffers.pt`.
Скрипты локального пайплайна находятся в `/opt/so101/restore`.
Полный процесс записи и подготовки датасета описан в локальном `DATASET_RU.md`.
При генерации конфига заменить старые серверные пути:

```bash
.venv/bin/python make_training_config.py datasets/box_lerobot \
  --output box_lora.json \
  --base /opt/so101/models/so101-pi05-base \
  --server-dataset /постоянный/диск/datasets/box_lerobot \
  --checkpoints /постоянный/диск/checkpoints/box_lora
```

Перед запуском проверить стек и датасет на ВМ:

```bash
export PATH=/opt/so101/env/bin:$PATH
export LD_LIBRARY_PATH=/opt/so101/env/lib:${LD_LIBRARY_PATH:-}
export VLASH_PALIGEMMA_PATH=/opt/so101/models/paligemma_tokenizer
export HF_HOME=/opt/so101/cache/huggingface
python /opt/so101/restore/check_training_stack.py /путь/box_lora.json
```

Обучение не запускается автоматически. Сначала провести 20-шаговый smoke run
с отдельным output_dir, затем проверить сохранённый адаптер через `check_adapter.py`.
Для обучения остановить сервер, чтобы освободить GPU.

Для обслуживания дообученной модели скопировать серверный JSON-конфиг в новый файл
и заменить `policy.path` на каталог `.../checkpoints/<step>/pretrained_model`.
Затем `manage.py stop` и `manage.py start --config /путь/adapter_server.json`.
Загрузчик возьмёт базу из `train_config.json`, загрузит адаптер и статистики;
без них запуск завершится ошибкой. Если базу перенесли, `SO101_BASE_CHECKPOINT`
может указать новое место **тех же самых базовых весов**.

## Хранение и повторное восстановление

На этой ВМ `/qwarium/home` — постоянный NFS с лимитом 8 ГБ.
`/opt/so101` находится на корневом overlay: места достаточно, но пересоздание ВМ
может удалить его. Датасеты и результаты обучения нужно хранить на постоянном
диске/в объектном хранилище. В резервную копию кода веса и HF-токен не входят;
веса можно скачать заново с зафиксированной ревизии.
Резервный ZIP исходников также сохраняется в постоянном домашнем каталоге
`/qwarium/home/m.shkut/SO101_server_restore_2026-10-27.zip`.

Источники: [пакет чекпоинта](https://github.com/jinnymo/so101-pi05-base),
[обучение LoRA](https://github.com/jinnymo/so101-pi05-base/blob/main/docs/07-lora-finetuning.md),
[VLASH](https://github.com/mit-han-lab/vlash). Ревизии и изменения: `NOTICE`.
