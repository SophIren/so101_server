# LoRA: PriMeR19/SO-101-teleop

ВМ: `ssh -l m.shkut proj.support-concept-car.mlc`.
Постоянный каталог запуска: `/qwarium/home/m.shkut/so101_runs/primer19_bag_lora_20261027`.
Конфиг основного запуска: `full/lora.json`; вывод trainer: `full/train.log`.
Обучение запускается отдельным процессом и продолжает работать после закрытия SSH.

## Данные

Источник: https://huggingface.co/datasets/PriMeR19/SO-101-teleop
Зафиксированная ревизия: `b83962819ea1b10eb3c690b04d2094058fbbf244`.
Вложенный датасет: `so101_teleop_bag`, 120 эпизодов, 106770 кадров, 30 fps.
Единственная инструкция: **`pick_place_bag`**. Её же нужно передавать модели при проверках.

Преобразование: `server/prepare_primer19.py`. Сохранена исходная привязка row ↔ video frame.
RGB front → base_0_rgb, wrist → left_wrist_0_rgb. Depth не используется.
Три слота базы сохранены в policy config, отсутствующий right wrist маскируется.
State пересчитан из follower raw encoders через сохранённую калибровку.
Целевая action взята из `action.goal_sent`; первые пять суставов переведены из градусов
в калиброванный диапазон -100…100, gripper сохранён в 0…100.
Это соответствует текущим `policy_client.py` и локальному контроллеру.

Детерминированное разделение по целым эпизодам: 108 train / 12 отложены для validation,
seed 1000. Trainer получает явный список train-эпизодов. Validation loss автоматически
этим trainer не вычисляется. Исходные записи и схема преобразования сохранены для аудита.

LeRobot 0.4.1 читает все локальные parquet-файлы, игнорируя фильтр episodes при
загрузке таблиц. Поэтому `isolate_training_splits.py` создаёт физически отдельные
каталоги и перенумеровывает индексы. Реальный DataLoader проверен на отсутствие
validation: 108 эпизодов и 96192 кадра. Holdout: 12 эпизодов и 10578 кадров.

Преобразованный полный датасет: `/opt/so101/datasets/primer19_bag_v21`.
Фактический training root: `/opt/so101/datasets/primer19_bag_train_v21`.
Отложенная выборка: `/opt/so101/datasets/primer19_bag_validation_v21`.
Публичные исходники и видео можно скачать повторно с указанной ревизии.
Пересоздание ВМ удалит `/opt`, поэтому чекпоинты сохраняются в постоянный NFS.

## Параметры и чекпоинты

База: `/opt/so101/models/so101-pi05-base`.
LoRA r=16, alpha=16; bf16; batch 5 × accumulation 2 = effective batch 10;
20 000 optimizer steps; lr 5e-5; warmup 1000; delay augmentation 0 для текущего
синхронного клиента. Фактическая частота сохранения записана в `full/lora.json`.
20 000 шагов соответствуют примерно 2,08 прохода по train. Сохранение каждые
2500 шагов даёт 8 адаптеров и оставляет место для состояния optimizer на NFS 8 ГиБ.

Каждый адаптер содержит `normalize_buffers.pt`. Последний чекпоинт также содержит
optimizer/scheduler/step/RNG для продолжения обучения. У предыдущих чекпоинтов
optimizer state удаляется для экономии NFS, сами адаптеры сохраняются.
Патч включается переменной `VLASH_SAVE_LOCAL_RESUME=1` в `train_job.py`.
Проверено до основного запуска: 20 optimizer steps, сохранение адаптера и статистик,
resume на шаг 21 с optimizer state, затем загрузка этого адаптера поверх базы и
inference по реальному RGB/state с конечным ответом `(1, 32, 6)`. Робот не использовался.

## Управление на ВМ

```bash
python3 /opt/so101/restore/server/train_job.py status \
  --run-dir /qwarium/home/m.shkut/so101_runs/primer19_bag_lora_20261027/full

tail -f /qwarium/home/m.shkut/so101_runs/primer19_bag_lora_20261027/full/train.log
```

Для остановки заменить `status` на `stop`. Это остановка процесса;
продолжение начнётся с последнего полностью сохранённого чекпоинта.
Для возобновления после остановки на той же ВМ:

```bash
python3 /opt/so101/restore/server/train_job.py start \
  --run-dir /qwarium/home/m.shkut/so101_runs/primer19_bag_lora_20261027/full \
  --config /qwarium/home/m.shkut/so101_runs/primer19_bag_lora_20261027/full/lora.json
```

На пересозданной ВМ сначала восстановить среду, базовые веса, токенизатор
и преобразованный датасет по тем же путям. Код запуска и конфиги находятся
в резервной копии исходников.

## Инференс после обучения

Inference-сервер остановлен на время обучения. Конфиг `full/serve_adapter.json`
указывает на последний адаптер, базовые веса и нормализацию загрузчик объединяет
автоматически. После завершения/остановки обучения:

```bash
python3 /opt/so101/restore/server/manage.py start \
  --config /qwarium/home/m.shkut/so101_runs/primer19_bag_lora_20261027/full/serve_adapter.json
```

Обычный SSH-туннель остаётся тем же. Сначала проверить модель без движения;
для реального запуска передавать `--prompt pick_place_bag`, правильные индексы
камер и `--hz 30`, соответствующие частоте демонстраций. Выбор лучшего чекпоинта
потребует отдельной проверки задачи; `checkpoints/last` означает последний сохранённый.
