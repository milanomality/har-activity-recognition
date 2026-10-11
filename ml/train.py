"""Обучение и экспорт моделей распознавания активности.

Обучает две модели:

* классификатор активности — одномерная остаточная свёрточная сеть по «сырым»
  окнам акселерометра и гироскопа. Внутри сети из них считаются величины,
  не зависящие от того, как повёрнут телефон: модуль ускорения, его
  вертикальная и горизонтальная составляющие, модуль вращения и поворот
  вокруг вертикали;
* классификатор положения телефона — MLP по 16 агрегированным признакам,
  в которых главную роль играют датчики приближения и освещённости.

Результат кладётся прямо в assets приложения: два .tflite и model_meta.json
со статистиками нормировки, списком каналов и списком классов. Приложение
читает метаданные и подстраивается под то, на чём модель реально обучена.

Примеры запуска:
    python ml/train.py --source all
    python ml/train.py --source uci,realworld,own --epochs 30
    python ml/train.py --source all --own-exclude-days 20261009

Источники: uci, realworld, motionsense, shoaib, extrasensory, own; можно
перечислить через запятую. both = uci+own, all = все шесть.

Честная проверка на собственных записях — ml/crossval.py: модель обучается
без записей одного дня и проверяется на них.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

import datasets
import features as feat

# Порядок классов должен совпадать с перечислениями в Kotlin (Labels.kt):
# приложение сопоставляет выходы модели по именам, но порядок id важен
# для раскладки вероятностей. Транспорта нет: приложение распознаёт
# активность самого человека, а не способ передвижения.
ACTIVITY_ORDER = [
    "STILL", "WALKING", "RUNNING",
    "STAIRS_UP", "STAIRS_DOWN", "CYCLING",
]
PLACEMENT_ORDER = ["POCKET", "IN_HAND", "AT_EAR", "ON_TABLE"]

# Каналы модели активности. Магнитометр не берётся, даже если он есть во всех
# источниках: рядом с металлом (велосипед, перила, ключи в кармане) поле
# искажается сильнее, чем меняется от движения, и сеть училась бы на помехах.
ACTIVITY_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"
CACHE_DIR = REPO_ROOT / "ml" / "data" / "cache"


# --------------------------------------------------------------------------
# Модель активности
# --------------------------------------------------------------------------

def _motion_features_layer():
    """Слой, дополняющий окно величинами, не зависящими от поворота телефона.

    Сеть получает сырые оси телефона, но в кармане телефон лежит как угодно:
    экраном к ноге или наружу, верхом вверх или вниз. Ходьба при этом остаётся
    ходьбой, а оси меняются местами и знаками. Поэтому к шести каналам
    добавляются пять, которые от поворота не зависят:

    * модуль ускорения;
    * вертикальная составляющая ускорения — проекция на направление тяжести
      (среднее ускорение за окно). Именно она отличает лестницу от ровной
      ходьбы: при подъёме тело толкается вверх на каждом шаге;
    * горизонтальная составляющая;
    * модуль угловой скорости;
    * вращение вокруг вертикали — повороты корпуса и бедра.

    Считается внутри модели, а не в приложении: так Kotlin-код не меняется,
    и вычисление на телефоне заведомо совпадает с обучением.
    """
    from tensorflow import keras
    from keras import ops

    class MotionFeatures(keras.layers.Layer):
        def call(self, x):
            acc = x[..., 0:3]
            gyro = x[..., 3:6]
            g = ops.mean(acc, axis=1, keepdims=True)
            g_unit = g / (ops.sqrt(ops.sum(g * g, axis=-1, keepdims=True)) + 1e-6)
            lin = acc - g
            vert = ops.sum(lin * g_unit, axis=-1, keepdims=True)
            horiz = ops.sqrt(ops.sum(ops.square(lin - vert * g_unit), axis=-1, keepdims=True) + 1e-6)
            acc_mag = ops.sqrt(ops.sum(acc * acc, axis=-1, keepdims=True) + 1e-6)
            gyro_mag = ops.sqrt(ops.sum(gyro * gyro, axis=-1, keepdims=True) + 1e-6)
            gyro_vert = ops.sum(gyro * g_unit, axis=-1, keepdims=True)
            return ops.concatenate([x, acc_mag, vert, horiz, gyro_mag, gyro_vert], axis=-1)

    return MotionFeatures(name="motion_features")


def build_activity_model(window_size: int, channels: int, classes: int,
                         feature_mean: np.ndarray, feature_var: np.ndarray):
    """Остаточная одномерная CNN по окну сигналов.

    Вход — сырые значения датчиков в единицах Android (м/с², рад/с):
    нормировка встроена в модель, поэтому в model_meta.json среднее 0 и СКО 1,
    и приложение подаёт показания как есть.

    Свёртки вдоль времени сами находят форму шага, толчка и педалирования.
    Остаточные связи позволяют сделать сеть глубже без потери обучаемости,
    четыре блока с понижением разрешения дают рецептивное поле во всё окно.
    Среднее и максимум по времени вместе: среднее описывает ритм, максимум
    ловит редкие резкие события — удар стопы о ступень при спуске.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    def conv_bn(x, filters, kernel):
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    def residual(x, filters, kernel):
        shortcut = x
        if x.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, use_bias=False)(x)
            shortcut = layers.BatchNormalization()(shortcut)
        y = conv_bn(x, filters, kernel)
        y = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        return layers.ReLU()(layers.Add()([shortcut, y]))

    motion_in = layers.Input(shape=(window_size, channels), name="motion")
    x = _motion_features_layer()(motion_in)
    x = layers.Normalization(mean=feature_mean, variance=feature_var, name="normalise")(x)

    # Ширина растёт по мере понижения разрешения: на полных 128 отсчётах
    # свёртки самые дорогие, поэтому там каналов меньше.
    x = conv_bn(x, 48, 7)
    x = residual(x, 48, 5)
    x = layers.MaxPooling1D(2)(x)
    x = residual(x, 96, 5)
    x = layers.MaxPooling1D(2)(x)
    x = residual(x, 128, 3)
    x = layers.MaxPooling1D(2)(x)
    x = residual(x, 192, 3)
    x = layers.SpatialDropout1D(0.1)(x)

    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inputs=motion_in, outputs=out, name="activity_rescnn")


def _rotation_matrices(rng: np.random.Generator, n: int, sigma_deg: float) -> np.ndarray:
    """n случайных поворотов: ось равномерно по сфере, угол ~ N(0, sigma)."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.normal(0.0, np.radians(sigma_deg), size=n)
    k = np.zeros((n, 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3)[None] + s * k + (1 - c) * (k @ k)


# Перевороты телефона на 180°: верхом вниз, экраном к телу, оба сразу.
# Все — собственные повороты (определитель +1), поэтому гироскоп как
# псевдовектор преобразуется той же матрицей, что и ускорение.
_FLIPS = np.array([
    np.diag([1.0, 1.0, 1.0]),
    np.diag([-1.0, -1.0, 1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([1.0, -1.0, -1.0]),
])


def augment_motion(x: np.ndarray, rng: np.random.Generator, p: float = 0.8) -> np.ndarray:
    """Аугментация окон acc+gyro, сохраняющая физику.

    * поворот телефона: небольшой случайный (σ = 20°) и с вероятностью 45 %
      переворот на 180° — так телефон по-разному лежит в кармане;
    * сила движения ±15 %: люди шагают и крутят педали с разной энергией;
      масштабируется только динамическая часть, тяжесть остаётся 9.8 м/с²;
    * лёгкий шум датчиков.

    Магнитометр и прочие каналы здесь не участвуют: модель видит только acc+gyro.
    """
    n = len(x)
    out = x.astype(np.float32, copy=True)
    use = rng.random(n) < p
    if not use.any():
        return out
    xs = out[use]
    m = len(xs)
    flips = _FLIPS[rng.choice(4, size=m, p=[0.55, 0.15, 0.15, 0.15])]
    rot = (_rotation_matrices(rng, m, 20.0) @ flips).astype(np.float32)
    acc = np.einsum("bij,btj->bti", rot, xs[..., 0:3])
    gyro = np.einsum("bij,btj->bti", rot, xs[..., 3:6])
    g = acc.mean(axis=1, keepdims=True)
    acc = g + rng.uniform(0.85, 1.15, size=(m, 1, 1)).astype(np.float32) * (acc - g)
    gyro = gyro * rng.uniform(0.85, 1.15, size=(m, 1, 1)).astype(np.float32)
    acc += rng.normal(0, 0.03, size=acc.shape).astype(np.float32)
    gyro += rng.normal(0, 0.005, size=gyro.shape).astype(np.float32)
    out[use] = np.concatenate([acc, gyro], axis=-1)
    return out


def _batches(x, y, w, batch_size, augment, seed):
    """Пачки для keras.fit с аугментацией на лету и перемешиванием каждую эпоху."""
    from tensorflow import keras

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(x))

        def __len__(self):
            return math.ceil(len(x) / batch_size)

        def __getitem__(self, i):
            idx = np.sort(self.order[i * batch_size:(i + 1) * batch_size])
            xb = x[idx]
            if augment:
                xb = augment_motion(xb, self.rng)
            return xb, y[idx], w[idx]

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(x))

    return Batches()


def balanced_weights(labels: np.ndarray, names: list[str], is_own: np.ndarray | None,
                     own_share: float, own_combo: np.ndarray | None = None,
                     combo_power: float = 0.0) -> np.ndarray:
    """Веса окон: свои записи получают заданную долю, классы уравниваются.

    Собственных записей — тысячи окон против сотен тысяч публичных, но только
    они сняты тем телефоном, на котором работает приложение. Без веса их
    вклад терялся бы, и сеть не училась бы особенностям именно этого датчика
    и этого человека. Доля задаётся до уравнивания классов: затем каждый
    класс получает одинаковый суммарный вес, иначе «покой» и велосипед,
    которых записано больше всего, перетягивали бы решение на себя.

    Внутри своих записей окно сочетания «активность × положение» из n окон
    получает множитель n^(−combo_power): часовая запись покоя в руке и
    десятки минут велосипеда иначе заглушали бы минуту ходьбы с телефоном
    в руке — а это та же активность, что и ходьба в кармане, но другой сигнал.
    """
    d = np.ones(len(labels), dtype=np.float64)
    if is_own is not None and is_own.any() and (~is_own).any() and own_share > 0:
        own_d = np.ones(int(is_own.sum()))
        if own_combo is not None and combo_power > 0:
            combos = own_combo[is_own]
            for c in np.unique(combos):
                m = combos == c
                own_d[m] = m.sum() ** -combo_power
        d[is_own] = own_d / own_d.sum() * own_share / (1 - own_share) * (~is_own).sum()
    w = np.zeros(len(labels), dtype=np.float64)
    present = [n for n in names if (labels == n).any()]
    for n in present:
        mask = labels == n
        w[mask] = d[mask] / d[mask].sum()
    return (w / w.mean()).astype(np.float32)


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def to_tflite(model) -> bytes:
    """Конвертирует модель Keras в TFLite.

    С TensorFlow 2.16 по умолчанию используется Keras 3, для которого
    прямая конвертация из объекта модели работает не во всех версиях.
    Поэтому сначала пробуем прямой путь, а при неудаче экспортируем
    в SavedModel и конвертируем из него.
    """
    import tensorflow as tf

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()
    except Exception as direct_error:  # noqa: BLE001
        print(f"  прямая конвертация не удалась ({type(direct_error).__name__}), "
              "пробуем через SavedModel")
        tmp = Path(tempfile.mkdtemp())
        try:
            export_dir = tmp / "saved"
            if hasattr(model, "export"):
                model.export(str(export_dir))
            else:  # pragma: no cover — старые версии Keras
                model.save(str(export_dir), save_format="tf")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
            return converter.convert()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def encode(labels: np.ndarray, names: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    return np.array([index[label] for label in labels], dtype=np.int64)


def report(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> float:
    """Печатает отчёт по классам и матрицу ошибок, возвращает accuracy."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = float(accuracy_score(y_true, y_pred))
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(names))),
        target_names=names,
        zero_division=0,
    ))

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    width = max(len(n) for n in names) + 1
    print("Матрица ошибок (строки — истина, столбцы — предсказание):")
    print(" " * width + "".join(f"{n[:6]:>8}" for n in names))
    for name, row in zip(names, matrix):
        print(f"{name:<{width}}" + "".join(f"{v:>8}" for v in row))

    # sklearn печатает для класса без тестовых примеров ту же строку нулей,
    # что и для класса, который модель полностью провалила. Это разные вещи,
    # и разница принципиальна: во втором случае модель плоха, в первом —
    # о ней просто ничего не известно.
    missing = [n for n, c in zip(names, matrix.sum(axis=1)) if c == 0]
    if missing:
        print(f"\nВНИМАНИЕ: в тесте нет примеров классов: {', '.join(missing)}.")
        print("Их строки нулей означают «не проверено», а не «не работает».")

    print(f"\nAccuracy: {accuracy:.4f} (на {len(y_true)} окнах)")
    return accuracy


def _split(subjects: np.ndarray, is_own: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Деление по испытуемым; собственные записи целиком идут в обучение.

    Валидация и тест — только публичные люди: на них выбирается эпоха
    и печатается отчёт. Свои записи проверяются отдельно и честнее —
    ml/crossval.py обучает модель без целого дня записей и смотрит на него.
    Отдавать часть своих записей в тест финальной модели значило бы
    лишить её самых ценных данных.
    """
    holder = datasets.Dataset(
        x=np.zeros((len(subjects), 1, 1), dtype=np.float32),
        context=np.zeros((len(subjects), 1), dtype=np.float32),
        y_activity=subjects, y_placement=None, subjects=subjects, channels=[],
    )
    train_idx, val_idx, test_idx = datasets.split_by_subject(
        holder, args.test_fraction, args.val_fraction, args.seed,
    )
    if args.own_in_test:
        return train_idx, val_idx, test_idx
    own_val, own_test = is_own[val_idx], is_own[test_idx]
    train_idx = np.sort(np.concatenate([train_idx, val_idx[own_val], test_idx[own_test]]))
    return train_idx, val_idx[~own_val], test_idx[~own_test]


def train_activity(data: datasets.Dataset, args) -> tuple[bytes, dict]:
    """Обучает классификатор активности. Возвращает tflite-модель и метаданные."""
    import tensorflow as tf
    from tensorflow import keras

    # Окна с классами вне ACTIVITY_ORDER (например, транспорт из старых
    # записей) отбрасываются: модель их выдавать не должна.
    known = np.isin(data.y_activity, ACTIVITY_ORDER)
    if not known.all():
        dropped = sorted(set(data.y_activity[~known]))
        print(f"Отброшено {int((~known).sum())} окон с классами вне списка: {', '.join(dropped)}")
    missing = [c for c in ACTIVITY_CHANNELS if c not in data.channels]
    if missing:
        raise SystemExit(f"В данных нет каналов {missing}")
    ch_idx = [data.channels.index(c) for c in ACTIVITY_CHANNELS]
    x_all = data.x[known][:, :, ch_idx]
    y_names = data.y_activity[known]
    subjects = data.subjects[known]
    groups = data.groups[known] if data.groups is not None else np.full(len(x_all), "unknown")
    is_own = np.char.startswith(groups.astype(str), "own:")

    present = [a for a in ACTIVITY_ORDER if a in set(y_names)]
    print(f"\nКлассы активности в данных: {', '.join(present)}")
    if len(present) < 2:
        raise SystemExit("Для обучения нужно минимум два класса активности")

    train_idx, val_idx, test_idx = _split(subjects, is_own, args)
    print(f"Обучение: {len(train_idx)} окон (из них своих {int(is_own[train_idx].sum())}), "
          f"валидация: {len(val_idx)}, тест: {len(test_idx)} — деление по испытуемым")

    y = encode(y_names, present)
    k = len(present)
    placements = (data.y_placement[known] if data.y_placement is not None
                  else np.full(len(x_all), "UNKNOWN"))
    combo = np.char.add(np.char.add(y_names.astype(str), "/"), placements.astype(str))
    w_train = balanced_weights(y_names[train_idx], present, is_own[train_idx], args.own_share,
                               combo[train_idx], args.own_combo_power)
    # Валидация взвешена по классам так же — иначе ранняя остановка
    # выбирала бы эпоху, лучшую для самого частого класса.
    w_val = balanced_weights(y_names[val_idx], present, None, 0.0)

    # Статистики нормировки входа модели — по обучающей части, уже после
    # расчёта производных каналов. Подмножества окон для этого достаточно.
    rng = np.random.default_rng(args.seed)
    sample = x_all[train_idx[rng.choice(len(train_idx), min(30000, len(train_idx)), replace=False)]]
    feats = np.asarray(_motion_features_layer()(tf.constant(sample)))
    f_mean = feats.reshape(-1, feats.shape[-1]).mean(axis=0)
    f_var = np.maximum(feats.reshape(-1, feats.shape[-1]).var(axis=0), 1e-6)

    model = build_activity_model(x_all.shape[1], x_all.shape[2], k, f_mean, f_var)
    steps = math.ceil(len(train_idx) / args.batch_size) * args.epochs
    model.compile(
        optimizer=keras.optimizers.AdamW(
            keras.optimizers.schedules.CosineDecay(args.learning_rate, steps, alpha=0.02),
            weight_decay=1e-4,
        ),
        # Сглаживание меток: на выходе — вероятности, которые дальше
        # сглаживаются по времени и перевзвешиваются положением телефона.
        # Сеть, уверенная на 0.999 в каждом окне, сделала бы эти поправки бессильными.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        weighted_metrics=["accuracy"],
    )
    model.summary(line_length=100)

    has_val = len(val_idx) > 0
    if not has_val:
        print("Валидационных испытуемых не осталось — обучение без ранней остановки.")
    # Валидация — чужие люди, а не телефон, на котором работает приложение,
    # и её потери скачут от эпохи к эпохе. Поэтому по умолчанию (--patience 0)
    # график скорости проходится целиком, а валидация только печатается:
    # ранняя остановка выбирала бы раннюю эпоху почти случайно.
    model.fit(
        _batches(x_all[train_idx], _onehot(y[train_idx], k), w_train,
                 args.batch_size, augment=not args.no_augment, seed=args.seed),
        validation_data=(x_all[val_idx], _onehot(y[val_idx], k), w_val) if has_val else None,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            ),
        ] if has_val and args.patience > 0 else [],
        verbose=2,
    )

    accuracy = 0.0
    if len(test_idx):
        print("\n=== Классификатор активности: оценка на отложенных испытуемых ===")
        print("(эти люди не участвовали ни в обучении, ни в ранней остановке)")
        y_pred = model.predict(x_all[test_idx], batch_size=512, verbose=0).argmax(axis=1)
        accuracy = report(y[test_idx], y_pred, present)

        # Средняя точность по всем позициям на теле вводит в заблуждение:
        # телефон носят в кармане и в руке, а не на голени и груди. Разбивка
        # показывает, как модель работает именно там, где приложение её и вызовет.
        print("\nТочность по источникам и позициям:")
        test_groups = groups[test_idx]
        for g in sorted(set(test_groups)):
            mask = test_groups == g
            print(f"  {g:<18} {float((y_pred[mask] == y[test_idx][mask]).mean()):.4f}  "
                  f"({int(mask.sum())} окон)")

    meta = {
        "channels": ACTIVITY_CHANNELS,
        "activity_labels": present,
        # Нормировка встроена в модель: приложение подаёт сырые показания.
        "channel_mean": [0.0] * len(ACTIVITY_CHANNELS),
        "channel_std": [1.0] * len(ACTIVITY_CHANNELS),
        "activity_accuracy": accuracy,
        "activity_architecture": "rescnn+invariant-channels",
    }
    return to_tflite(model), meta


# --------------------------------------------------------------------------
# Модель положения
# --------------------------------------------------------------------------

STRAPPED_GROUPS = {"rw:thigh", "rw:head"}


def _without_light(x: np.ndarray) -> np.ndarray:
    """Признаки положения так, как если бы датчика света не было (lux = −1)."""
    ix = {n: i for i, n in enumerate(feat.PLACEMENT_FEATURE_NAMES)}
    out = x.copy()
    out[:, ix["light_available"]] = 0.0
    out[:, ix["light_log_mean"]] = 0.0
    out[:, ix["light_log_std"]] = 0.0
    # Отсутствующий датчик даёт −1, после отсечения 0 лк — ниже порога темноты.
    out[:, ix["light_dark_ratio"]] = 1.0
    return out


def build_placement_model(n_features: int, classes: int):
    """MLP по агрегированным признакам.

    Свёртки не нужны: вход — уже готовые статистики окна, временной структуры
    в нём нет. Нормализация батчей между слоями стабилизирует обучение на
    смеси источников, у которых признаки распределены очень по-разному
    (у части датасетов нет ни света, ни приближения).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(128, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(classes, activation="softmax"),
        ],
        name="placement_mlp",
    )


def train_placement(sources: list[datasets.Dataset], args) -> tuple[bytes, dict] | None:
    """Обучает классификатор положения телефона.

    Берёт все источники, где положение размечено: собственные записи,
    RealWorld (бедро — карман, голова — у уха), MotionSense и Shoaib (карман),
    ExtraSensory (карман, рука, стол, у уха). Окна с меткой UNKNOWN
    отбрасываются: это позиции, которым не соответствует ни один класс.

    Возвращает None, если размеченных данных меньше двух классов —
    штатная ситуация, в которой приложение определяет положение правилами.
    """
    from tensorflow import keras

    labelled = [d for d in sources if d is not None and d.y_placement is not None]
    if not labelled:
        print("\nНет источников с разметкой положения — модель не обучается.")
        return None

    # Признаки положения — это тот же контекстный вектор, который уже
    # посчитан для каждого окна при загрузке. Пересчитывать не нужно.
    x_all = np.concatenate([d.context for d in labelled])
    y_all = np.concatenate([d.y_placement for d in labelled])
    s_all = np.concatenate([d.subjects for d in labelled])
    a_all = np.concatenate([d.y_activity for d in labelled]).astype(str)
    g_all = np.concatenate([d.groups if d.groups is not None else np.full(len(d), "unknown")
                            for d in labelled]).astype(str)

    keep = y_all != "UNKNOWN"
    x_all, y_all, s_all, a_all, g_all = x_all[keep], y_all[keep], s_all[keep], a_all[keep], g_all[keep]
    is_own = np.char.startswith(g_all, "own:")

    # В RealWorld телефоны закреплены поверх одежды: на бедре ремешком,
    # на голове повязкой. Освещённость там — не «карман» и не «у уха»:
    # у трети окон «бедра» больше 20 лк, бывает и 2500. С такими окнами модель
    # учит, что карман бывает светлым, и перестаёт доверять датчику света,
    # хотя в настоящем кармане темно. Поэтому для этих окон свет считается
    # неизмеренным — так же, как в датасетах без этого датчика: позу и движение
    # они учат по-прежнему, а смысл освещённости модель берёт из записей
    # с телефона.
    if not args.keep_strapped_light:
        strapped = np.isin(g_all, list(STRAPPED_GROUPS))
        if strapped.any():
            x_all = x_all.copy()
            x_all[strapped] = _without_light(x_all[strapped])
            print(f"\nОсвещённость не учитывается в {int(strapped.sum())} окнах "
                  f"закреплённых снаружи телефонов ({', '.join(sorted(set(g_all[strapped])))})")

    # Ограничение перекоса. RealWorld даёт десятки тысяч окон «в кармане»
    # и «у уха», а собственные записи — сотни окон «на столе». Урезаем
    # крупные публичные классы, сохраняя разнообразие испытуемых.
    # Свои записи не урезаются никогда.
    counts = {c: int(np.count_nonzero((y_all == c) & ~is_own)) for c in set(y_all)}
    smallest = min((v for v in counts.values() if v > 0), default=0)
    cap = max(args.placement_min_cap, smallest * args.placement_cap_ratio)
    rng = np.random.default_rng(args.seed)
    selected = [np.flatnonzero(is_own)]
    for cls in counts:
        idx = np.flatnonzero((y_all == cls) & ~is_own)
        if len(idx) > cap:
            picked = []
            subs = np.unique(s_all[idx])
            per_subject = max(1, int(cap // len(subs)))
            for sub in subs:
                sub_idx = idx[s_all[idx] == sub]
                picked.append(rng.choice(sub_idx, size=min(per_subject, len(sub_idx)), replace=False))
            idx = np.concatenate(picked)
        selected.append(idx)
    selected = np.sort(np.concatenate(selected))
    if len(selected) < len(y_all):
        print(f"\nПерекос классов положения урезан: {len(y_all)} → {len(selected)} окон "
              f"(потолок {cap} публичных окон на класс)")
    x_all, y_all, s_all, a_all, is_own = (x_all[selected], y_all[selected], s_all[selected],
                                          a_all[selected], is_own[selected])

    present = [p for p in PLACEMENT_ORDER if p in set(y_all)]
    if len(present) < 2:
        print(f"\nРазмечено только {len(present)} класс(а) положения — "
              "модель не обучается, приложение будет использовать правила.")
        return None

    print(f"\nКлассы положения в данных: {', '.join(present)}")
    for p in present:
        m = y_all == p
        print(f"  {p:<9} окон {int(m.sum()):>6}, из них своих {int((m & is_own).sum())}")

    train_idx, val_idx, test_idx = _split(s_all, is_own, args)
    if len(train_idx) == 0:
        print("Слишком мало записей для обучения — модель не обучается.")
        return None

    y = encode(y_all, present)
    k = len(present)
    mean = x_all[train_idx].mean(axis=0)
    std = x_all[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    xn = ((x_all - mean) / std).astype(np.float32)

    x_tr, y_tr, own_tr = x_all[train_idx], y_all[train_idx], is_own[train_idx]
    combo_tr = np.char.add(np.char.add(a_all[train_idx], "/"), y_tr.astype(str))
    w_train = balanced_weights(y_tr, present, own_tr, args.placement_own_share,
                               combo_tr, args.own_combo_power)
    has_val = len(val_idx) > 0
    validation = None
    if has_val:
        validation = (xn[val_idx], _onehot(y[val_idx], k),
                      balanced_weights(y_all[val_idx], present, None, 0.0))

    model = build_placement_model(x_all.shape[1], k)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        weighted_metrics=["accuracy"],
    )
    # Ранняя остановка здесь вредна вдвойне: в публичной валидации есть только
    # карман и ухо, а «в руке» и «на столе» — только в своих записях,
    # которые целиком в обучении. Поэтому по умолчанию — фиксированное число эпох.
    model.fit(
        ((x_tr - mean) / std).astype(np.float32), _onehot(encode(y_tr, present), k),
        sample_weight=w_train,
        validation_data=validation,
        epochs=args.placement_epochs,
        batch_size=256,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )] if has_val and args.patience > 0 else [],
        verbose=2,
    )

    accuracy = 0.0
    if len(test_idx):
        print("\n=== Классификатор положения телефона: отложенные испытуемые ===")
        y_pred = model.predict(xn[test_idx], verbose=0).argmax(axis=1)
        accuracy = report(y[test_idx], y_pred, present)

    meta = {
        "placement_labels": present,
        "placement_feature_names": feat.PLACEMENT_FEATURE_NAMES,
        "placement_mean": mean.tolist(),
        "placement_std": std.tolist(),
        "placement_accuracy": accuracy,
    }
    return to_tflite(model), meta


# --------------------------------------------------------------------------
# Загрузка источников и запуск
# --------------------------------------------------------------------------

SOURCES = ["uci", "realworld", "motionsense", "shoaib", "extrasensory", "own"]


def parse_sources(spec: str) -> list[str]:
    aliases = {"both": ["uci", "own"], "all": list(SOURCES)}
    wanted: list[str] = []
    for item in spec.split(","):
        item = item.strip().lower()
        for name in aliases.get(item, [item]):
            if name not in SOURCES:
                raise SystemExit(f"Неизвестный источник: {name}. Допустимо: {', '.join(SOURCES)}, both, all")
            if name not in wanted:
                wanted.append(name)
    return wanted


def load_public(wanted: list[str], args) -> list[datasets.Dataset]:
    """Публичные источники: из кэша .npz, а при первом запуске — скачать и разобрать."""
    loaders = {
        "uci": ("UCI HAR", lambda: datasets.load_uci(args.uci_dir)),
        "realworld": ("RealWorld HAR",
                      lambda: datasets.load_realworld(args.realworld_dir, args.realworld_subjects)),
        "motionsense": ("MotionSense", lambda: datasets.load_motionsense(args.motionsense_dir)),
        "shoaib": ("Shoaib et al.", lambda: datasets.load_shoaib(args.shoaib_dir)),
        "extrasensory": ("ExtraSensory",
                         lambda: datasets.load_extrasensory(args.extrasensory_dir,
                                                            args.extrasensory_max_minutes,
                                                            args.extrasensory_users, args.seed)),
    }
    out = []
    for key in wanted:
        if key == "own":
            continue
        title, load = loaders[key]
        tag = key
        if key == "realworld":
            tag += f"_s{args.realworld_subjects}"
        if key == "extrasensory":
            tag += f"_m{args.extrasensory_max_minutes}_u{args.extrasensory_users or 'all'}_seed{args.seed}"
        cache = args.cache_dir / f"{tag}.npz"
        ds = None if args.no_cache else datasets.load_cache(cache)
        if ds is not None:
            print(f"\n{title}: из кэша {cache.name}")
        else:
            print(f"\nЗагрузка {title} …")
            ds = load()
            if ds is None:
                if len(wanted) == 1:
                    raise SystemExit(f"Не удалось загрузить {title}")
                print(f"  {title} недоступен, пропускаем")
                continue
            # Словари полных окон нужны только при разборе; в памяти RealWorld
            # они занимают гигабайты.
            if hasattr(ds, "raw_windows"):
                del ds.raw_windows
            if not args.no_cache:
                datasets.save_cache(ds, cache)
        print(ds.summary())
        if args.max_windows_per_subject_activity > 0:
            before = len(ds)
            ds = datasets.cap_per_subject_activity(ds, args.max_windows_per_subject_activity, args.seed)
            if len(ds) < before:
                print(f"  {before} -> {len(ds)} окон "
                      f"(не больше {args.max_windows_per_subject_activity} на человека и активность)")
        out.append(ds)
    return out


def load_own_filtered(args, exclude_days: set[str] | None = None,
                      only_days: set[str] | None = None) -> datasets.Dataset | None:
    print(f"\nЗагрузка собственных записей из {args.own_dir} …")
    own = datasets.load_own(args.own_dir, exclude_days=exclude_days, only_days=only_days)
    if own is not None:
        print(own.summary())
    return own


def train_models(sources: list[datasets.Dataset], args) -> tuple[bytes, bytes | None, dict]:
    """Обучает обе модели на готовых источниках. Возвращает (активность, положение, meta)."""
    data = datasets.merge(sources)
    if len(sources) > 1:
        print(f"\nОбъединённый набор: {len(data)} окон, общие каналы: {', '.join(data.channels)}")

    activity_tflite, meta = train_activity(data, args)
    del data

    placement = train_placement(sources, args)
    if placement is not None:
        placement_tflite, placement_meta = placement
        meta.update(placement_meta)
    else:
        placement_tflite = None
        # Метаданные положения всё равно нужны: приложение проверяет
        # совместимость набора признаков до загрузки моделей.
        meta.update({
            "placement_labels": PLACEMENT_ORDER,
            "placement_feature_names": feat.PLACEMENT_FEATURE_NAMES,
            "placement_mean": [0.0] * feat.PLACEMENT_FEATURE_COUNT,
            "placement_std": [1.0] * feat.PLACEMENT_FEATURE_COUNT,
            "placement_accuracy": 0.0,
        })

    meta.update({
        "version": 1,
        "sample_rate_hz": datasets.SAMPLE_RATE_HZ,
        "window_size": datasets.WINDOW_SIZE,
        "window_stride": datasets.WINDOW_STRIDE,
        "source": args.source,
        "train_windows": int(sum(len(s) for s in sources)),
    })
    return activity_tflite, placement_tflite, meta


def write_assets(assets_dir: Path, activity_tflite: bytes, placement_tflite: bytes | None,
                 meta: dict) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    activity_path = assets_dir / "activity_model.tflite"
    activity_path.write_bytes(activity_tflite)
    print(f"\nЗаписано: {activity_path} ({len(activity_tflite) / 1024:.1f} КБ)")

    stale = assets_dir / "placement_model.tflite"
    if placement_tflite is not None:
        stale.write_bytes(placement_tflite)
        print(f"Записано: {stale} ({len(placement_tflite) / 1024:.1f} КБ)")
    elif stale.exists():
        # Оставить старую модель рядом с новыми метаданными нельзя:
        # статистики нормировки уже не те, и предсказания были бы мусором.
        stale.unlink()
        print(f"Удалена устаревшая {stale.name} (новых данных о положении нет)")

    meta_path = assets_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="all",
                        help="источники через запятую: uci, realworld, motionsense, shoaib, "
                             "extrasensory, own; both = uci+own; all = все")
    data = REPO_ROOT / "ml" / "data"
    parser.add_argument("--uci-dir", type=Path, default=data, help="где искать UCI HAR")
    parser.add_argument("--own-dir", type=Path, default=data / "own",
                        help="каталог с CSV из режима «Сбор данных»")
    parser.add_argument("--own-exclude-days", default="",
                        help="дни YYYYMMDD через запятую, чьи записи не участвуют в обучении")
    parser.add_argument("--realworld-dir", type=Path, default=data / "realworld")
    parser.add_argument("--motionsense-dir", type=Path, default=data / "motionsense")
    parser.add_argument("--shoaib-dir", type=Path, default=data / "shoaib")
    parser.add_argument("--extrasensory-dir", type=Path, default=data / "extrasensory")
    parser.add_argument("--extrasensory-max-minutes", type=int,
                        default=datasets.EXTRASENSORY_MAX_MINUTES,
                        help="минут одного сочетания (активность, положение) у одного человека")
    parser.add_argument("--extrasensory-users", type=int, default=None,
                        help="сколько участников ExtraSensory брать (по умолчанию все 60)")
    parser.add_argument("--realworld-subjects", type=int, default=datasets.REALWORLD_SUBJECTS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="кэш разобранных публичных датасетов (.npz)")
    parser.add_argument("--no-cache", action="store_true", help="не читать и не писать кэш")
    parser.add_argument("--max-windows-per-subject-activity", type=int, default=1500,
                        help="потолок окон на пару (человек, активность) в публичных источниках; "
                             "0 — без ограничения")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=0,
                        help="ранняя остановка по валидации; 0 — пройти весь график скорости")
    parser.add_argument("--placement-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--own-share", type=float, default=0.3,
                        help="доля суммарного веса собственных записей в обучении модели активности")
    parser.add_argument("--placement-own-share", type=float, default=0.4,
                        help="то же для модели положения")
    parser.add_argument("--no-augment", action="store_true", help="без аугментации окон")
    parser.add_argument("--keep-strapped-light", action="store_true",
                        help="не отключать освещённость у закреплённых снаружи телефонов RealWorld")
    parser.add_argument("--own-combo-power", type=float, default=0.5,
                        help="выравнивание сочетаний «активность × положение» в своих записях: "
                             "вес окна ∝ n^(−p); 0 — без выравнивания")
    parser.add_argument("--own-in-test", action="store_true",
                        help="делить и собственные записи на train/val/test (по умолчанию все в обучение)")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                        help="доля публичных испытуемых в тестовой выборке")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="доля публичных испытуемых в валидационной выборке (ранняя остановка)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement-cap-ratio", type=int, default=20,
                        help="во сколько раз крупный класс положения может превышать самый малый")
    parser.add_argument("--placement-min-cap", type=int, default=3000,
                        help="нижняя граница потолка публичных окон на класс положения")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR,
                        help="куда положить модели (assets приложения)")
    return parser


def setup(args) -> None:
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
    except ImportError:
        raise SystemExit("Не установлен TensorFlow: pip install -r ml/requirements.txt")
    tf.random.set_seed(args.seed)
    print(f"TensorFlow {tf.__version__}")


def main() -> None:
    args = build_parser().parse_args()
    setup(args)
    wanted = parse_sources(args.source)
    sources = load_public(wanted, args)
    if "own" in wanted:
        exclude = {d for d in args.own_exclude_days.split(",") if d}
        own = load_own_filtered(args, exclude_days=exclude or None)
        if own is None:
            if wanted == ["own"]:
                raise SystemExit(f"В {args.own_dir} нет пригодных CSV.")
            print("  собственных записей нет, продолжаем без них")
        else:
            sources.append(own)
    if not sources:
        raise SystemExit("Нет ни одного источника данных")

    activity_tflite, placement_tflite, meta = train_models(sources, args)
    write_assets(args.assets_dir, activity_tflite, placement_tflite, meta)
    print("\nГотово. Пересоберите приложение, чтобы модели попали в APK.")


if __name__ == "__main__":
    main()
