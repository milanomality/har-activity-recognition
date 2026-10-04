"""Обучение и экспорт моделей распознавания активности.

Обучает две модели:

* классификатор активности — одномерная свёрточная сеть по «сырым» окнам
  сигналов движения (акселерометр, гироскоп и, если данные позволяют,
  магнитометр);
* классификатор положения телефона — небольшой MLP по 16 агрегированным
  признакам, в которых главную роль играют датчики приближения и
  освещённости.

Результат кладётся прямо в assets приложения: два .tflite и model_meta.json
со статистиками нормировки, списком каналов и списком классов. Приложение
читает метаданные и подстраивается под то, на чём модель реально обучена.

Примеры запуска:
    python ml/train.py --source uci
    python ml/train.py --source own --own-dir ml/data/own
    python ml/train.py --source both --epochs 60
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"


def build_activity_model(window_size: int, channels: int, context_size: int, classes: int):
    """Двухветочная сеть: свёртки по сигналам плюс контекстные признаки.

    Ветка сигналов — одномерная CNN вдоль времени. Свёртки, а не полносвязная
    сеть по признакам: сеть сама находит характерные формы шага и толчка,
    и не нужно вручную подбирать частотные полосы. Три блока с понижением
    разрешения дают рецептивное поле порядка всего окна. GlobalAveragePooling
    вместо Flatten делает модель устойчивой к тому, в какой момент окна
    начался шаг, и резко сокращает число параметров.

    Ветка контекста — маленький MLP по 16 агрегированным признакам.
    Через неё в решение попадают датчики освещённости и приближения,
    которых нет среди каналов движения, и ориентация телефона. Это и есть
    механизм, которым все пять датчиков влияют на определение активности:
    один и тот же рисунок ускорения означает разное, когда телефон
    в кармане и когда он лежит экраном вверх на столе.

    Ветка намеренно узкая. Контекст — 16 чисел против 128×C отсчётов
    сигнала, и без ограничения ёмкости сеть склонна опереться на лёгкие
    признаки вместо формы сигнала.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    motion_in = layers.Input(shape=(window_size, channels), name="motion")
    context_in = layers.Input(shape=(context_size,), name="context")

    m = layers.Conv1D(64, 7, padding="same", activation="relu")(motion_in)
    m = layers.BatchNormalization()(m)
    m = layers.MaxPooling1D(2)(m)
    m = layers.Conv1D(128, 5, padding="same", activation="relu")(m)
    m = layers.BatchNormalization()(m)
    m = layers.MaxPooling1D(2)(m)
    m = layers.Conv1D(128, 3, padding="same", activation="relu")(m)
    m = layers.BatchNormalization()(m)
    m = layers.GlobalAveragePooling1D()(m)
    m = layers.Dropout(0.4)(m)

    c = layers.Dense(32, activation="relu")(context_in)
    c = layers.Dropout(0.3)(c)
    c = layers.Dense(16, activation="relu")(c)

    joined = layers.Concatenate()([m, c])
    joined = layers.Dense(64, activation="relu")(joined)
    joined = layers.Dropout(0.2)(joined)
    out = layers.Dense(classes, activation="softmax")(joined)

    return keras.Model(inputs=[motion_in, context_in], outputs=out, name="activity_cnn_ctx")


def build_placement_model(n_features: int, classes: int):
    """MLP по агрегированным признакам.

    Здесь свёртки не нужны: вход — уже готовые статистики окна, временной
    структуры в нём нет. Сеть намеренно маленькая, потому что собственных
    размеченных данных о положении телефона обычно десятки минут, а не часы.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(classes, activation="softmax"),
        ],
        name="placement_mlp",
    )


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


def class_weights(labels: np.ndarray, names: list[str]) -> dict[int, float]:
    """Веса классов, обратные их частоте.

    В HAR-данных перекос неизбежен: «покой» набирается сам собой, а подъём
    по лестнице приходится записывать специально. Без взвешивания сеть
    выучивает частый класс и игнорирует редкий.
    """
    index = {name: i for i, name in enumerate(names)}
    counts = np.zeros(len(names))
    for label in labels:
        if label in index:
            counts[index[label]] += 1
    total = counts.sum()
    weights = {}
    for i, count in enumerate(counts):
        # Класс может отсутствовать в обучающей части после деления по людям.
        # Вес 1.0, а не 0.0: нулевой вес молча выключил бы класс, если бы
        # он всё-таки встретился.
        weights[i] = float(total / (len(names) * count)) if count > 0 else 1.0
    return weights


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
        print("Итоговая accuracy посчитана без них и завышена относительно "
              "полноценной проверки.")

    print(f"\nAccuracy: {accuracy:.4f} (на {len(y_true)} окнах)")
    return accuracy


def train_activity(data: datasets.Dataset, args) -> tuple[bytes, dict]:
    """Обучает классификатор активности. Возвращает tflite-модель и метаданные."""
    from tensorflow import keras

    # Окна с классами вне ACTIVITY_ORDER (например, транспорт из старых
    # записей) отбрасываются: модель их выдавать не должна.
    known = np.isin(data.y_activity, ACTIVITY_ORDER)
    if not known.all():
        dropped = sorted(set(data.y_activity[~known]))
        print(f"Отброшено {int((~known).sum())} окон с классами вне списка: {', '.join(dropped)}")
        data = dataclasses.replace(
            data,
            x=data.x[known],
            context=data.context[known],
            y_activity=data.y_activity[known],
            y_placement=None if data.y_placement is None else data.y_placement[known],
            subjects=data.subjects[known],
            groups=None if data.groups is None else data.groups[known],
        )

    present = [a for a in ACTIVITY_ORDER if a in set(data.y_activity)]
    print(f"\nКлассы активности в данных: {', '.join(present)}")
    if len(present) < 2:
        raise SystemExit("Для обучения нужно минимум два класса активности")

    train_idx, val_idx, test_idx = datasets.split_by_subject(
        data, args.test_fraction, args.val_fraction, args.seed,
    )
    print(f"Обучение: {len(train_idx)} окон, валидация: {len(val_idx)}, "
          f"тест: {len(test_idx)} — деление по испытуемым, группы не пересекаются")

    x_train, x_val, x_test = data.x[train_idx], data.x[val_idx], data.x[test_idx]
    c_train, c_val, c_test = data.context[train_idx], data.context[val_idx], data.context[test_idx]
    y_train = encode(data.y_activity[train_idx], present)
    y_val = encode(data.y_activity[val_idx], present)
    y_test = encode(data.y_activity[test_idx], present)

    # Статистики нормировки считаем ТОЛЬКО по обучающей части, иначе
    # информация о тесте просачивается в модель и оценка завышается.
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    c_mean = c_train.mean(axis=0)
    c_std = c_train.std(axis=0)
    c_std[c_std < 1e-6] = 1.0

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std
    x_test_n = (x_test - mean) / std
    c_train_n = (c_train - c_mean) / c_std
    c_val_n = (c_val - c_mean) / c_std
    c_test_n = (c_test - c_mean) / c_std

    # Предупреждаем о ловушке: если флаг доступности датчика принимает
    # одно и то же значение во всей выборке, он несёт не информацию
    # о датчике, а метку источника данных.
    for flag in ("light_available", "prox_available"):
        i = feat.PLACEMENT_FEATURE_NAMES.index(flag)
        values = np.unique(data.context[:, i])
        if len(values) == 1:
            print(f"  внимание: признак {flag} постоянен ({values[0]:.0f}) — "
                  "соответствующий датчик не участвует в решении")

    model = build_activity_model(
        data.x.shape[1], data.x.shape[2], data.context.shape[1], len(present),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Ранняя остановка смотрит на ВАЛИДАЦИЮ. Тестовые испытуемые не
    # участвуют ни в обучении, ни в выборе эпохи.
    # При двух-трёх испытуемых валидационной группы может не остаться —
    # тогда обучаемся фиксированное число эпох, а не молча теряем
    # раннюю остановку вместе с восстановлением лучших весов.
    has_val = len(val_idx) > 0
    if not has_val:
        print("Валидационных испытуемых не осталось — обучение без ранней остановки.")
    model.fit(
        {"motion": x_train_n, "context": c_train_n}, y_train,
        validation_data=({"motion": x_val_n, "context": c_val_n}, y_val) if has_val else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(data.y_activity[train_idx], present),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=12, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
            ),
        ] if has_val else [],
        verbose=2,
    )

    print("\n=== Классификатор активности: оценка на отложенных испытуемых ===")
    print("(эти люди не участвовали ни в обучении, ни в ранней остановке)")
    test_inputs = {"motion": x_test_n, "context": c_test_n}
    y_pred = model.predict(test_inputs, verbose=0).argmax(axis=1)
    accuracy = report(y_test, y_pred, present)

    # Насколько решение опирается на контекст и на какие именно датчики.
    # Обнуление группы признаков (после нормировки ноль = среднее обучающей
    # выборки) равносильно «модель этого не видит». Общий вклад контекста
    # мало что говорит: ориентация телефона выводится из акселерометра,
    # а освещённость и приближение — отдельные датчики, и именно их вклад
    # отвечает на вопрос, участвуют ли они в решении на самом деле.
    names = feat.PLACEMENT_FEATURE_NAMES
    groups = {
        "весь контекст": list(range(len(names))),
        "освещённость": [i for i, n in enumerate(names) if n.startswith("light")],
        "приближение": [i for i, n in enumerate(names) if n.startswith("prox")],
        "ориентация": [i for i, n in enumerate(names) if n.startswith("grav")],
    }

    # Средняя точность по всем позициям на теле вводит в заблуждение:
    # телефон носят в кармане и в руке, а не на голени и груди. Разбивка
    # показывает, как модель работает именно там, где приложение её и вызовет.
    if data.groups is not None:
        print("\nТочность по источникам и позициям:")
        test_groups = data.groups[test_idx]
        for g in sorted(set(test_groups)):
            mask = test_groups == g
            if mask.sum() == 0:
                continue
            group_accuracy = float((y_pred[mask] == y_test[mask]).mean())
            print(f"  {g:<18} {group_accuracy:.4f}  ({int(mask.sum())} окон)")

    print("\nВклад групп признаков (обнуление группы на тесте):")
    ablation = {}
    for title, idx in groups.items():
        masked = c_test_n.copy()
        masked[:, idx] = 0.0
        without = model.predict(
            {"motion": x_test_n, "context": masked}, verbose=0
        ).argmax(axis=1)
        without_accuracy = float((without == y_test).mean())
        ablation[title] = without_accuracy
        print(f"  без «{title}»: {without_accuracy:.4f} "
              f"(вклад {accuracy - without_accuracy:+.4f})")

    blind_accuracy = ablation["весь контекст"]
    light_accuracy = ablation["освещённость"]

    meta = {
        "channels": data.channels,
        "activity_labels": present,
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "activity_context_features": feat.PLACEMENT_FEATURE_NAMES,
        "activity_context_mean": c_mean.tolist(),
        "activity_context_std": c_std.tolist(),
        "activity_accuracy": accuracy,
        "activity_accuracy_without_context": blind_accuracy,
        "activity_accuracy_without_light": light_accuracy,
    }
    return to_tflite(model), meta


def train_placement(sources: list[datasets.Dataset], args) -> tuple[bytes, dict] | None:
    """Обучает классификатор положения телефона.

    Берёт все источники, где положение размечено: собственные записи и
    RealWorld HAR (бедро — карман, голова — у уха). Окна с меткой UNKNOWN
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

    keep = y_all != "UNKNOWN"
    x_all, y_all, s_all = x_all[keep], y_all[keep], s_all[keep]

    # Ограничение перекоса. RealWorld даёт десятки тысяч окон «в кармане»
    # и «у уха», а собственные записи — десятки окон «в руке» и «на столе».
    # При соотношении 1000:1 взвешивание классов уже не спасает: редкий
    # класс просто не встречается в мини-батчах. Урезаем крупные классы
    # до кратного размеру самого малого, сохраняя разнообразие испытуемых.
    counts = {c: int(np.count_nonzero(y_all == c)) for c in set(y_all)}
    smallest = min(counts.values())
    cap = max(smallest * args.placement_cap_ratio, args.placement_min_cap)
    rng = np.random.default_rng(args.seed)
    selected = []
    for cls, total in counts.items():
        idx = np.where(y_all == cls)[0]
        if total > cap:
            # Отбираем равномерно по испытуемым, а не случайно из всей кучи,
            # иначе редкий испытуемый может выпасть целиком.
            picked = []
            subs = np.unique(s_all[idx])
            per_subject = max(1, int(cap // len(subs)))
            for sub in subs:
                sub_idx = idx[s_all[idx] == sub]
                take = min(per_subject, len(sub_idx))
                picked.append(rng.choice(sub_idx, size=take, replace=False))
            idx = np.concatenate(picked)
        selected.append(idx)
    selected = np.sort(np.concatenate(selected))
    if len(selected) < len(y_all):
        print(f"\nПерекос классов положения урезан: {len(y_all)} → {len(selected)} окон "
              f"(потолок {cap} на класс)")
    x_all, y_all, s_all = x_all[selected], y_all[selected], s_all[selected]

    present = [p for p in PLACEMENT_ORDER if p in set(y_all)]
    if len(present) < 2:
        print(f"\nРазмечено только {len(present)} класс(а) положения — "
              "модель не обучается, приложение будет использовать правила.")
        return None

    print(f"\nКлассы положения в данных: {', '.join(present)}")
    vals, counts = np.unique(y_all, return_counts=True)
    print("  окон: " + ", ".join(f"{v}={c}" for v, c in zip(vals, counts)))

    # Упаковываем в Dataset только ради деления по испытуемым — оно
    # работает с subjects, а остальные поля здесь не нужны.
    holder = datasets.Dataset(
        x=np.zeros((len(x_all), 1, 1), dtype=np.float32),
        context=x_all,
        y_activity=y_all,
        y_placement=y_all,
        subjects=s_all,
        channels=[],
    )
    x = x_all
    labels = y_all
    train_idx, val_idx, test_idx = datasets.split_by_subject(
        holder, args.test_fraction, args.val_fraction, args.seed,
    )
    if len(test_idx) == 0 or len(train_idx) == 0:
        print("Слишком мало записей для честного деления — модель не обучается.")
        return None

    y = encode(labels, present)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0

    # Записей обычно мало, и отдельной валидации может не остаться.
    # Тогда обходимся без ранней остановки, но тест всё равно не трогаем.
    has_val = len(val_idx) > 0
    validation = ((x[val_idx] - mean) / std, y[val_idx]) if has_val else None
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )] if has_val else []
    if not has_val:
        print("Валидационных записей нет — обучаем фиксированное число эпох.")

    model = build_placement_model(x.shape[1], len(present))
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        (x_train - mean) / std, y_train,
        validation_data=validation,
        epochs=args.epochs,
        batch_size=min(args.batch_size, max(8, len(x_train) // 8)),
        class_weight=class_weights(labels[train_idx], present),
        callbacks=callbacks,
        verbose=2,
    )

    print("\n=== Классификатор положения телефона ===")
    y_pred = model.predict((x_test - mean) / std, verbose=0).argmax(axis=1)
    accuracy = report(y_test, y_pred, present)

    meta = {
        "placement_labels": present,
        "placement_feature_names": feat.PLACEMENT_FEATURE_NAMES,
        "placement_mean": mean.tolist(),
        "placement_std": std.tolist(),
        "placement_accuracy": accuracy,
    }
    return to_tflite(model), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["uci", "own", "realworld", "both", "all"],
                        default="uci",
                        help="источник данных: uci | own | realworld | "
                             "both (uci+own) | all (uci+realworld+own)")
    parser.add_argument("--uci-dir", type=Path, default=REPO_ROOT / "ml" / "data",
                        help="куда скачивать и где искать UCI HAR")
    parser.add_argument("--own-dir", type=Path, default=REPO_ROOT / "ml" / "data" / "own",
                        help="каталог с CSV из режима «Сбор данных»")
    parser.add_argument("--realworld-dir", type=Path,
                        default=REPO_ROOT / "ml" / "data" / "realworld",
                        help="кэш датасета RealWorld HAR")
    parser.add_argument("--realworld-subjects", type=int, default=datasets.REALWORLD_SUBJECTS,
                        help="сколько испытуемых RealWorld загружать (1..15)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="доля испытуемых в тестовой выборке")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="доля испытуемых в валидационной выборке (ранняя остановка)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placement-cap-ratio", type=int, default=20,
                        help="во сколько раз крупный класс положения может превышать самый малый")
    parser.add_argument("--placement-min-cap", type=int, default=400,
                        help="нижняя граница потолка на класс положения")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR,
                        help="куда положить модели (assets приложения)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        print(f"TensorFlow {tf.__version__}")
    except ImportError:
        raise SystemExit("Не установлен TensorFlow: pip install -r ml/requirements.txt")

    sources: list[datasets.Dataset] = []

    if args.source in ("uci", "both", "all"):
        print("\nЗагрузка UCI HAR …")
        uci = datasets.load_uci(args.uci_dir)
        print(uci.summary())
        sources.append(uci)

    if args.source in ("realworld", "all"):
        print(f"\nЗагрузка RealWorld HAR в {args.realworld_dir} …")
        print("  первый запуск качает около 1.5 ГБ, дальше берётся из кэша")
        realworld = datasets.load_realworld(args.realworld_dir, args.realworld_subjects)
        if realworld is None:
            if args.source == "realworld":
                raise SystemExit("Не удалось загрузить ни одной записи RealWorld HAR")
            print("  RealWorld недоступен, пропускаем")
        else:
            print(realworld.summary())
            sources.append(realworld)

    if args.source in ("own", "both", "all"):
        print(f"\nЗагрузка собственных записей из {args.own_dir} …")
        own = datasets.load_own(args.own_dir)
        if own is None:
            if args.source == "own":
                raise SystemExit(
                    f"В {args.own_dir} нет пригодных CSV. Запишите данные "
                    "во вкладке «Сбор данных» и выгрузите файлы на компьютер."
                )
            print("  собственных записей нет, продолжаем без них")
        else:
            print(own.summary())
            sources.append(own)

    data = datasets.merge(sources)
    if len(sources) > 1:
        print(f"\nОбъединённый набор: {len(data)} окон, "
              f"общие каналы: {', '.join(data.channels)}")

    activity_tflite, meta = train_activity(data, args)

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
        "train_windows": int(len(data)),
    })

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    activity_path = args.assets_dir / "activity_model.tflite"
    activity_path.write_bytes(activity_tflite)
    print(f"\nЗаписано: {activity_path} ({len(activity_tflite) / 1024:.1f} КБ)")

    if placement_tflite is not None:
        placement_path = args.assets_dir / "placement_model.tflite"
        placement_path.write_bytes(placement_tflite)
        print(f"Записано: {placement_path} ({len(placement_tflite) / 1024:.1f} КБ)")
    else:
        # Оставить старую модель рядом с новыми метаданными нельзя:
        # статистики нормировки уже не те, и предсказания были бы мусором.
        stale = args.assets_dir / "placement_model.tflite"
        if stale.exists():
            stale.unlink()
            print(f"Удалена устаревшая {stale.name} (новых данных о положении нет)")

    meta_path = args.assets_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано: {meta_path}")
    print("\nГотово. Пересоберите приложение, чтобы модели попали в APK.")


if __name__ == "__main__":
    main()
