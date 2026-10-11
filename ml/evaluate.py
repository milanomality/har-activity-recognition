"""Проверка экспортированных .tflite на собственных записях.

Обучение и экспорт — разные вещи: модель может учиться прекрасно, а после
конвертации в TFLite давать другие числа (из-за слияния слоёв, других
реализаций операций, потери точности при квантизации). Этот скрипт
прогоняет именно тот файл, который поедет в APK, и печатает его метрики.

Записи, на которых модель обучалась, дадут завышенные цифры. Честная
проверка — ml/crossval.py, либо --days с днями, исключёнными из обучения.

Запуск:
    python ml/evaluate.py --own-dir ml/data/own
    python ml/evaluate.py --days 20261009 --assets-dir путь/к/моделям
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import datasets
import features as feat

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"


def run_tflite(model_path: Path, x: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
    """Прогоняет батч через TFLite-интерпретатор по одному примеру.

    Экспортированная модель имеет фиксированный батч 1 — ровно так же,
    как её вызывает приложение, поэтому и проверяем в том же режиме.

    У двухвходовой модели активности входы сопоставляются по рангу, как
    в приложении: порядок входов в .tflite не гарантирован, а незаполненный
    вход молча остаётся нулевым и портит метрики.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    out = interpreter.get_output_details()[0]

    if len(inputs) == 1:
        feeds = [(inputs[0], x)]
    else:
        if context is None:
            raise ValueError(f"{model_path.name} ждёт {len(inputs)} входа, а контекст не передан")
        motion = next(d for d in inputs if len(d["shape"]) == 3)
        ctx = next(d for d in inputs if len(d["shape"]) == 2)
        feeds = [(motion, x), (ctx, context)]

    predictions = []
    for i in range(len(x)):
        for detail, data in feeds:
            interpreter.set_tensor(detail["index"], data[i][np.newaxis, ...].astype(np.float32))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(out["index"])[0])
    return np.stack(predictions)


def predict_activity(meta: dict, own: datasets.Dataset, assets: Path) -> np.ndarray | None:
    """Вероятности классов `meta["activity_labels"]` для каждого окна — как в приложении."""
    model_path = assets / "activity_model.tflite"
    if not model_path.exists():
        return None
    # Берём из записи ровно те каналы, на которых обучалась модель,
    # в том же порядке — именно это делает приложение по model_meta.json.
    idx = [own.channels.index(ch) for ch in meta["channels"]]
    mean = np.array(meta["channel_mean"], dtype=np.float32)
    std = np.array(meta["channel_std"], dtype=np.float32)
    std[std <= 1e-6] = 1.0
    x = (own.x[:, :, idx] - mean) / std

    context = None
    if meta.get("activity_context_mean"):
        c_mean = np.array(meta["activity_context_mean"], dtype=np.float32)
        c_std = np.array(meta["activity_context_std"], dtype=np.float32)
        c_std[c_std <= 1e-6] = 1.0
        # Как в приложении: признаки контекста выбираются по именам из meta.
        names = meta.get("activity_context_features", feat.PLACEMENT_FEATURE_NAMES)
        ctx_idx = [feat.PLACEMENT_FEATURE_NAMES.index(n) for n in names]
        context = (own.context[:, ctx_idx] - c_mean) / c_std
    return run_tflite(model_path, x, context)


def predict_placement(meta: dict, own: datasets.Dataset, assets: Path) -> np.ndarray | None:
    """Вероятности классов `meta["placement_labels"]` для каждого окна."""
    model_path = assets / "placement_model.tflite"
    if not model_path.exists():
        return None
    # own.context — те же 16 признаков положения, что считает приложение.
    mean = np.array(meta["placement_mean"], dtype=np.float32)
    std = np.array(meta["placement_std"], dtype=np.float32)
    std[std <= 1e-6] = 1.0
    return run_tflite(model_path, (own.context - mean) / std)


def evaluate_activity(meta: dict, own: datasets.Dataset, assets: Path) -> None:
    probs = predict_activity(meta, own, assets)
    if probs is None:
        print("Нет activity_model.tflite — сначала запустите ml/train.py")
        return
    labels = meta["activity_labels"]
    known = np.isin(own.y_activity, labels)
    if not known.all():
        print(f"Пропущены классы, отсутствующие в модели: {', '.join(sorted(set(own.y_activity[~known])))}")
    y_true = np.array([labels.index(a) for a in own.y_activity[known]])
    print("\n=== Активность: .tflite на собственных записях ===")
    _report(y_true, probs[known].argmax(axis=1), labels)


def evaluate_placement(meta: dict, own: datasets.Dataset, assets: Path) -> None:
    probs = predict_placement(meta, own, assets)
    if probs is None:
        print("\nНет placement_model.tflite — положение телефона определяется правилами")
        return
    labels = meta["placement_labels"]
    known = np.isin(own.y_placement, labels)
    y_true = np.array([labels.index(p) for p in own.y_placement[known]])
    print("\n=== Положение телефона: .tflite на собственных записях ===")
    _report(y_true, probs[known].argmax(axis=1), labels)


def _report(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> None:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(names))),
        target_names=names,
        zero_division=0,
    ))
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    width = max(len(n) for n in names) + 1
    print("Матрица ошибок:")
    print(" " * width + "".join(f"{n[:6]:>8}" for n in names))
    for name, row in zip(names, matrix):
        print(f"{name:<{width}}" + "".join(f"{v:>8}" for v in row))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--own-dir", type=Path, default=REPO_ROOT / "ml" / "data" / "own")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--days", default="", help="только записи этих дней YYYYMMDD, через запятую")
    args = parser.parse_args()

    meta_path = args.assets_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Нет {meta_path}. Сначала запустите ml/train.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    days = {d for d in args.days.split(",") if d} or None
    own = datasets.load_own(args.own_dir, only_days=days)
    if own is None:
        raise SystemExit(
            f"В {args.own_dir} нет записей. Проверять экспорт можно только "
            "на данных, снятых тем же приложением."
        )

    print(f"Загружено {len(own)} окон из собственных записей")
    print(f"Модель обучена на каналах: {', '.join(meta['channels'])}")

    evaluate_activity(meta, own, args.assets_dir)
    evaluate_placement(meta, own, args.assets_dir)


if __name__ == "__main__":
    main()
