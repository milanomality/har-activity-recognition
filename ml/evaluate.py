"""Проверка экспортированных .tflite на собственных записях.

Обучение и экспорт — разные вещи: модель может учиться прекрасно, а после
конвертации в TFLite давать другие числа (из-за слияния слоёв, других
реализаций операций, потери точности при квантизации). Этот скрипт
прогоняет именно тот файл, который поедет в APK, и печатает его метрики.

Запуск:
    python ml/evaluate.py --own-dir ml/data/own
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


def run_tflite(model_path: Path, x: np.ndarray) -> np.ndarray:
    """Прогоняет батч через TFLite-интерпретатор по одному примеру.

    Экспортированная модель имеет фиксированный батч 1 — ровно так же,
    как её вызывает приложение, поэтому и проверяем в том же режиме.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    predictions = []
    for sample in x:
        interpreter.set_tensor(inp["index"], sample[np.newaxis, ...].astype(np.float32))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(out["index"])[0])
    return np.stack(predictions)


def evaluate_activity(meta: dict, own: datasets.Dataset, assets: Path) -> None:
    model_path = assets / "activity_model.tflite"
    if not model_path.exists():
        print(f"Нет {model_path.name} — сначала запустите ml/train.py")
        return

    labels = meta["activity_labels"]
    channels = meta["channels"]

    # Берём из записи ровно те каналы, на которых обучалась модель,
    # в том же порядке — именно это делает приложение по model_meta.json.
    idx = [own.channels.index(ch) for ch in channels]
    x = own.x[:, :, idx]

    mean = np.array(meta["channel_mean"], dtype=np.float32)
    std = np.array(meta["channel_std"], dtype=np.float32)
    x = (x - mean) / std

    known = np.array([a in labels for a in own.y_activity])
    if not known.any():
        print("В записях нет классов, которые знает модель")
        return
    if not known.all():
        skipped = sorted(set(own.y_activity[~known]))
        print(f"Пропущены классы, отсутствующие в модели: {', '.join(skipped)}")

    y_true = np.array([labels.index(a) for a in own.y_activity[known]])
    y_pred = run_tflite(model_path, x[known]).argmax(axis=1)

    print("\n=== Активность: .tflite на собственных записях ===")
    _report(y_true, y_pred, labels)


def evaluate_placement(meta: dict, own: datasets.Dataset, assets: Path) -> None:
    model_path = assets / "placement_model.tflite"
    if not model_path.exists():
        print(f"\nНет {model_path.name} — положение телефона определяется правилами")
        return
    if own.y_placement is None or not hasattr(own, "raw_windows"):
        return

    labels = meta["placement_labels"]
    x = np.stack([feat.placement_features(w) for w in own.raw_windows])
    mean = np.array(meta["placement_mean"], dtype=np.float32)
    std = np.array(meta["placement_std"], dtype=np.float32)
    x = (x - mean) / std

    known = np.array([p in labels for p in own.y_placement])
    if not known.any():
        return

    y_true = np.array([labels.index(p) for p in own.y_placement[known]])
    y_pred = run_tflite(model_path, x[known]).argmax(axis=1)

    print("\n=== Положение телефона: .tflite на собственных записях ===")
    _report(y_true, y_pred, labels)


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
    args = parser.parse_args()

    meta_path = args.assets_dir / "model_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Нет {meta_path}. Сначала запустите ml/train.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    own = datasets.load_own(args.own_dir)
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
