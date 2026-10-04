"""Загрузка обучающих данных: публичный UCI HAR и собственные записи из приложения.

Два источника решают разные задачи.

UCI HAR — 30 человек, 50 Гц, окна 2.56 с с перекрытием 50 %, то есть ровно
та геометрия, которую использует приложение. Но в нём нет ни магнитометра,
ни датчиков приближения и освещённости, и нет классов «бег» и «велосипед».

Собственные записи из режима «Сбор данных» содержат все пять датчиков и
любую разметку, включая положение телефона, — без них модель положения
обучать не на чем. Зато их мало.

Отсюда правило: состав каналов и список классов определяются данными,
а не зашиты в код. То, что получилось, записывается в model_meta.json,
и приложение подстраивается под модель.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import features as feat
from features import estimate_gravity

SAMPLE_RATE_HZ = 50
WINDOW_SIZE = 128
WINDOW_STRIDE = 64

# На сколько отрезков дробится одна собственная запись при делении выборки.
# Три — компромисс: меньше даёт слишком грубое деление, больше делает
# отрезки короче окна у минутных записей.
OWN_SEGMENTS = 3
G = 9.80665

ALL_MOTION_CHANNELS = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
]

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

# Классы UCI HAR → классы приложения. Сидение, стояние и лежание
# с точки зрения акселерометра неразличимы по движению (его просто нет),
# поэтому все три сводятся к одному классу «покой».
UCI_LABEL_MAP = {
    1: "WALKING",
    2: "STAIRS_UP",
    3: "STAIRS_DOWN",
    4: "STILL",
    5: "STILL",
    6: "STILL",
}


@dataclass
class Dataset:
    """Нарезанные окна с разметкой.

    Attributes:
        x: (N, WINDOW_SIZE, C) — сигналы окон.
        context: (N, 16) — агрегированные признаки окна: освещённость,
            приближение, ориентация и статистики движения. Это второй вход
            модели активности; именно через него в решение попадают датчики,
            которых нет среди каналов движения.
        y_activity: (N,) — имена классов активности.
        y_placement: (N,) — имена классов положения телефона или None.
        subjects: (N,) — идентификатор испытуемого/записи; нужен, чтобы
            делить выборку по людям, а не случайно: окна одного человека
            перекрываются, и случайное деление даёт завышенную точность.
        channels: список имён каналов в порядке последней оси x.
    """

    x: np.ndarray
    context: np.ndarray
    y_activity: np.ndarray
    y_placement: np.ndarray | None
    subjects: np.ndarray
    channels: list[str]
    # Метка источника и позиции для каждого окна («rw:thigh», «uci:waist»,
    # «own:POCKET»). В обучении не участвует — нужна, чтобы разложить
    # итоговую точность по позициям: телефон носят не везде, и средняя
    # цифра по всем позициям скрывает, как модель работает в кармане.
    groups: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x)

    def summary(self) -> str:
        lines = [
            f"  окон: {len(self.x)}, каналов: {len(self.channels)} "
            f"({', '.join(self.channels)})",
            f"  испытуемых: {len(np.unique(self.subjects))}",
        ]
        vals, counts = np.unique(self.y_activity, return_counts=True)
        lines.append("  активности: " + ", ".join(f"{v}={c}" for v, c in zip(vals, counts)))
        if self.y_placement is not None:
            vals, counts = np.unique(self.y_placement, return_counts=True)
            lines.append("  положения:  " + ", ".join(f"{v}={c}" for v, c in zip(vals, counts)))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# UCI HAR
# --------------------------------------------------------------------------

def ensure_uci(root: Path) -> Path:
    """Возвращает путь к распакованному UCI HAR, при необходимости скачивая его."""
    target = root / "UCI HAR Dataset"
    if (target / "train" / "y_train.txt").exists():
        return target

    root.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание UCI HAR Dataset в {root} …")
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Нужен пакет requests (pip install -r ml/requirements.txt), "
            f"либо скачайте датасет вручную в {root}"
        ) from exc

    response = requests.get(UCI_URL, timeout=300)
    response.raise_for_status()

    outer = zipfile.ZipFile(io.BytesIO(response.content))
    # Архив с сайта UCI содержит вложенный zip с самим датасетом.
    inner_name = next((n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")), None)
    if inner_name:
        with outer.open(inner_name) as inner_file:
            zipfile.ZipFile(io.BytesIO(inner_file.read())).extractall(root)
    else:
        outer.extractall(root)

    if not (target / "train" / "y_train.txt").exists():
        raise SystemExit(f"Датасет распакован не туда, ожидался каталог {target}")
    return target


def _load_uci_split(base: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals_dir = base / split / "Inertial Signals"
    # total_acc — полное ускорение с гравитацией, как отдаёт Android.
    # body_acc из датасета нам не нужен: приложение его не измеряет,
    # а вычисляет само тем же способом.
    names = [
        ("total_acc_x", G), ("total_acc_y", G), ("total_acc_z", G),
        ("body_gyro_x", 1.0), ("body_gyro_y", 1.0), ("body_gyro_z", 1.0),
    ]
    channels = []
    for name, scale in names:
        path = signals_dir / f"{name}_{split}.txt"
        # Файлы большие (7352 × 128), np.loadtxt на них уходит десятки секунд.
        # Ускорение в UCI HAR выражено в единицах g — переводим в м/с².
        raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32).to_numpy()
        channels.append(raw * scale)

    x = np.stack(channels, axis=-1)  # (N, 128, 6)
    y = np.loadtxt(base / split / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(base / split / f"subject_{split}.txt", dtype=int)
    return x, y, subjects


def _context_from_arrays(
    x: np.ndarray,
    channels: list[str],
    light: np.ndarray | None = None,
    proximity: np.ndarray | None = None,
    proximity_near: np.ndarray | None = None,
    grav: np.ndarray | None = None,
) -> np.ndarray:
    """Считает контекстные признаки для пачки окон.

    Недостающие датчики не подменяются правдоподобными числами: их признаки
    остаются нулевыми, а соответствующий флаг `*_available` — нулём. Так
    модель учится опираться на освещённость и приближение только там, где
    они действительно измерены, а не запоминает выдуманную константу.
    """
    n, t, _ = x.shape
    idx = {c: channels.index(c) for c in channels}

    def ch(name: str) -> np.ndarray:
        return x[:, :, idx[name]] if name in idx else np.zeros((n, t), dtype=np.float32)

    acc = np.stack([ch("acc_x"), ch("acc_y"), ch("acc_z")], axis=-1)
    if grav is None:
        grav = feat.estimate_gravity_batch(acc)

    if light is None:
        light = np.full((n, t), -1.0, dtype=np.float32)
    if proximity is None:
        proximity = np.full((n, t), -1.0, dtype=np.float32)
    if proximity_near is None:
        proximity_near = np.zeros((n, t), dtype=bool)

    out = np.empty((n, feat.PLACEMENT_FEATURE_COUNT), dtype=np.float32)
    for i in range(n):
        out[i] = feat.placement_features({
            "acc_x": acc[i, :, 0], "acc_y": acc[i, :, 1], "acc_z": acc[i, :, 2],
            "gyro_x": ch("gyro_x")[i], "gyro_y": ch("gyro_y")[i], "gyro_z": ch("gyro_z")[i],
            "grav_x": grav[i, :, 0], "grav_y": grav[i, :, 1], "grav_z": grav[i, :, 2],
            "light_lux": light[i],
            "proximity_cm": proximity[i],
            "proximity_near": proximity_near[i],
        })
    return out


def load_uci(root: Path) -> Dataset:
    """Читает UCI HAR целиком (train + test) и переводит метки в классы приложения."""
    base = ensure_uci(root)
    xs, ys, ss = [], [], []
    for split in ("train", "test"):
        x, y, s = _load_uci_split(base, split)
        xs.append(x)
        ys.append(y)
        ss.append(s)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    subjects = np.concatenate(ss)

    labels = np.array([UCI_LABEL_MAP[int(v)] for v in y])
    # Префикс в идентификаторе испытуемого: при объединении с собственными
    # записями номера не должны пересечься.
    subjects = np.array([f"uci{int(s)}" for s in subjects])

    channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    print("  расчёт контекстных признаков …")
    # В UCI HAR нет ни освещённости, ни приближения, ни магнитометра:
    # соответствующие признаки будут нулями с нулевым флагом доступности.
    context = _context_from_arrays(x, channels)

    return Dataset(
        x=x.astype(np.float32),
        context=context,
        y_activity=labels,
        y_placement=None,
        subjects=subjects,
        channels=channels,
        groups=np.full(len(x), "uci:waist"),
    )


# --------------------------------------------------------------------------
# RealWorld HAR (Университет Мангейма, 2016)
# --------------------------------------------------------------------------

REALWORLD_URL = (
    "https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/"
    "Projects/sensor/realworld/s{subject}/data/{sensor}_{activity}_csv.zip"
)

REALWORLD_SUBJECTS = 15

# Сколько раз повторять скачивание архива при сетевом сбое.
REALWORLD_RETRIES = 3

# Датчик → приставка в имени архива и в имени CSV внутри него.
REALWORLD_SENSORS = {
    "acc": "acc",
    "gyr": "Gyroscope",
    "mag": "MagneticField",
    "lig": "Light",
}

REALWORLD_ACTIVITY_MAP = {
    "walking": "WALKING",
    "running": "RUNNING",
    "climbingup": "STAIRS_UP",
    "climbingdown": "STAIRS_DOWN",
    # Сидение, стояние и лежание различаются позой, а не движением;
    # приложение их не разделяет, поэтому все три — «покой».
    "sitting": "STILL",
    "standing": "STILL",
    "lying": "STILL",
    # jumping пропущен: соответствующего класса в приложении нет.
}

# Позиции на теле, в которых телефон реально носят.
# shin (голень) исключена: телефон туда не кладут, и её данные
# только размыли бы обучение под нашу задачу.
REALWORLD_POSITIONS = ["thigh", "forearm", "waist", "chest", "upperarm", "head"]

# Позиция → класс положения телефона.
# forearm намеренно не сопоставлен: устройство на предплечье не имеет
# датчика освещённости, и признак «света нет» стал бы идеальным
# предсказателем класса «в руке» — модель выучила бы артефакт разметки,
# а не физику. Класс IN_HAND остаётся за собственными записями.
REALWORLD_PLACEMENT_MAP = {
    "thigh": "POCKET",
    "head": "AT_EAR",
}


def _resample_hold(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Приводит нерегулярный ряд к равномерной сетке методом sample-and-hold.

    Берётся последнее значение, известное к моменту отсчёта сетки, — ровно
    так же, как `SensorHub` на устройстве подставляет показания медленных
    датчиков. Интерполяция была бы точнее, но создала бы данные, которых
    приложение никогда не увидит.
    """
    idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
    return values[idx]


def _download_realworld_part(root: Path, subject: int, sensor: str, activity: str) -> Path | None:
    """Скачивает и распаковывает один архив, если его ещё нет в кэше."""
    target = root / f"s{subject}" / f"{sensor}_{activity}"
    if target.is_dir() and any(target.glob("*.csv")):
        return target

    import requests

    url = REALWORLD_URL.format(subject=subject, sensor=sensor, activity=activity)
    # Сервер изредка рвёт соединение на большом архиве. Без повтора один
    # сетевой сбой молча выкидывает целую запись из обучающей выборки,
    # и заметить это потом почти невозможно.
    response = None
    for attempt in range(1, REALWORLD_RETRIES + 1):
        try:
            response = requests.get(url, timeout=600)
            response.raise_for_status()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == REALWORLD_RETRIES:
                print(f"    не скачан {sensor}_{activity} s{subject} "
                      f"({REALWORLD_RETRIES} попыт.): {exc}")
                return None
            time.sleep(2 * attempt)
    if response is None:
        return None

    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    (target / Path(name).name).write_bytes(z.read(name))
    except zipfile.BadZipFile:
        print(f"    битый архив {sensor}_{activity} s{subject}")
        return None
    return target


def _read_realworld_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Читает CSV RealWorld: id, attr_time, затем attr_x/y/z либо attr_light."""
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if "attr_time" not in df.columns:
        return None
    cols = [c for c in ("attr_x", "attr_y", "attr_z", "attr_light") if c in df.columns]
    if not cols:
        return None
    times = df["attr_time"].to_numpy(dtype=np.int64)
    values = df[cols].to_numpy(dtype=np.float32)
    # Файлы изредка содержат неупорядоченные строки — searchsorted этого не прощает.
    order = np.argsort(times, kind="stable")
    return times[order], values[order]


def load_realworld(
    root: Path,
    subjects: int = REALWORLD_SUBJECTS,
    positions: list[str] | None = None,
) -> Dataset | None:
    """Загружает RealWorld HAR: 15 человек, 7 позиций на теле, все пять каналов.

    Ценность датасета для нас в двух вещах. Во-первых, здесь есть датчик
    освещённости — единственный публичный источник, где он записан вместе
    с движением. Во-вторых, одна и та же активность снята одновременно
    в нескольких местах на теле, а значит появляются пары «одна активность —
    разные положения», ради которых и сделана контекстная ветка модели.

    Данные качаются по одному архиву на связку (человек, датчик, активность)
    и кэшируются в [root]; повторный запуск ничего не скачивает заново.
    """
    positions = positions or REALWORLD_POSITIONS
    root.mkdir(parents=True, exist_ok=True)

    windows, activities, placements, subject_ids, group_ids = [], [], [], [], []

    for subject in range(1, subjects + 1):
        for activity, label in REALWORLD_ACTIVITY_MAP.items():
            parts = {}
            for sensor in REALWORLD_SENSORS:
                part = _download_realworld_part(root, subject, sensor, activity)
                if part is not None:
                    parts[sensor] = part
            if "acc" not in parts:
                continue

            for position in positions:
                series = {}
                for sensor, prefix in REALWORLD_SENSORS.items():
                    if sensor not in parts:
                        continue
                    path = parts[sensor] / f"{prefix}_{activity}_{position}.csv"
                    if not path.exists():
                        continue
                    read = _read_realworld_csv(path)
                    if read is not None:
                        series[sensor] = read

                # Акселерометр и гироскоп обязательны: они входят в общий
                # набор каналов, и подставить вместо гироскопа нули значило бы
                # обучать сеть на «ходьбе без единого поворота» — сигнале,
                # которого на реальном устройстве не бывает. Если архив
                # не скачался, запись пропускается целиком.
                if "acc" not in series or "gyr" not in series:
                    continue

                # Общая сетка 50 Гц на пересечении интервалов всех датчиков.
                start = max(t[0] for t, _ in series.values())
                end = min(t[-1] for t, _ in series.values())
                if end - start < WINDOW_SIZE * 20:
                    continue
                grid = np.arange(start, end, 1000 // SAMPLE_RATE_HZ, dtype=np.int64)

                acc = _resample_hold(*series["acc"], grid)
                gyro = _resample_hold(*series["gyr"], grid)
                # Магнитометр — единственный необязательный из трёх: он всё
                # равно выпадает при пересечении каналов с UCI HAR, где его нет.
                mag = (_resample_hold(*series["mag"], grid) if "mag" in series
                       else np.zeros_like(acc))
                # Отсутствующий датчик освещённости обозначается −1,
                # как это делает SensorHub, а не нулём: ноль означал бы темноту.
                light = (_resample_hold(*series["lig"], grid)[:, 0] if "lig" in series
                         else np.full(len(grid), -1.0, dtype=np.float32))

                grav = estimate_gravity(acc.T).T
                n = len(grid)
                data = {
                    "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
                    "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
                    "mag_x": mag[:, 0], "mag_y": mag[:, 1], "mag_z": mag[:, 2],
                    "grav_x": grav[:, 0], "grav_y": grav[:, 1], "grav_z": grav[:, 2],
                    "light_lux": light.astype(np.float32),
                    # Датчика приближения в датасете нет вообще.
                    "proximity_cm": np.full(n, -1.0, dtype=np.float32),
                    "proximity_near": np.zeros(n, dtype=bool),
                }

                placement = REALWORLD_PLACEMENT_MAP.get(position, "UNKNOWN")
                for s in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
                    sl = slice(s, s + WINDOW_SIZE)
                    windows.append({k: v[sl] for k, v in data.items()})
                    activities.append(label)
                    placements.append(placement)
                    # Испытуемый — человек, а не позиция: все позиции одного
                    # человека обязаны остаться по одну сторону от деления.
                    subject_ids.append(f"rw{subject}")
                    group_ids.append(f"rw:{position}")

        print(f"  испытуемый s{subject}: всего окон {len(windows)}")

    if not windows:
        return None

    x = np.stack([
        np.stack([w[ch] for ch in ALL_MOTION_CHANNELS], axis=-1) for w in windows
    ]).astype(np.float32)
    context = np.stack([feat.placement_features(w) for w in windows]).astype(np.float32)

    ds = Dataset(
        x=x,
        context=context,
        y_activity=np.array(activities),
        y_placement=np.array(placements),
        subjects=np.array(subject_ids),
        channels=list(ALL_MOTION_CHANNELS),
        groups=np.array(group_ids),
    )
    ds.raw_windows = windows  # type: ignore[attr-defined]
    return ds


# --------------------------------------------------------------------------
# Собственные записи из приложения
# --------------------------------------------------------------------------

def load_own(directory: Path, parts: int = OWN_SEGMENTS) -> Dataset | None:
    """Читает CSV-файлы, записанные режимом «Сбор данных».

    Каждый файл — одна непрерывная запись с постоянной разметкой.
    Окна нарезаются внутри файла, поэтому окно никогда не пересекает
    границу между двумя разными активностями.

    Каждая запись дополнительно делится на [parts] последовательных отрезков,
    и каждый отрезок считается отдельным «испытуемым». Без этого класс,
    записанный одним файлом, целиком уходил бы либо в обучение, либо в тест:
    в первом случае по нему нет оценки, во втором модель его не видела.
    Окна, попадающие на границу отрезков, отбрасываются — иначе соседние
    отрезки делили бы общие отсчёты и тест переставал быть независимым.
    """
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.csv"))
    if not files:
        return None

    windows, activities, placements, subjects, group_ids = [], [], [], [], []

    for path in files:
        df = pd.read_csv(path)
        if len(df) < WINDOW_SIZE:
            print(f"  пропущен {path.name}: всего {len(df)} отсчётов")
            continue

        required = set(ALL_MOTION_CHANNELS) | {
            "proximity_cm", "light_lux", "activity", "placement",
        }
        if not required.issubset(df.columns):
            print(f"  пропущен {path.name}: нет колонок {required - set(df.columns)}")
            continue

        data = {ch: df[ch].to_numpy(dtype=np.float32) for ch in ALL_MOTION_CHANNELS}
        # Приложение пишет оценку гравитации, но старые файлы могут её не иметь.
        if "grav_x" in df.columns:
            grav = np.stack([df[f"grav_{a}"].to_numpy(dtype=np.float32) for a in "xyz"])
        else:
            acc = np.stack([data[f"acc_{a}"] for a in "xyz"])
            grav = estimate_gravity(acc)
        for i, axis in enumerate("xyz"):
            data[f"grav_{axis}"] = grav[i].astype(np.float32)

        data["proximity_cm"] = df["proximity_cm"].to_numpy(dtype=np.float32)
        data["light_lux"] = df["light_lux"].to_numpy(dtype=np.float32)
        # Признак «датчик перекрыт» приложение считает по maximumRange
        # конкретного устройства и пишет готовым — воспроизводить этот порог
        # здесь по одному окну значило бы разойтись с тем, что видит модель
        # на телефоне.
        if "proximity_near" in df.columns:
            data["proximity_near"] = df["proximity_near"].to_numpy().astype(bool)

        activity = str(df["activity"].iloc[0])
        placement = str(df["placement"].iloc[0])
        n = len(df)

        segment_len = max(WINDOW_SIZE, -(-n // max(1, parts)))
        kept = 0
        for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
            segment = start // segment_len
            # Окно целиком должно лежать внутри своего отрезка.
            if (start + WINDOW_SIZE - 1) // segment_len != segment:
                continue
            sl = slice(start, start + WINDOW_SIZE)
            windows.append({k: v[sl] for k, v in data.items()})
            activities.append(activity)
            placements.append(placement)
            subjects.append(f"{path.stem}#{segment}")
            group_ids.append(f"own:{placement}")
            kept += 1
        print(f"  {path.name}: {kept} окон, отрезков {min(parts, -(-n // segment_len))}")

    if not windows:
        return None

    x = np.stack([
        np.stack([w[ch] for ch in ALL_MOTION_CHANNELS], axis=-1) for w in windows
    ]).astype(np.float32)

    # Здесь измерены все пять датчиков, поэтому контекст считается
    # прямо из окна — без заглушек, в отличие от UCI HAR.
    context = np.stack([feat.placement_features(w) for w in windows]).astype(np.float32)

    ds = Dataset(
        x=x,
        context=context,
        y_activity=np.array(activities),
        y_placement=np.array(placements),
        subjects=np.array(subjects),
        channels=list(ALL_MOTION_CHANNELS),
        groups=np.array(group_ids),
    )
    # Полные окна нужны отдельно: признаки положения телефона считаются
    # по датчикам, которых нет среди каналов движения.
    ds.raw_windows = windows  # type: ignore[attr-defined]
    return ds


# --------------------------------------------------------------------------
# Объединение источников
# --------------------------------------------------------------------------

def merge(datasets: list[Dataset]) -> Dataset:
    """Склеивает наборы, оставляя каналы, которые есть во всех источниках.

    Пересечение, а не объединение: подмешивать нулевой магнитометр там,
    где его не измеряли, — значит учить сеть на артефакте, который
    на устройстве никогда не встретится.
    """
    datasets = [d for d in datasets if d is not None and len(d) > 0]
    if not datasets:
        raise SystemExit("Нет ни одного источника данных")
    if len(datasets) == 1:
        return datasets[0]

    common = [ch for ch in ALL_MOTION_CHANNELS if all(ch in d.channels for d in datasets)]
    if not common:
        raise SystemExit("У источников нет общих каналов")

    xs, ctx, ya, yp, ss, gs = [], [], [], [], [], []
    for d in datasets:
        idx = [d.channels.index(ch) for ch in common]
        xs.append(d.x[:, :, idx])
        # Контекст не зависит от того, какие каналы движения уцелели
        # при пересечении: он посчитан из полного окна каждого источника.
        ctx.append(d.context)
        ya.append(d.y_activity)
        ss.append(d.subjects)
        gs.append(d.groups if d.groups is not None else np.full(len(d), "unknown"))
        yp.append(
            d.y_placement if d.y_placement is not None
            else np.full(len(d), "UNKNOWN")
        )

    return Dataset(
        x=np.concatenate(xs).astype(np.float32),
        context=np.concatenate(ctx).astype(np.float32),
        y_activity=np.concatenate(ya),
        y_placement=np.concatenate(yp),
        subjects=np.concatenate(ss),
        channels=common,
        groups=np.concatenate(gs),
    )


def split_by_subject(
    dataset: Dataset,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Делит выборку по испытуемым на train / val / test.

    Два независимых принципа.

    Первый: деление именно по людям. Соседние окна перекрываются на 50 %,
    и при случайном делении почти каждое тестовое окно имеет near-duplicate
    в обучении — точность выходит 97–99 % и не имеет отношения к работе
    на новом человеке.

    Второй: валидация и тест — разные люди. Валидация используется для
    ранней остановки и выбора лучшей эпохи, то есть модель косвенно
    подстраивается под неё. Если отчитываться на той же выборке, итоговая
    цифра окажется завышенной. Тестовые испытуемые не участвуют в обучении
    вообще никак и смотрятся ровно один раз — в финальном отчёте.

    Returns:
        индексы (train, val, test)
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(dataset.subjects)
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_fraction)))
    n_val = max(1, int(round(len(subjects) * val_fraction)))

    # При совсем малом числе испытуемых (собственные записи) на обучение
    # может ничего не остаться — тогда жертвуем валидацией, а не тестом.
    if n_test + n_val >= len(subjects):
        n_val = max(0, len(subjects) - n_test - 1)

    test_subjects = set(subjects[:n_test].tolist())
    val_subjects = set(subjects[n_test:n_test + n_val].tolist())

    which = np.array([
        2 if s in test_subjects else 1 if s in val_subjects else 0
        for s in dataset.subjects
    ])
    return np.where(which == 0)[0], np.where(which == 1)[0], np.where(which == 2)[0]
