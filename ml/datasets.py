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
# Общая нарезка записей для публичных датасетов
# --------------------------------------------------------------------------

class _WindowCollector:
    """Копит окна из непрерывных записей и собирает из них Dataset.

    Каждая запись — один человек, одна активность, одно положение, 50 Гц,
    единицы и оси Android (ускорение в м/с² вместе с гравитацией, рад/с, мкТл).
    Окна нарезаются внутри записи и никогда не пересекают её границу.
    Освещённости и приближения в этих датасетах нет: они обозначаются −1,
    как это делает SensorHub при отсутствии датчика.
    """

    def __init__(self, with_mag: bool):
        self.with_mag = with_mag
        self.windows: list[dict] = []
        self.activities: list[str] = []
        self.placements: list[str] = []
        self.subjects: list[str] = []
        self.groups: list[str] = []

    def add(self, acc: np.ndarray, gyro: np.ndarray, mag: np.ndarray | None,
            activity: str, placement: str, subject: str, group: str) -> None:
        n = len(acc)
        if n < WINDOW_SIZE:
            return
        grav = estimate_gravity(acc.T).T
        if mag is None:
            mag = np.zeros_like(acc)
        data = {
            "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
            "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
            "mag_x": mag[:, 0], "mag_y": mag[:, 1], "mag_z": mag[:, 2],
            "grav_x": grav[:, 0], "grav_y": grav[:, 1], "grav_z": grav[:, 2],
            "light_lux": np.full(n, -1.0, dtype=np.float32),
            "proximity_cm": np.full(n, -1.0, dtype=np.float32),
            "proximity_near": np.zeros(n, dtype=bool),
        }
        data = {k: np.asarray(v, dtype=bool if k == "proximity_near" else np.float32)
                for k, v in data.items()}
        for s in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
            sl = slice(s, s + WINDOW_SIZE)
            self.windows.append({k: v[sl] for k, v in data.items()})
            self.activities.append(activity)
            self.placements.append(placement)
            self.subjects.append(subject)
            self.groups.append(group)

    def build(self) -> Dataset | None:
        if not self.windows:
            return None
        channels = list(ALL_MOTION_CHANNELS) if self.with_mag else ALL_MOTION_CHANNELS[:6]
        x = np.stack([
            np.stack([w[ch] for ch in channels], axis=-1) for w in self.windows
        ]).astype(np.float32)
        context = np.stack([feat.placement_features(w) for w in self.windows]).astype(np.float32)
        return Dataset(
            x=x,
            context=context,
            y_activity=np.array(self.activities),
            y_placement=np.array(self.placements),
            subjects=np.array(self.subjects),
            channels=channels,
            groups=np.array(self.groups),
        )


def _download(url: str, target: Path, retries: int = 3) -> Path | None:
    """Скачивает файл в target, если его там ещё нет. Повторяет при сбое сети."""
    if target.exists() and target.stat().st_size > 0:
        return target
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                print(f"    не скачан {url}: {exc}")
                return None
            time.sleep(2 * attempt)
    return None


# --------------------------------------------------------------------------
# MotionSense (Queen Mary University of London, 2018)
# --------------------------------------------------------------------------

MOTIONSENSE_URL = (
    "https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/"
    "A_DeviceMotion_data.zip"
)

MOTIONSENSE_ACTIVITY_MAP = {
    "dws": "STAIRS_DOWN",
    "ups": "STAIRS_UP",
    "wlk": "WALKING",
    "jog": "RUNNING",
    "sit": "STILL",
    "std": "STILL",
}

# Начало и конец каждой записи — телефон кладут в карман и достают.
MOTIONSENSE_TRIM = 2 * SAMPLE_RATE_HZ


def load_motionsense(root: Path) -> Dataset | None:
    """Загружает MotionSense: 24 человека, iPhone 6s в переднем кармане брюк, 50 Гц.

    CoreMotion делит ускорение на гравитацию и ускорение пользователя, оба в g,
    и знак у него противоположный Android: лежащий экраном вверх iPhone даёт
    гравитацию (0, 0, −1). Отсюда перевод в соглашение Android:
    `acc = −(gravity + userAcceleration) · g`. Гироскоп — те же правые оси
    в рад/с, перевода не требует. Магнитометра в этой части датасета нет.
    """
    archive = _download(MOTIONSENSE_URL, root / "A_DeviceMotion_data.zip")
    if archive is None:
        return None

    collector = _WindowCollector(with_mag=False)
    with zipfile.ZipFile(archive) as z:
        names = sorted(
            n for n in z.namelist()
            if n.endswith(".csv") and "__MACOSX" not in n and "/sub_" in n
        )
        for name in names:
            folder, file = name.split("/")[-2:]
            act = folder.split("_")[0]
            label = MOTIONSENSE_ACTIVITY_MAP.get(act)
            if label is None:
                continue
            subject = file.removeprefix("sub_").removesuffix(".csv")
            df = pd.read_csv(io.BytesIO(z.read(name)))
            gravity = df[["gravity.x", "gravity.y", "gravity.z"]].to_numpy(np.float32)
            user = df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].to_numpy(np.float32)
            gyro = df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].to_numpy(np.float32)
            acc = -(gravity + user) * G
            sl = slice(MOTIONSENSE_TRIM, len(df) - MOTIONSENSE_TRIM)
            collector.add(acc[sl], gyro[sl], None, label, "POCKET", f"ms{subject}", "ms:pocket")

    ds = collector.build()
    if ds is not None:
        print(f"  MotionSense: {len(ds)} окон")
    return ds


# --------------------------------------------------------------------------
# Shoaib et al. (Университет Твенте, 2014)
# --------------------------------------------------------------------------

SHOAIB_URL = (
    "https://www.utwente.nl/en/eemcs/ps/dataset-folder/"
    "sensors-activity-recognition-dataset-shoaib.rar"
)

# Порядок блоков в строке CSV (не тот, что в readme): 14 колонок на позицию —
# время, Ax..Az, Lx..Lz (линейное), Gx..Gz, Mx..Mz, пустая.
SHOAIB_POSITIONS = ["left_pocket", "right_pocket", "wrist", "upper_arm", "belt"]
SHOAIB_LABEL_COLUMN = 69

SHOAIB_PLACEMENT_MAP = {
    "left_pocket": "POCKET",
    "right_pocket": "POCKET",
    # Запястье, плечо и пояс не соответствуют ни одному положению приложения,
    # но для модели активности это полезное разнообразие мест на теле.
}

SHOAIB_ACTIVITY_MAP = {
    "walking": "WALKING",
    # В статье «бег», но по readme это трусца — для приложения это бег.
    "jogging": "RUNNING",
    "sitting": "STILL",
    "standing": "STILL",
    "biking": "CYCLING",
    "upstairs": "STAIRS_UP",
    "upsatirs": "STAIRS_UP",  # опечатка в файле участника 8
    "downstairs": "STAIRS_DOWN",
}

# Гироскоп в карманах изредка упирается в предел шкалы ±10 рад/с,
# магнитометр рядом с металлом показывает до 280 мкТл. Обрезаем до
# правдоподобного диапазона, чтобы единичные выбросы не портили нормировку.
SHOAIB_GYRO_LIMIT = 10.0
SHOAIB_MAG_LIMIT = 100.0


def _extract_rar(archive: Path, target: Path) -> bool:
    """Распаковывает RAR внешним 7-Zip: в стандартной библиотеке Python RAR нет."""
    import shutil
    import subprocess

    candidates = [shutil.which("7z"), shutil.which("7za"),
                  r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]
    exe = next((c for c in candidates if c and Path(c).exists()), None)
    if exe is None:
        return False
    target.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([exe, "x", "-y", f"-o{target}", str(archive)],
                            capture_output=True, text=True)
    return result.returncode == 0


def load_shoaib(root: Path) -> Dataset | None:
    """Загружает датасет Shoaib et al.: 10 человек, 5 мест на теле, 50 Гц, Galaxy S II.

    Значения — сырые события датчиков Android: ускорение в м/с² с гравитацией,
    гироскоп в рад/с, поле в мкТл, перевод осей не нужен. Все пять позиций
    записаны одновременно, строка к строке. Активности идут блоками по 9000
    строк; окна режутся внутри непрерывного отрезка одной метки.
    Телефон в карманах висел верхом вниз, на поясе — горизонтально.
    """
    data_dir = root / "DataSet"
    files = sorted(data_dir.glob("Participant_*.csv"))
    if not files:
        archive = _download(SHOAIB_URL, root / "shoaib.rar")
        if archive is None:
            return None
        if not _extract_rar(archive, root):
            print(f"  Не удалось распаковать {archive}: нужен 7-Zip. "
                  f"Распакуйте архив вручную так, чтобы файлы лежали в {data_dir}")
            return None
        files = sorted(data_dir.glob("Participant_*.csv"))

    collector = _WindowCollector(with_mag=True)
    for path in files:
        subject = path.stem.split("_")[-1]
        df = pd.read_csv(path, skiprows=2, header=None, encoding="latin-1", low_memory=False)
        labels = df[SHOAIB_LABEL_COLUMN].astype(str).str.strip().str.lower().to_numpy()
        # Границы непрерывных отрезков одной метки.
        bounds = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(labels)]])
        for k, position in enumerate(SHOAIB_POSITIONS):
            b = 14 * k
            block = df.iloc[:, b + 1:b + 13].to_numpy(np.float32)
            acc = block[:, 0:3]
            gyro = np.clip(block[:, 6:9], -SHOAIB_GYRO_LIMIT, SHOAIB_GYRO_LIMIT)
            mag = np.clip(block[:, 9:12], -SHOAIB_MAG_LIMIT, SHOAIB_MAG_LIMIT)
            placement = SHOAIB_PLACEMENT_MAP.get(position, "UNKNOWN")
            for s, e in zip(starts, ends):
                label = SHOAIB_ACTIVITY_MAP.get(labels[s])
                if label is None:
                    continue
                collector.add(acc[s:e], gyro[s:e], mag[s:e], label, placement,
                              f"sh{subject}", f"sh:{position}")
        print(f"  Shoaib, участник {subject}: всего окон {len(collector.windows)}")

    return collector.build()


# --------------------------------------------------------------------------
# ExtraSensory (UC San Diego, 2017)
# --------------------------------------------------------------------------

EXTRASENSORY_BASE = "http://extrasensory.ucsd.edu/data/"
EXTRASENSORY_LABELS_URL = EXTRASENSORY_BASE + "primary_data_files/ExtraSensory.per_uuid_features_labels.zip"
EXTRASENSORY_FOLDS_URL = EXTRASENSORY_BASE + "cv5Folds.zip"
EXTRASENSORY_RAW = {
    "acc": (EXTRASENSORY_BASE + "raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip", "m_raw_acc"),
    "gyro": (EXTRASENSORY_BASE + "raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip", "m_proc_gyro"),
}

# Метка → класс приложения. Транспорт и лифт в выборку не идут вовсе.
EXTRASENSORY_ACTIVITY_MAP = {
    "label:FIX_walking": "WALKING",
    "label:FIX_running": "RUNNING",
    "label:STAIRS_-_GOING_UP": "STAIRS_UP",
    "label:STAIRS_-_GOING_DOWN": "STAIRS_DOWN",
    "label:BICYCLING": "CYCLING",
    "label:SITTING": "STILL",
    "label:LYING_DOWN": "STILL",
    "label:OR_standing": "STILL",
}
EXTRASENSORY_EXCLUDE = ["label:IN_A_CAR", "label:ON_A_BUS", "label:ELEVATOR", "label:DRIVE_-_I_M_THE_DRIVER",
                        "label:DRIVE_-_I_M_A_PASSENGER"]
EXTRASENSORY_PLACEMENT_MAP = {
    "label:PHONE_IN_POCKET": "POCKET",
    "label:PHONE_IN_HAND": "IN_HAND",
    "label:PHONE_ON_TABLE": "ON_TABLE",
    # PHONE_IN_BAG не соответствует ни одному положению приложения.
}
# Метки «у уха» в датасете нет. Есть признак состояния телефона «идёт разговор»;
# вместе с закрытым датчиком приближения это и есть телефон у уха.
EXTRASENSORY_ON_PHONE = "discrete:on_the_phone:is_True"

# Сколько минут одного сочетания (активность, положение) брать у одного человека.
EXTRASENSORY_MAX_MINUTES = 15
EXTRASENSORY_WORKERS = 16


def _zip_index(url: str, cache: Path) -> dict[str, tuple[int, int, int]]:
    """Оглавление удалённого zip: имя → (смещение заголовка, сжатый размер, метод).

    Архивы ExtraSensory весят 6–10 ГБ, а нужна малая часть файлов. Оглавление
    читается один раз через HTTP Range и кэшируется; дальше каждый файл
    достаётся отдельным запросом.
    """
    import pickle

    if cache.exists():
        return pickle.loads(cache.read_bytes())
    import requests

    class _HttpFile(io.RawIOBase):
        def __init__(self):
            self.size = int(requests.head(url, timeout=60).headers["Content-Length"])
            self.pos = 0

        def seekable(self): return True
        def readable(self): return True
        def tell(self): return self.pos

        def seek(self, offset, whence=0):
            self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
            return self.pos

        def readinto(self, b):
            if self.pos >= self.size or len(b) == 0:
                return 0
            end = min(self.pos + len(b), self.size) - 1
            data = requests.get(url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=600).content
            b[:len(data)] = data
            self.pos += len(data)
            return len(data)

    with zipfile.ZipFile(io.BufferedReader(_HttpFile(), buffer_size=1 << 22)) as z:
        index = {i.filename: (i.header_offset, i.compress_size, i.compress_type) for i in z.infolist()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(index))
    return index


def _zip_member(url: str, entry: tuple[int, int, int]) -> bytes:
    """Достаёт один файл из удалённого zip одним Range-запросом."""
    import struct
    import zlib

    import requests

    offset, size, method = entry
    # Локальный заголовок — 30 байт плюс имя и extra; с запасом на их длину.
    raw = requests.get(url, headers={"Range": f"bytes={offset}-{offset + 30 + 1024 + size}"},
                       timeout=600).content
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    start = 30 + name_len + extra_len
    data = raw[start:start + size]
    if method == zipfile.ZIP_STORED:
        return data
    return zlib.decompressobj(-15).decompress(data)


def _read_es_dat(raw: bytes) -> np.ndarray | None:
    """Файл минуты: строки «время x y z» через пробел. None — пусто или nan."""
    try:
        arr = np.loadtxt(io.StringIO(raw.decode("ascii", "ignore")), dtype=np.float64, ndmin=2)
    except ValueError:
        return None
    if arr.shape[1] < 4 or len(arr) < 10:
        return None
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) < 10:
        return None
    return arr[np.argsort(arr[:, 0], kind="stable")]


def _es_platforms(root: Path) -> dict[str, str]:
    """UUID → 'android' | 'iphone' по спискам из cv5Folds.zip."""
    archive = _download(EXTRASENSORY_FOLDS_URL, root / "cv5Folds.zip")
    out: dict[str, str] = {}
    if archive is None:
        return out
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            for platform in ("android", "iphone"):
                if name.endswith(f"_{platform}_uuids.txt"):
                    for line in z.read(name).decode().split():
                        out[line.strip()] = platform
    return out


def _es_select_minutes(df: pd.DataFrame, android: bool, max_minutes: int, rng) -> list[tuple]:
    """Выбирает минуты с однозначной активностью: (время, активность, положение, лк, близко)."""
    act_cols = [c for c in EXTRASENSORY_ACTIVITY_MAP if c in df.columns]
    acts = df[act_cols].fillna(0).to_numpy() == 1
    # Сидя, стоя и лёжа — всё «покой»: несколько таких меток не противоречат друг другу.
    classes = np.array([EXTRASENSORY_ACTIVITY_MAP[c] for c in act_cols])
    excluded = np.zeros(len(df), dtype=bool)
    for c in EXTRASENSORY_EXCLUDE:
        if c in df.columns:
            excluded |= df[c].fillna(0).to_numpy() == 1

    on_phone = (df[EXTRASENSORY_ON_PHONE].fillna(0).to_numpy() == 1
                if EXTRASENSORY_ON_PHONE in df.columns else np.zeros(len(df), dtype=bool))
    light = df.get("lf_measurements:light")
    prox = df.get("lf_measurements:proximity_cm")

    chosen: dict[tuple[str, str], list[tuple]] = {}
    for i in range(len(df)):
        if excluded[i]:
            continue
        labels = set(classes[acts[i]])
        if len(labels) != 1:
            continue
        activity = labels.pop()

        lux = -1.0
        near = None
        if android:
            # У Android свет записан в логарифме: lux = exp(v).
            if light is not None and np.isfinite(light.iloc[i]):
                lux = float(np.exp(light.iloc[i]))
            # Приближение — 0 (закрыт) или 8 см (максимум, открыт).
            if prox is not None and np.isfinite(prox.iloc[i]):
                near = bool(prox.iloc[i] < 4)

        placement = "UNKNOWN"
        for col, cls in EXTRASENSORY_PLACEMENT_MAP.items():
            if col in df.columns and df[col].iloc[i] == 1:
                placement = cls
                break
        # Разговор при закрытом датчике — телефон у уха. У iPhone датчик
        # приближения в датасете неинформативен, поэтому только Android.
        if on_phone[i] and near is True and placement != "ON_TABLE":
            placement = "AT_EAR"

        chosen.setdefault((activity, placement), []).append(
            (int(df["timestamp"].iloc[i]), activity, placement, lux, near)
        )

    out = []
    for items in chosen.values():
        if len(items) > max_minutes:
            items = [items[j] for j in rng.choice(len(items), max_minutes, replace=False)]
        out.extend(items)
    return out


def _es_window_ok(acc: np.ndarray, activity: str, placement: str) -> bool:
    """Отсев шумной разметки по самому сигналу.

    Метки ставились самими людьми раз в минуту и часто не совпадают с тем,
    что происходило: «ходьба в кармане» с идеально неподвижным телефоном,
    «на столе» с трясущимся. Такие окна учат модель неправде.
    """
    sd = float(np.linalg.norm(acc, axis=1).std())
    if activity in ("WALKING", "RUNNING", "STAIRS_UP", "STAIRS_DOWN", "CYCLING") and sd < 0.5:
        return False
    if activity == "STILL" and sd > 1.5:
        return False
    if placement == "ON_TABLE" and sd > 0.2:
        return False
    return True


def load_extrasensory(root: Path, max_minutes: int = EXTRASENSORY_MAX_MINUTES,
                      max_users: int | None = None, seed: int = 42) -> Dataset | None:
    """Загружает ExtraSensory: 60 человек в обычной жизни, iPhone и Android.

    Ценность — в разметке положения телефона (карман, рука, стол) и в том, что
    люди записаны не в лаборатории. У Android-участников есть свет и
    приближение (по одному значению на минуту); они подставляются в окно,
    и из «разговор + датчик закрыт» получаются примеры положения у уха.

    Перевод в соглашение Android: iPhone пишет ускорение в g с обратным знаком
    (лёжа экраном вверх z = −1), поэтому acc = −9.80665 · a; гироскоп у обеих
    платформ в рад/с с одинаковыми осями. Частоты разные (Android 50 Гц,
    iPhone 34–40 Гц с неровным шагом), поэтому каждая минута приводится
    к 50 Гц линейной интерполяцией по собственным меткам времени файла.
    Магнитометр не берётся: у iPhone он с огромным смещением.
    """
    from concurrent.futures import ThreadPoolExecutor

    root.mkdir(parents=True, exist_ok=True)
    labels_zip = _download(EXTRASENSORY_LABELS_URL, root / "features_labels.zip")
    if labels_zip is None:
        return None
    platforms = _es_platforms(root)
    print("  читаем оглавление архивов сырых данных (один раз, дальше из кэша) …")
    indexes = {k: _zip_index(url, root / f"index_{k}.pkl") for k, (url, _) in EXTRASENSORY_RAW.items()}

    rng = np.random.default_rng(seed)
    collector = _WindowCollector(with_mag=False)

    def fetch(kind: str, uuid: str, ts: int) -> bytes | None:
        url, suffix = EXTRASENSORY_RAW[kind]
        cached = root / kind / uuid / f"{ts}.dat"
        if cached.exists():
            return cached.read_bytes()
        entry = None
        for prefix in (f"{uuid}/{ts}.{suffix}.dat", f"{suffix.removeprefix('m_')}/{uuid}/{ts}.{suffix}.dat"):
            entry = indexes[kind].get(prefix)
            if entry:
                break
        if entry is None:
            return None
        try:
            data = _zip_member(url, entry)
        except Exception:  # noqa: BLE001
            return None
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return data

    with zipfile.ZipFile(labels_zip) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".features_labels.csv.gz"))
        if max_users:
            members = members[:max_users]
        for member in members:
            uuid = Path(member).name.split(".")[0]
            android = platforms.get(uuid) == "android"
            df = pd.read_csv(io.BytesIO(z.read(member)), compression="gzip")
            minutes = _es_select_minutes(df, android, max_minutes, rng)
            if not minutes:
                continue

            with ThreadPoolExecutor(EXTRASENSORY_WORKERS) as pool:
                accs = list(pool.map(lambda m: fetch("acc", uuid, m[0]), minutes))
                gyros = list(pool.map(lambda m: fetch("gyro", uuid, m[0]), minutes))

            before = len(collector.windows)
            for (ts, activity, placement, lux, near), raw_acc, raw_gyro in zip(minutes, accs, gyros):
                if raw_acc is None or raw_gyro is None:
                    continue
                a = _read_es_dat(raw_acc)
                g = _read_es_dat(raw_gyro)
                if a is None or g is None:
                    continue
                start, end = max(a[0, 0], g[0, 0]), min(a[-1, 0], g[-1, 0])
                if end - start < WINDOW_SIZE / SAMPLE_RATE_HZ:
                    continue
                grid = np.arange(start, end, 1.0 / SAMPLE_RATE_HZ)
                acc = np.stack([np.interp(grid, a[:, 0], a[:, k]) for k in (1, 2, 3)], axis=1)
                gyro = np.stack([np.interp(grid, g[:, 0], g[:, k]) for k in (1, 2, 3)], axis=1)
                if not android:
                    acc = -G * acc
                acc, gyro = acc.astype(np.float32), gyro.astype(np.float32)
                if not _es_window_ok(acc, activity, placement):
                    continue
                n_before = len(collector.windows)
                collector.add(acc, gyro, None, activity, placement, f"es{uuid[:8]}",
                              f"es:{placement.lower()}")
                # Свет и приближение известны по минуте — подставляем во все окна минуты.
                for w in collector.windows[n_before:]:
                    w["light_lux"][:] = lux
                    if near is not None:
                        w["proximity_cm"][:] = 0.0 if near else 8.0
                        w["proximity_near"][:] = near
            print(f"  ExtraSensory {uuid[:8]} ({'Android' if android else 'iPhone'}): "
                  f"{len(minutes)} мин -> {len(collector.windows) - before} окон")

    return collector.build()


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

def cap_per_subject_activity(ds: Dataset, cap: int, seed: int = 42) -> Dataset:
    """Оставляет не больше [cap] окон на пару (испытуемый, активность).

    RealWorld даёт сотни тысяч почти одинаковых окон покоя: модели они почти
    ничего не добавляют, а память при обучении съедают гигабайтами. Отбор
    случайный внутри пары, поэтому разнообразие людей и мест на теле
    сохраняется, а перекос в сторону самого многословного источника уходит.
    """
    if cap <= 0:
        return ds
    rng = np.random.default_rng(seed)
    keep = []
    pairs = np.char.add(np.char.add(ds.subjects.astype(str), "|"), ds.y_activity.astype(str))
    for pair in np.unique(pairs):
        idx = np.flatnonzero(pairs == pair)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    if len(keep) == len(ds):
        return ds
    return Dataset(
        x=ds.x[keep],
        context=ds.context[keep],
        y_activity=ds.y_activity[keep],
        y_placement=None if ds.y_placement is None else ds.y_placement[keep],
        subjects=ds.subjects[keep],
        channels=ds.channels,
        groups=None if ds.groups is None else ds.groups[keep],
    )


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
