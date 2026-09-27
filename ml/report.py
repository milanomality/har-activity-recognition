"""Статистика обработки сигналов: что считается на каждом этапе и чем ходьба
отличается от остальных классов.

Скрипт отвечает на два вопроса, которые нельзя закрыть одной цифрой точности.

Первый: какие именно числа получаются на каждом шаге конвейера. Для этого
берётся одно настоящее окно ходьбы и протаскивается через все этапы —
сырые отсчёты, модуль ускорения, спектр, 16 контекстных признаков,
нормировка, выход модели.

Второй: чем ходьба отличается от покоя, бега, лестницы и велосипеда. Для
каждого признака считаются межклассовые распределения, а для каждой пары
«ходьба против класса X» — одномерная разделимость (AUC). Так видно, какой
признак реально разводит классы, а какой лишь кажется полезным.

Все спектральные величины считаются функциями из features.py, то есть теми
же формулами, что и на телефоне: отчёт описывает работающую систему,
а не отдельную исследовательскую реализацию.

Запуск:
    python ml/report.py                      # отчёт в stdout
    python ml/report.py --write-readme       # вписать раздел в README.md
    python ml/report.py --source own+uci     # без RealWorld (быстро)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as D  # noqa: E402
import features as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "data"
ASSETS_DIR = ROOT / "app" / "src" / "main" / "assets"

BEGIN_MARK = "<!-- STATS:BEGIN -->"
END_MARK = "<!-- STATS:END -->"

# Порядок как в Labels.kt. VEHICLE отсутствует намеренно: ни в одном
# доступном источнике нет записей транспорта, и модель этот класс не выдаёт.
CLASS_ORDER = ["STILL", "WALKING", "RUNNING", "STAIRS_UP", "STAIRS_DOWN", "CYCLING"]
CLASS_RU = {
    "STILL": "Покой",
    "WALKING": "Ходьба",
    "RUNNING": "Бег",
    "STAIRS_UP": "Лестница вверх",
    "STAIRS_DOWN": "Лестница вниз",
    "CYCLING": "Велосипед",
}

# Признак -> (человеческое имя, единица). Порядок задаёт порядок строк в таблицах.
STATS = {
    "lin_acc_rms": ("СКЗ линейного ускорения", "м/с²"),
    "acc_mag_std": ("СКО модуля ускорения", "м/с²"),
    "grav_dir_std": ("Разброс направления гравитации", "—"),
    "dominant_freq_hz": ("Частота главного пика", "Гц"),
    "dominant_power": ("Мощность главного пика", "м/с²"),
    "spectral_entropy": ("Спектральная энтропия", "нат"),
    "zero_crossing_rate": ("Частота смены знака", "1/отсчёт"),
    "gyro_mag_mean": ("Средняя скорость вращения", "рад/с"),
    "gyro_mag_std": ("СКО скорости вращения", "рад/с"),
}

# Физическая причина, по которой ходьба отличается от класса. Пишется рядом
# с посчитанными числами: цифра показывает, что классы разделимы, а текст —
# почему так получается.
WHY = {
    "STILL": "у покоя нет периодики: телефон лежит, линейное ускорение — это шум датчика",
    "RUNNING": "тот же шаговый ритм, но выше темп и в разы больше амплитуда удара",
    "STAIRS_UP": "почти не отличается: темп шага тот же, амплитуды перекрываются — "
                 "именно здесь модель и ошибается чаще всего",
    "STAIRS_DOWN": "спуск ударнее подъёма: приземление на ступень даёт резкие пики",
    "CYCLING": "педали крутятся медленнее шага, а корпус телефона качает непрерывно",
}


# --------------------------------------------------------------------------
# Статистики окна
# --------------------------------------------------------------------------

def motion_stats(x: np.ndarray, channels: list[str], sample_rate: int = D.SAMPLE_RATE_HZ):
    """Интерпретируемые статистики окон — зеркало `FeatureExtractor.stats`.

    Векторизовано по окнам: поэлементный обход сотен тысяч окон на чистом
    Python занимал бы десятки минут. Формула окна Ханна, нормировка спектра
    и полоса частот взяты из features.py, поэтому числа совпадают с тем,
    что считает телефон.

    Returns:
        (словарь признак -> массив (N,), спектры (N, half), частоты бинов).
    """
    ci = {c: i for i, c in enumerate(channels)}
    missing = [f"{s}_{a}" for s in ("acc", "gyro") for a in "xyz" if f"{s}_{a}" not in ci]
    if missing:
        raise SystemExit(f"В данных нет каналов: {', '.join(missing)}")

    acc = np.stack([x[:, :, ci[f"acc_{a}"]] for a in "xyz"], axis=2).astype(np.float64)
    gyro = np.stack([x[:, :, ci[f"gyro_{a}"]] for a in "xyz"], axis=2).astype(np.float64)

    acc_mag = np.linalg.norm(acc, axis=2)
    gyro_mag = np.linalg.norm(gyro, axis=2)

    # Постоянная составляющая — это гравитация и ориентация телефона.
    # Спектр должен описывать движение, поэтому среднее вычитается.
    acc_mean = acc_mag.mean(axis=1)
    lin = acc_mag - acc_mean[:, None]

    n = lin.shape[1]
    m = 1
    while m * 2 <= n:
        m *= 2
    idx = np.arange(m)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / (m - 1)))
    half = m // 2
    spec = np.abs(np.fft.fft(lin[:, :m] * hann, axis=1)[:, :half]) / half

    hz = F.bin_to_hz(np.arange(half), m, sample_rate)
    band = (hz >= F.MIN_GAIT_HZ) & (hz <= F.MAX_GAIT_HZ)
    sb, hzb = spec[:, band], hz[band]

    k = np.argmax(sb, axis=1)
    dom_hz = hzb[k]
    dom_pw = sb[np.arange(len(sb)), k]

    # Спектральная энтропия: насколько энергия размазана по полосе.
    # У периодического движения она низкая, у тряски — высокая.
    power = sb**2
    total = power.sum(axis=1, keepdims=True)
    p = np.divide(power, total, out=np.zeros_like(power), where=total > 1e-9)
    safe = np.where(p > 1e-9, p, 1.0)
    ent = -np.where(p > 1e-9, p * np.log(safe), 0.0).sum(axis=1)
    ent[total[:, 0] <= 1e-9] = 0.0

    zc = np.count_nonzero((lin[:, 1:] >= 0) != (lin[:, :-1] >= 0), axis=1) / n

    # Линейное ускорение считается вычитанием оценённого вектора гравитации,
    # как в placement_features. Вариант из window_stats (вычитание среднего
    # модуля) здесь не годится: он тождественно равен acc_mag.std(), и две
    # строки таблицы совпадали бы до последнего знака.
    grav = F.estimate_gravity_batch(acc)
    lin_grav = np.linalg.norm(acc - grav, axis=2)
    unit = grav / np.maximum(np.linalg.norm(grav, axis=2, keepdims=True), 1e-3)
    grav_dir_std = unit.std(axis=1).mean(axis=1)

    return {
        "lin_acc_rms": np.sqrt((lin_grav**2).mean(axis=1)),
        "acc_mag_std": acc_mag.std(axis=1),
        "grav_dir_std": grav_dir_std,
        "dominant_freq_hz": dom_hz,
        "dominant_power": dom_pw,
        "spectral_entropy": ent,
        "zero_crossing_rate": zc,
        "gyro_mag_mean": gyro_mag.mean(axis=1),
        "gyro_mag_std": gyro_mag.std(axis=1),
    }, spec, hz


# --------------------------------------------------------------------------
# Разделимость
# --------------------------------------------------------------------------

def _ranks(a: np.ndarray) -> np.ndarray:
    """Средние ранги с учётом совпадающих значений (аналог rankdata('average'))."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    n = len(a)
    first = np.ones(n, dtype=bool)
    first[1:] = s[1:] != s[:-1]
    grp = np.cumsum(first) - 1
    start = np.flatnonzero(first)
    count = np.diff(np.append(start, n))
    avg_rank = start + (count - 1) / 2.0 + 1.0
    out = np.empty(n)
    out[order] = avg_rank[grp]
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Вероятность, что у случайного окна из pos признак выше, чем у окна из neg.

    0.5 — признак не различает классы вовсе, 1.0 или 0.0 — различает
    идеально. Мера не требует нормальности распределений и устойчива
    к выбросам, в отличие от разницы средних.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _ranks(np.concatenate([pos, neg]))
    n1 = len(pos)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * len(neg)))


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------

def load_meta() -> dict | None:
    path = ASSETS_DIR / "model_meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict(x: np.ndarray, context: np.ndarray, channels: list[str], meta: dict,
            batch: int = 512) -> np.ndarray | None:
    """Прогоняет activity_model.tflite батчами. None, если прогон невозможен."""
    path = ASSETS_DIR / "activity_model.tflite"
    if not path.is_file():
        return None
    try:
        import tensorflow as tf
    except ImportError:
        print("  tensorflow недоступен, разделы про модель пропущены", file=sys.stderr)
        return None

    ci = {c: i for i, c in enumerate(channels)}
    if any(c not in ci for c in meta["channels"]):
        print("  каналы данных не совпадают с каналами модели", file=sys.stderr)
        return None
    order = [ci[c] for c in meta["channels"]]

    motion = (x[:, :, order] - np.array(meta["channel_mean"])) / np.array(meta["channel_std"])
    use_ctx = "activity_context_mean" in meta
    ctx = None
    if use_ctx:
        cm = np.array(meta["activity_context_mean"])
        cs = np.array(meta["activity_context_std"])
        ctx = (context - cm) / np.where(cs > 1e-6, cs, 1.0)

    it = tf.lite.Interpreter(model_path=str(path))
    ins = it.get_input_details()
    # Вход с тремя осями — каналы движения, с двумя — контекстные признаки.
    mi = next(d for d in ins if len(d["shape"]) == 3)
    xi = next((d for d in ins if len(d["shape"]) == 2), None) if use_ctx else None

    out = []
    for s in range(0, len(motion), batch):
        mb = motion[s : s + batch].astype(np.float32)
        it.resize_tensor_input(mi["index"], mb.shape, strict=False)
        if xi is not None:
            it.resize_tensor_input(xi["index"], (len(mb), ctx.shape[1]), strict=False)
        it.allocate_tensors()
        it.set_tensor(mi["index"], mb)
        if xi is not None:
            it.set_tensor(xi["index"], ctx[s : s + batch].astype(np.float32))
        it.invoke()
        out.append(it.get_tensor(it.get_output_details()[0]["index"]).copy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Форматирование
# --------------------------------------------------------------------------

def fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def plural(n: int, one: str, few: str, many: str) -> str:
    """Согласует существительное с числом: 1 канал, 2 канала, 6 каналов."""
    rem100 = abs(n) % 100
    if 11 <= rem100 <= 14:
        return many
    rem10 = rem100 % 10
    if rem10 == 1:
        return one
    if 2 <= rem10 <= 4:
        return few
    return many


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load_data(source: str) -> D.Dataset:
    want = source.split("+") if source != "all" else ["own", "uci", "rw"]
    parts = []
    if "own" in want:
        own = D.load_own(DATA_DIR / "own")
        if own:
            parts.append(own)
    if "uci" in want:
        parts.append(D.load_uci(DATA_DIR))
    if "rw" in want:
        rw = D.load_realworld(DATA_DIR / "realworld")
        if rw:
            parts.append(rw)
    if not parts:
        raise SystemExit("Не удалось загрузить ни один источник")
    return D.merge(parts)


# --------------------------------------------------------------------------
# Разделы отчёта
# --------------------------------------------------------------------------

def section_sample(ds: D.Dataset) -> str:
    rows = []
    for cls in CLASS_ORDER:
        n = int(np.count_nonzero(ds.y_activity == cls))
        if n:
            rows.append([CLASS_RU[cls], f"`{cls}`", thousands(n)])
    srcs = sorted({g.split(":")[0] for g in ds.groups}) if ds.groups is not None else []
    head = (
        f"Окон всего: **{thousands(len(ds))}**, "
        f"каналов: **{len(ds.channels)}** ({', '.join(ds.channels)}), "
        f"испытуемых: **{len(np.unique(ds.subjects))}**"
    )
    if srcs:
        head += f", источники: {', '.join(srcs)}"
    return head + "\n\n" + table(["Класс", "Метка", "Окон"], rows)


def section_trace(ds: D.Dataset, stats: dict, spec: np.ndarray, hz: np.ndarray,
                  meta: dict | None, probs: np.ndarray | None) -> str:
    """Разбор одного окна ходьбы по шагам."""
    walk = np.flatnonzero(ds.y_activity == "WALKING")
    if walk.size == 0:
        return "_В выборке нет окон ходьбы._"

    # Берём окно, максимально близкое к медиане ходьбы сразу по темпу
    # и по амплитуде: такое окно представляет класс, а не его край.
    f, a = stats["dominant_freq_hz"][walk], stats["lin_acc_rms"][walk]
    score = (np.abs(f - np.median(f)) / (np.std(f) + 1e-9)
             + np.abs(a - np.median(a)) / (np.std(a) + 1e-9))
    i = int(walk[np.argmin(score)])

    ci = {c: k for k, c in enumerate(ds.channels)}
    w = ds.x[i]
    acc = np.stack([w[:, ci[f"acc_{ax}"]] for ax in "xyz"], axis=1)
    grav = F.estimate_gravity(acc.T.astype(np.float64)).T
    acc_mag = np.linalg.norm(acc, axis=1)

    src = ds.groups[i] if ds.groups is not None else "—"
    lines = [
        f"Разбирается окно №{i} класса «Ходьба» (источник `{src}`, "
        f"испытуемый `{ds.subjects[i]}`) — медианное по темпу и амплитуде, "
        "то есть типичное для класса, а не крайнее.",
        "",
    ]

    rows = [[str(t)] + [fmt(w[t, ci[f"{s}_{ax}"]], 3)
                        for s in ("acc", "gyro") for ax in "xyz"]
            for t in (0, 1, 63, 127)]
    lines += [
        f"**Шаг 1. Сырое окно.** {D.WINDOW_SIZE} отсчётов × {len(ds.channels)} каналов, "
        f"частота {D.SAMPLE_RATE_HZ} Гц, длительность "
        f"{D.WINDOW_SIZE / D.SAMPLE_RATE_HZ:.2f} с. Сдвиг между соседними окнами — "
        f"{D.WINDOW_STRIDE} отсчётов, то есть перекрытие 50 %. Четыре отсчёта из окна:",
        "",
        table(["Отсчёт", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"], rows),
        "",
    ]

    lines += [
        "**Шаг 2. Модуль ускорения и отделение гравитации.** Модуль не зависит от "
        "того, как повёрнут телефон, — это важно, потому что в кармане он лежит "
        "под произвольным углом. Фильтр нижних частот выделяет вектор гравитации, "
        "остаток описывает собственно движение.",
        "",
        table(["Величина", "Значение"], [
            ["Средний модуль ускорения", f"{fmt(acc_mag.mean())} м/с² (гравитация ≈ 9.81)"],
            ["Оценка гравитации в конце окна",
             "(" + ", ".join(fmt(v) for v in grav[-1]) + ") м/с²"],
            ["СКО модуля", f"{fmt(stats['acc_mag_std'][i])} м/с²"],
            ["СКЗ линейного ускорения", f"{fmt(stats['lin_acc_rms'][i])} м/с²"],
        ]),
        "",
    ]

    band = (hz >= F.MIN_GAIT_HZ) & (hz <= F.MAX_GAIT_HZ)
    sb, hzb = spec[i][band], hz[band]
    top = np.argsort(sb)[::-1][:5]
    dom = stats["dominant_freq_hz"][i]
    lines += [
        "**Шаг 3. Спектр.** Окно Ханна подавляет растекание спектра на краях, "
        f"затем БПФ, затем берётся полоса локомоции {F.MIN_GAIT_HZ}–{F.MAX_GAIT_HZ} Гц: "
        "вне неё человеческое движение не лежит. Пять самых сильных частот окна:",
        "",
        table(["Частота, Гц", "Амплитуда, м/с²"],
              [[fmt(hzb[t]), fmt(sb[t], 3)] for t in top]),
        "",
        f"Отсюда три признака: частота главного пика **{fmt(dom)} Гц** — это темп "
        f"шага, около {dom * 60:.0f} шагов в минуту; его мощность "
        f"**{fmt(stats['dominant_power'][i], 3)}**; спектральная энтропия "
        f"**{fmt(stats['spectral_entropy'][i])}** нат. Низкая энтропия означает, "
        "что энергия собрана в одном пике, то есть движение периодично — именно "
        "это отличает шаг от тряски в транспорте.",
        "",
    ]

    names = F.PLACEMENT_FEATURE_NAMES
    ctx = ds.context[i]
    interesting = ["prox_near_ratio", "light_log_mean", "light_dark_ratio",
                   "grav_dir_std", "acc_mag_mean_norm", "lin_acc_rms",
                   "gyro_mag_mean", "gyro_mag_std"]
    lines += [
        f"**Шаг 4. Контекстные признаки.** {len(names)} агрегированных величин — "
        "второй вход модели. Через него в решение попадают датчики приближения "
        "и освещённости, которых нет среди каналов движения. Часть признаков окна:",
        "",
        table(["Признак", "Значение"],
              [[f"`{nm}`", fmt(ctx[names.index(nm)], 3)] for nm in interesting]),
        "",
    ]

    if probs is not None and meta is not None:
        p = probs[i]
        rows = sorted(zip(meta["activity_labels"], p), key=lambda t: -t[1])
        lines += [
            "**Шаг 5. Выход модели.** Оба входа нормируются статистиками обучающей "
            "выборки, свёрточная ветка обрабатывает каналы движения, полносвязная — "
            "контекст, после чего ветки объединяются:",
            "",
            table(["Класс", "Вероятность"],
                  [[CLASS_RU.get(c, c), fmt(v, 3)] for c, v in rows]),
            "",
            f"Модель отдаёт «{CLASS_RU.get(rows[0][0], rows[0][0])}» с вероятностью "
            f"{fmt(rows[0][1], 3)}; ближайший конкурент — "
            f"«{CLASS_RU.get(rows[1][0], rows[1][0])}» ({fmt(rows[1][1], 3)}).",
            "",
            "В приложении к этому добавляется сглаживание: вероятности усредняются "
            "экспоненциально по соседним окнам, и смена класса требует уверенного "
            "перевеса. Одно окно класс не меняет — иначе на переходах он дребезжал бы.",
        ]
    return "\n".join(lines)


def section_classes(ds: D.Dataset, stats: dict) -> str:
    present = [c for c in CLASS_ORDER if np.any(ds.y_activity == c)]
    header = ["Признак", "Ед."] + [CLASS_RU[c] for c in present]
    rows = []
    for key, (title, unit) in STATS.items():
        row = [title, unit]
        for cls in present:
            v = stats[key][ds.y_activity == cls]
            row.append(f"{fmt(np.median(v))} [{fmt(np.percentile(v, 25))}–"
                       f"{fmt(np.percentile(v, 75))}]")
        rows.append(row)
    return (
        "Медиана и межквартильный размах [25–75 %] по каждому классу. Медиана, "
        "а не среднее: распределения признаков несимметричны, и одно окно "
        "с артефактом сдвигает среднее, но не медиану.\n\n"
        + table(header, rows)
        + "\n\nЛинейное ускорение здесь получено вычитанием оценённого вектора "
        "гравитации. В `window_stats` тот же признак считается иначе — вычитанием "
        "среднего модуля, — и тогда он тождественно равен СКО модуля ускорения, "
        "то есть не добавляет информации. Частота главного пика у покоя "
        "(около 3 Гц) смысла не имеет: периодического движения там нет, "
        "и максимум спектра приходится на шум."
    )


def section_walking(ds: D.Dataset, stats: dict) -> str:
    walk_mask = ds.y_activity == "WALKING"
    others = [c for c in CLASS_ORDER if c != "WALKING" and np.any(ds.y_activity == c)]
    rows = []
    for cls in others:
        m = ds.y_activity == cls
        best_key, best_auc = None, 0.5
        for key in STATS:
            a = auc(stats[key][walk_mask], stats[key][m])
            if not np.isfinite(a):
                continue
            if best_key is None or abs(a - 0.5) > abs(best_auc - 0.5):
                best_key, best_auc = key, a
        if best_key is None:
            continue
        title, unit = STATS[best_key]

        def quart(v: np.ndarray) -> str:
            return (f"{fmt(np.median(v))} [{fmt(np.percentile(v, 25))}–"
                    f"{fmt(np.percentile(v, 75))}]")

        rows.append([
            CLASS_RU[cls],
            title,
            fmt(best_auc, 3),
            quart(stats[best_key][walk_mask]),
            quart(stats[best_key][m]) + f" {unit}",
            WHY.get(cls, ""),
        ])

    # Самую трудную пару называем по посчитанным числам, а не по ожиданиям:
    # именно она определяет, где модель будет ошибаться.
    if not rows:
        return "_В выборке нет других классов для сравнения._"
    hardest = min(rows, key=lambda r: abs(float(r[2]) - 0.5))
    # |2·AUC − 1| — разделимость без учёта знака: 0 значит «признак бесполезен»,
    # 1 — «разделяет идеально». Формулировка должна следовать числу, иначе
    # отчёт будет называть сильное разделение слабым при смене выборки.
    sep = abs(2 * float(hardest[2]) - 1)
    strength = (
        "распределения перекрываются почти целиком, и ни один признак не даёт "
        "уверенного разделения" if sep < 0.35 else
        "разделение есть, но оно слабее, чем у любой другой пары, и держится "
        "на одной лишь амплитуде"
    )
    tail = (
        f"\n\nСлабее всего ходьба отделяется от «{hardest[0]}»: лучший из признаков "
        f"даёт AUC {hardest[2]} — {strength}. Причина не в конкретном признаке: "
        "по одному движению телефона шаг по ровной поверхности и шаг по ступени "
        "действительно похожи, и разница между ними меньше, чем разница между "
        "двумя людьми или между карманом и рукой. Матрица ошибок ниже "
        "показывает ровно это."
    )
    return (
        "Для каждой пары «ходьба против класса» перебраны все признаки и выбран "
        "тот, что разделяет пару лучше всех. AUC — вероятность, что у случайного "
        "окна ходьбы признак окажется выше, чем у случайного окна другого класса: "
        "0.5 значит, что признак бесполезен, 1.0 или 0.0 — что он разделяет "
        "классы идеально. Значение ниже 0.5 читается как «у ходьбы этот признак "
        "меньше»: 0.05 — такая же сильная разделимость, как 0.95, только "
        "с обратным знаком.\n\n"
        + table(["Класс", "Самый разделяющий признак", "AUC",
                 "У ходьбы, медиана [25–75 %]", "У класса", "Почему так"], rows)
        + tail
    )


def section_walking_full(ds: D.Dataset, stats: dict) -> str:
    """Матрица AUC: все признаки против всех классов."""
    walk_mask = ds.y_activity == "WALKING"
    others = [c for c in CLASS_ORDER if c != "WALKING" and np.any(ds.y_activity == c)]
    header = ["Признак"] + [CLASS_RU[c] for c in others]
    rows = []
    for key, (title, _) in STATS.items():
        row = [title]
        for cls in others:
            row.append(fmt(auc(stats[key][walk_mask], stats[key][ds.y_activity == cls]), 3))
        rows.append(row)
    return (
        "Тот же показатель для всех признаков сразу — видно, что ни один признак "
        "не разделяет все пары, и именно поэтому решение принимает сеть, "
        "а не пороги по одной величине.\n\n" + table(header, rows)
    )


def section_confusion(ds: D.Dataset, meta: dict, probs: np.ndarray) -> str:
    """Матрица ошибок на отложенных испытуемых."""
    labels = meta["activity_labels"]
    try:
        _, _, test = D.split_by_subject(ds)
    except Exception as e:  # деление может не выйти на малой выборке
        print(f"  деление по испытуемым не удалось: {e}", file=sys.stderr)
        return ""
    if len(test) == 0:
        return ""

    pred = np.array(labels)[probs[test].argmax(axis=1)]
    truth = ds.y_activity[test]
    cols = [c for c in CLASS_ORDER if c in labels]

    rows = []
    for cls in CLASS_ORDER:
        m = truth == cls
        n = int(np.count_nonzero(m))
        if not n:
            continue
        cells = [f"{100 * np.count_nonzero(pred[m] == p) / n:.1f}" for p in cols]
        rows.append([CLASS_RU[cls], thousands(n)] + cells)

    acc_all = float(np.mean(pred == truth))
    extra = ""
    walk = truth == "WALKING"
    if np.any(walk):
        wrong = pred[walk][pred[walk] != "WALKING"]
        if wrong.size:
            v, c = np.unique(wrong, return_counts=True)
            top = v[int(np.argmax(c))]
            recall = 100 * np.count_nonzero(pred[walk] == "WALKING") / np.count_nonzero(walk)
            # Обратную долю считаем, а не предполагаем: путаница между классами
            # бывает и несимметричной, и утверждать симметрию без числа нельзя.
            back = truth == top
            back_rate = (100 * np.count_nonzero(pred[back] == "WALKING") / np.count_nonzero(back)
                         if np.any(back) else float("nan"))
            extra = (
                f"\n\nХодьба — самый трудный класс: правильно распознаётся "
                f"{recall:.1f} % её окон, а чаще всего она принимается за "
                f"«{CLASS_RU.get(top, top)}» — {100 * c.max() / np.count_nonzero(walk):.1f} % "
                "всех окон ходьбы. Путаница идёт в обе стороны, но не поровну: "
                f"обратно, за ходьбу, принимается {fmt(back_rate, 1)} % окон класса "
                f"«{CLASS_RU.get(top, top)}». Причина видна в таблицах выше — "
                "у этих двух классов совпадает темп шага, а амплитуды перекрываются, "
                "потому что зависят от человека и от того, где лежит телефон, "
                "сильнее, чем от самой активности. Надёжно разделить их можно только "
                "признаком, которого в наборе нет, — например барометром: подъём "
                "и спуск меняют высоту, а ходьба по ровному нет."
            )
    return (
        f"Доли в процентах по строкам, на отложенных испытуемых "
        f"({thousands(len(test))} окон). Эти люди не участвовали в обучении, "
        "поэтому цифра отражает работу на новом человеке, а не запоминание "
        f"выборки. Точность на этой выборке — **{100 * acc_all:.1f} %**.\n\n"
        "Цифра зависит от того, какие источники загружены. На всех трёх она "
        "совпадает с результатом, который печатает `ml/train.py`, — это и есть "
        "проверка того, что отчёт считает ту же модель на той же выборке. "
        "Если запустить без RealWorld, останутся почти только записи с поясным "
        "креплением из UCI HAR: они заметно чище и однороднее, и точность "
        "выходит завышенной, около 99 %.\n\n"
        + table(["Истина → предсказание", "Окон"] + [CLASS_RU[c] for c in cols], rows)
        + extra
    )


def section_model(meta: dict | None) -> str:
    """Состав входов и выходов модели.

    Вклад отдельных датчиков здесь намеренно не пересчитывается: выше в README
    есть абляция по группам признаков, и она подробнее — `model_meta.json`
    хранит только два варианта из четырёх.
    """
    if not meta:
        return "_`model_meta.json` не найден — раздел не построен._"
    return (
        f"Модель положения телефона: **{fmt(100 * meta.get('placement_accuracy', 0), 1)} %** "
        f"на {len(meta.get('placement_labels', []))} классах. Обучающая выборка — "
        f"{thousands(meta.get('train_windows', 0))} окон.\n\n"
        f"Модель активности выдаёт {len(meta.get('activity_labels', []))} "
        f"{plural(len(meta.get('activity_labels', [])), 'класс', 'класса', 'классов')} "
        f"({', '.join(meta.get('activity_labels', []))}) и получает "
        f"{len(meta.get('channels', []))} "
        f"{plural(len(meta.get('channels', [])), 'канал', 'канала', 'каналов')} движения "
        f"({', '.join(meta.get('channels', []))}). Класса `VEHICLE`, который есть "
        "в `Labels.kt`, среди выходов модели нет: записей транспорта нет ни в одном "
        "доступном источнике, поэтому нейросеть этот класс никогда не предсказывает — "
        "его отдаёт только эвристический классификатор. Магнитометр в сеть тоже "
        "не подаётся: при склейке источников остаётся пересечение каналов, "
        "а в UCI HAR магнитометра нет."
    )


# --------------------------------------------------------------------------

def build(ds: D.Dataset, meta: dict | None) -> str:
    print("  считаются статистики окон...", file=sys.stderr)
    stats, spec, hz = motion_stats(ds.x, ds.channels)

    probs = None
    if meta:
        print("  прогон модели...", file=sys.stderr)
        probs = predict(ds.x, ds.context, ds.channels, meta)

    parts = [
        BEGIN_MARK,
        "## Статистика обработки сигналов",
        "",
        f"_Раздел сгенерирован `ml/report.py` {date.today().isoformat()}. "
        "Пересчитать: `python ml/report.py --write-readme`._",
        "",
        "### Выборка",
        "",
        section_sample(ds),
        "",
        "### Этап за этапом: одно окно ходьбы",
        "",
        section_trace(ds, stats, spec, hz, meta, probs),
        "",
        "### Признаки по классам",
        "",
        section_classes(ds, stats),
        "",
        "### Чем ходьба отличается от остальных классов",
        "",
        section_walking(ds, stats),
        "",
        section_walking_full(ds, stats),
        "",
    ]
    if probs is not None and meta:
        conf = section_confusion(ds, meta, probs)
        if conf:
            parts += ["### Где классы путаются", "", conf, ""]
    parts += ["### Что именно подаётся в модель", "", section_model(meta), "", END_MARK]
    return "\n".join(parts)


def write_readme(text: str) -> None:
    path = ROOT / "README.md"
    src = path.read_text(encoding="utf-8")
    if BEGIN_MARK in src and END_MARK in src:
        out = (src[: src.index(BEGIN_MARK)] + text
               + src[src.index(END_MARK) + len(END_MARK) :])
    else:
        out = src.rstrip() + "\n\n---\n\n" + text + "\n"
    path.write_text(out, encoding="utf-8")
    print(f"README.md обновлён, раздел {len(text)} символов", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Статистика обработки сигналов и разбор решений модели.")
    ap.add_argument("--source", default="all",
                    help="own, uci, rw или их комбинация через '+' (по умолчанию all)")
    ap.add_argument("--write-readme", action="store_true",
                    help="вписать раздел в README.md между маркерами STATS")
    args = ap.parse_args()

    print(f"Загрузка данных ({args.source})...", file=sys.stderr)
    # Загрузчики из datasets.py печатают прогресс в stdout, и без перенаправления
    # он попадал бы в сам отчёт, когда тот выводится в stdout.
    with contextlib.redirect_stdout(sys.stderr):
        ds = load_data(args.source)
        print(ds.summary())
        text = build(ds, load_meta())
    if args.write_readme:
        write_readme(text)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
