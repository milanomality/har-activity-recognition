"""Признаки для модели положения телефона.

Это зеркало `FeatureExtractor.kt`. Модель обучается на признаках,
посчитанных здесь, а применяется к признакам, посчитанным в Kotlin,
поэтому обе реализации обязаны давать одинаковые числа на одинаковом
окне. Любое расхождение проявится не как ошибка, а как необъяснимое
падение точности на устройстве — отлаживать такое дорого.

Порядок признаков зафиксирован в PLACEMENT_FEATURE_NAMES и совпадает
с одноимённым списком в Kotlin.
"""

from __future__ import annotations

import numpy as np

G = 9.80665

# Верхняя граница нормировки освещённости: прямой солнечный свет ~40 000 лк.
LUX_MAX = 40_000.0
LOG_LUX_MAX = np.log(1.0 + LUX_MAX)

# Ниже этого порога считаем, что телефон в кармане или сумке.
DARK_LUX_THRESHOLD = 10.0

# Доля от maximumRange, ниже которой датчик приближения считается перекрытым.
PROXIMITY_NEAR_FRACTION = 0.5

# Полоса частот, в которой лежит человеческая локомоция.
MIN_GAIT_HZ = 0.5
MAX_GAIT_HZ = 5.0

PLACEMENT_FEATURE_NAMES = [
    "prox_available",
    "prox_near_ratio",
    "prox_transitions",
    "light_available",
    "light_log_mean",
    "light_log_std",
    "light_dark_ratio",
    "grav_x_norm",
    "grav_y_norm",
    "grav_z_norm",
    "grav_dir_std",
    "acc_mag_mean_norm",
    "acc_mag_std",
    "lin_acc_rms",
    "gyro_mag_mean",
    "gyro_mag_std",
]

PLACEMENT_FEATURE_COUNT = len(PLACEMENT_FEATURE_NAMES)


def _std(a: np.ndarray) -> float:
    """СКО по смещённой оценке — как в Kotlin (делим на n, а не на n-1)."""
    return float(np.std(a)) if a.size >= 2 else 0.0


def placement_features(window: dict[str, np.ndarray]) -> np.ndarray:
    """Считает 16 признаков положения телефона для одного окна.

    Окно передаётся словарём каналов; каждый канал — одномерный массив
    одинаковой длины. Ожидаемые ключи: acc_x/y/z, gyro_x/y/z,
    grav_x/y/z, proximity_cm, light_lux, proximity_near.
    """
    n = len(window["acc_x"])

    prox = window.get("proximity_cm")
    near = window.get("proximity_near")
    if prox is None:
        prox = np.full(n, -1.0)
    if near is None:
        # Запасной путь для старых записей без готовой колонки. Порог берём
        # от максимума в окне: у бинарных датчиков это и есть maximumRange.
        # Отрицательные показания означают отсутствие датчика — тогда
        # «перекрыт» ложно везде, как и в SensorHub.kt.
        max_range = float(np.max(prox))
        if max_range <= 0:
            near = np.zeros(n, dtype=bool)
        else:
            near = (prox >= 0) & (prox < max_range * PROXIMITY_NEAR_FRACTION)
    near = np.asarray(near, dtype=bool)

    lux = window.get("light_lux")
    if lux is None:
        lux = np.full(n, -1.0)

    prox_available = 1.0 if np.any(prox >= 0) else 0.0
    light_available = 1.0 if np.any(lux >= 0) else 0.0

    near_ratio = float(np.count_nonzero(near)) / n
    transitions = float(np.count_nonzero(near[1:] != near[:-1])) / n

    lux_clipped = np.maximum(lux, 0.0)
    log_lux = np.log1p(lux_clipped) / LOG_LUX_MAX
    dark_ratio = float(np.count_nonzero(lux_clipped < DARK_LUX_THRESHOLD)) / n

    grav = np.stack([window["grav_x"], window["grav_y"], window["grav_z"]])
    acc = np.stack([window["acc_x"], window["acc_y"], window["acc_z"]])
    gyro = np.stack([window["gyro_x"], window["gyro_y"], window["gyro_z"]])

    acc_mag = np.linalg.norm(acc, axis=0)
    lin_acc = np.linalg.norm(acc - grav, axis=0)
    gyro_mag = np.linalg.norm(gyro, axis=0)

    return np.array(
        [
            prox_available,
            near_ratio,
            transitions,
            light_available,
            float(np.mean(log_lux)),
            _std(log_lux),
            dark_ratio,
            float(np.mean(grav[0])) / G,
            float(np.mean(grav[1])) / G,
            float(np.mean(grav[2])) / G,
            gravity_direction_std(grav),
            float(np.mean(acc_mag)) / G,
            _std(acc_mag),
            float(np.sqrt(np.mean(lin_acc**2))),
            float(np.mean(gyro_mag)),
            _std(gyro_mag),
        ],
        dtype=np.float32,
    )


def magnitude_spectrum(signal: np.ndarray) -> np.ndarray:
    """Спектр амплитуд — зеркало `Fft.magnitudeSpectrum` из Kotlin.

    Окно Ханна подавляет растекание спектра из-за разрыва на краях окна.
    Нормировка на n/2 и ровно та же формула окна обязательны: без них
    абсолютные значения признаков разойдутся с теми, что считает телефон.
    """
    n = 1
    while n * 2 <= len(signal):
        n *= 2
    if n < 2:
        return np.zeros(0, dtype=np.float32)

    i = np.arange(n)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (n - 1)))
    spectrum = np.fft.fft(signal[:n].astype(np.float64) * hann)
    half = n // 2
    return (np.abs(spectrum[:half]) / half).astype(np.float32)


def bin_to_hz(index: int | np.ndarray, n: int, sample_rate: int):
    """Частота бина БПФ — зеркало `Fft.binToHz`."""
    return index * sample_rate / n


def refine_peak(spectrum: np.ndarray, idx: int) -> float:
    """Множитель к частоте бина по параболе через три точки — зеркало `refinePeak`.

    Без уточнения частота шага квантуется с шагом 50/128 = 0.39 Гц.
    """
    if idx <= 0 or idx >= len(spectrum) - 1:
        return 1.0
    a, b, c = (float(spectrum[idx - 1]), float(spectrum[idx]), float(spectrum[idx + 1]))
    denom = a - 2 * b + c
    if abs(denom) < 1e-9:
        return 1.0
    delta = min(max(0.5 * (a - c) / denom, -0.5), 0.5)
    return (idx + delta) / idx


def window_stats(window: dict[str, np.ndarray], sample_rate: int = 50) -> dict[str, float]:
    """Интерпретируемая сводка окна — зеркало `FeatureExtractor.stats` из Kotlin.

    Нужна, чтобы сверять две реализации на одних и тех же данных: приложение
    пишет эти же величины в журнал, и расхождение означало бы, что модель
    на телефоне видит не то, на чём обучалась.
    """
    acc = np.stack([window["acc_x"], window["acc_y"], window["acc_z"]])
    gyro = np.stack([window["gyro_x"], window["gyro_y"], window["gyro_z"]])
    mag = np.stack([window["mag_x"], window["mag_y"], window["mag_z"]])
    grav = np.stack([window["grav_x"], window["grav_y"], window["grav_z"]])
    n = acc.shape[1]

    acc_mag = np.linalg.norm(acc, axis=0)
    gyro_mag = np.linalg.norm(gyro, axis=0)
    mag_mag = np.linalg.norm(mag, axis=0)

    acc_mean = float(np.mean(acc_mag))
    # Убираем постоянную составляющую: спектр должен описывать движение,
    # а не то, как телефон повёрнут.
    lin_acc = acc_mag - acc_mean

    spectrum = magnitude_spectrum(lin_acc)
    hz = bin_to_hz(np.arange(len(spectrum)), n, sample_rate)
    band = (hz >= MIN_GAIT_HZ) & (hz <= MAX_GAIT_HZ)

    dominant_hz = 0.0
    peak = 0.0
    entropy = 0.0
    if band.any() and spectrum[band].size:
        peak_idx = int(np.argmax(spectrum[band]))
        peak = float(spectrum[band][peak_idx])
        full_idx = int(np.flatnonzero(band)[peak_idx])
        dominant_hz = float(bin_to_hz(full_idx, n, sample_rate)) * refine_peak(spectrum, full_idx)
        power = spectrum[band] ** 2
        total = float(power.sum())
        if total > 1e-9:
            p = power / total
            p = p[p > 1e-9]
            entropy = float(-(p * np.log(p)).sum())

    crossings = int(np.count_nonzero((lin_acc[1:] >= 0) != (lin_acc[:-1] >= 0)))

    gm = grav.mean(axis=1)
    g_norm = float(np.linalg.norm(gm))
    tilt = np.degrees(np.arccos(np.clip(gm[2] / g_norm, -1.0, 1.0))) if g_norm > 1e-3 else 0.0

    lux = window.get("light_lux")
    lux_valid = lux[lux >= 0] if lux is not None else np.zeros(0)
    prox_near = window.get("proximity_near")
    near_ratio = (
        float(np.count_nonzero(np.asarray(prox_near, dtype=bool))) / n
        if prox_near is not None else 0.0
    )

    return {
        "acc_mag_mean": acc_mean,
        "acc_mag_std": _std(acc_mag),
        "lin_acc_rms": float(np.sqrt(np.mean(lin_acc**2))),
        "gyro_mag_mean": float(np.mean(gyro_mag)),
        "gyro_mag_std": _std(gyro_mag),
        "mag_mag_mean": float(np.mean(mag_mag)),
        "light_lux": float(np.mean(lux_valid)) if lux_valid.size else -1.0,
        "proximity_near_ratio": near_ratio,
        "dominant_freq_hz": dominant_hz,
        "dominant_power": peak,
        "spectral_entropy": entropy,
        "tilt_deg": float(tilt),
        "zero_crossing_rate": crossings / n,
    }


def gravity_direction_std(grav: np.ndarray) -> float:
    """Разброс направления гравитации: среднее СКО нормированных компонент."""
    norm = np.maximum(np.linalg.norm(grav, axis=0), 1e-3)
    unit = grav / norm
    return float(np.mean([_std(unit[i]) for i in range(3)]))


def estimate_gravity_batch(acc: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Оценка гравитации сразу для пачки окон.

    Args:
        acc: массив (N, T, 3) показаний акселерометра.

    Цикл идёт только по времени и векторизован по окнам и осям: поэлементный
    обход 10 000 окон по 128 отсчётов на чистом Python занимал бы минуты.
    """
    out = np.empty_like(acc, dtype=np.float64)
    out[:, 0, :] = acc[:, 0, :]
    for t in range(1, acc.shape[1]):
        out[:, t, :] = alpha * out[:, t - 1, :] + (1 - alpha) * acc[:, t, :]
    return out


def estimate_gravity(acc: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Оценка гравитации фильтром нижних частот — как в `SensorHub.kt`.

    Нужна, когда исходные данные не содержат отдельного канала гравитации
    (например, собственные записи до версии приложения, которая его пишет).

    Args:
        acc: массив (3, N) показаний акселерометра.
        alpha: вес предыдущего состояния фильтра.

    Returns:
        массив (3, N) оценки вектора гравитации.
    """
    out = np.empty_like(acc, dtype=np.float64)
    out[:, 0] = acc[:, 0]
    for i in range(1, acc.shape[1]):
        out[:, i] = alpha * out[:, i - 1] + (1 - alpha) * acc[:, i]
    return out
