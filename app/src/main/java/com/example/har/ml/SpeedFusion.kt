package com.example.har.ml

import com.example.har.features.WindowStats

/**
 * Уточнение вероятностей активности по скорости из инерциальной навигации.
 *
 * Работает поверх любого источника — нейросети или эвристики: вероятность
 * каждого класса умножается на правдоподобие наблюдаемой скорости для этого
 * класса, затем распределение нормируется заново. Так скорость может подтвердить
 * или ослабить решение модели, но не подменяет его: даже при скорости вне
 * диапазона класс сохраняет долю [FLOOR] своей вероятности.
 *
 * Скорость берётся из [WindowStats.effectiveSpeedMs]: по инерциальной навигации,
 * пока ей можно верить, иначе по шагам. Если нет ни той, ни другой
 * (давно не было ZUPT и нет шагового ритма), вероятности не меняются.
 * Класс, которому модель дала ноль, скорость не воскресит — умножение
 * нуля остаётся нулём.
 */
object SpeedFusion {

    /** Минимальный множитель: скорость ослабляет класс, но не запрещает его. */
    const val FLOOR = 0.3f

    /**
     * Множитель для физически невозможного: скорость больше чем вдвое выше
     * верхней границы класса. Ходить 15 км/ч или подниматься по лестнице
     * 10 км/ч человек не может, и тут скорость — уже не подсказка, а запрет.
     */
    const val IMPOSSIBLE = 0.02f

    /** Типичная горизонтальная скорость класса, м/с: [нижняя, верхняя] граница. */
    private val RANGES: Map<ActivityType, Pair<Float, Float>> = mapOf(
        ActivityType.STILL to (0f to 0.3f),
        ActivityType.WALKING to (0.6f to 2.0f),       // 2–7 км/ч
        ActivityType.RUNNING to (2.0f to 6.5f),       // 7–23 км/ч
        // На лестнице горизонтальная скорость ниже: часть шага уходит в высоту.
        ActivityType.STAIRS_UP to (0.2f to 1.2f),
        ActivityType.STAIRS_DOWN to (0.2f to 1.4f),
        ActivityType.CYCLING to (2.5f to 10f),        // 9–36 км/ч
    )

    fun apply(probabilities: FloatArray, stats: WindowStats): FloatArray {
        val v = stats.effectiveSpeedMs ?: return probabilities
        val out = FloatArray(probabilities.size) { i ->
            probabilities[i] * likelihood(ActivityType.fromId(i), v)
        }
        val sum = out.sum()
        if (sum <= 1e-9f) return probabilities
        for (i in out.indices) out[i] /= sum
        return out
    }

    /** Правдоподобие скорости [v] для класса: 1 внутри диапазона, плавно до [FLOOR] вне его. */
    fun likelihood(type: ActivityType, v: Float): Float {
        val (lo, hi) = RANGES[type] ?: return 1f
        if (v > 2f * hi) return IMPOSSIBLE
        // Плавные края шириной в четверть диапазона, но не уже 0.3 м/с:
        // жёсткий порог на границе дал бы дребезг класса.
        val margin = maxOf((hi - lo) * 0.25f, 0.3f)
        val inside = when {
            v < lo - margin || v > hi + margin -> 0f
            v < lo -> (v - (lo - margin)) / margin
            v > hi -> ((hi + margin) - v) / margin
            else -> 1f
        }
        return FLOOR + (1f - FLOOR) * inside
    }
}
