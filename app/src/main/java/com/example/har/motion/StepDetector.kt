package com.example.har.motion

import com.example.har.ml.PhonePlacement
import com.example.har.sensors.SensorHub
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Детектор шагов и оценка скорости по шагам (пешеходное счисление пути, PDR).
 *
 * Работает по вертикальному ускорению в земных осях из [InertialSpeedEstimator].
 * Вертикаль выбрана не случайно: при каждом шаге центр масс тела поднимается
 * и опускается, и это видно одинаково в кармане, в руке и у уха. Раскачка руки
 * или болтание телефона в кармане уходят в основном в горизонталь и вращение,
 * поэтому по вертикали шаг ищется устойчивее, чем по модулю ускорения.
 *
 * Алгоритм на каждом кадре:
 *  1. ФНЧ около 3 Гц убирает удары пятки и дребезг, оставляя ритм шага.
 *  2. Шаг — локальный максимум выше верхнего порога, которому предшествовал
 *     провал ниже нижнего (гистерезис): одиночный толчок без «обратного хода»
 *     шагом не считается. Пороги подстраиваются под амплитуду последних шагов,
 *     поэтому одинаково работают и для осторожной ходьбы с телефоном в руке,
 *     и для бега с телефоном в кармане.
 *  3. Между шагами — не меньше [MIN_STEP_INTERVAL_S] (быстрее человек не бегает)
 *     и не больше [MAX_STEP_INTERVAL_S] (медленнее — уже не ходьба).
 *  4. Длина шага по формуле Вайнберга: L = K · (a_max − a_min)^¼. Коэффициент K
 *     зависит от того, где телефон: в кармане бедро добавляет к размаху свою
 *     амплитуду, в руке её гасит локоть. Поэтому K передаётся снаружи —
 *     по последнему определённому положению телефона ([stepLengthK]).
 *  5. Скорость = темп × длина шага.
 *
 * В отличие от интегрирования ускорения, ошибка здесь не накапливается —
 * каждый шаг оценивается заново. Зато метод работает только для ходьбы и бега.
 */
class StepDetector(
    sampleRateHz: Int = SensorHub.SAMPLE_RATE_HZ,
) {
    /** Состояние после обработки кадра. */
    data class State(
        /** Сколько шагов насчитано с начала сессии. */
        val stepCount: Int,
        /** Темп шагов, Гц (шагов в секунду); 0 — ритма нет. */
        val cadenceHz: Float,
        /** Размах вертикального ускорения за последний шаг, м/с². */
        val stepAmplitude: Float,
        /** Длина шага по Вайнбергу, м. */
        val stepLengthM: Float,
        /** Скорость по шагам, м/с. */
        val speedMs: Float,
        /** На этом кадре засчитан шаг. */
        val stepDetected: Boolean,
    )

    private val dt = 1.0 / sampleRateHz
    private val minStepSamples = (MIN_STEP_INTERVAL_S * sampleRateHz).toInt()
    private val maxStepSamples = (MAX_STEP_INTERVAL_S * sampleRateHz).toInt()

    // Коэффициент ФНЧ первого порядка для частоты среза LOWPASS_HZ.
    private val alpha = run {
        val rc = 1.0 / (2 * Math.PI * LOWPASS_HZ)
        dt / (rc + dt)
    }

    private var filtered = 0.0
    private var prev = 0.0
    private var prevPrev = 0.0
    private var armed = false
    private var troughMin = 0.0
    private var envelope = INITIAL_ENVELOPE
    private var samplesSinceStep = Int.MAX_VALUE / 2

    private val intervals = ArrayDeque<Double>()
    private var stepCount = 0
    private var amplitude = 0.0

    fun reset() {
        filtered = 0.0; prev = 0.0; prevPrev = 0.0
        armed = false
        troughMin = 0.0
        envelope = INITIAL_ENVELOPE
        samplesSinceStep = Int.MAX_VALUE / 2
        intervals.clear()
        stepCount = 0
        amplitude = 0.0
    }

    /**
     * @param verticalAcc вертикальное ускорение без гравитации, м/с²
     * @param k коэффициент Вайнберга для текущего положения телефона
     */
    fun update(verticalAcc: Float, k: Float = DEFAULT_K): State {
        filtered += alpha * (verticalAcc - filtered)
        val x = filtered
        samplesSinceStep++

        val high = max(MIN_HIGH_THRESHOLD, HIGH_FRACTION * envelope)
        val low = max(MIN_LOW_THRESHOLD, LOW_FRACTION * envelope)

        troughMin = min(troughMin, x)
        if (x < -low) armed = true

        // Пик на предыдущем отсчёте: он выше соседей и выше порога.
        var detected = false
        if (armed && prev > high && prev >= prevPrev && x < prev && samplesSinceStep >= minStepSamples) {
            val interval = samplesSinceStep * dt
            if (samplesSinceStep <= maxStepSamples) {
                intervals.addLast(interval)
                if (intervals.size > CADENCE_STEPS) intervals.removeFirst()
            } else {
                // Долгая пауза: прежний ритм к этому шагу отношения не имеет.
                intervals.clear()
            }
            val swing = prev - troughMin
            amplitude = if (amplitude == 0.0) swing else amplitude + AMPLITUDE_SMOOTHING * (swing - amplitude)
            envelope += ENVELOPE_RATE * (prev - envelope)
            stepCount++
            armed = false
            troughMin = x
            samplesSinceStep = 0
            detected = true
        }

        if (samplesSinceStep > maxStepSamples) {
            // Шагов давно нет — сбрасываем ритм и порог, чтобы первый шаг после
            // остановки не потерялся под порогом, подогнанным под бег.
            intervals.clear()
            envelope = INITIAL_ENVELOPE
            amplitude = 0.0
        }

        prevPrev = prev
        prev = x

        // Темп объявляем только после двух интервалов подряд: один интервал —
        // это может быть случайный толчок, а не ритм.
        val cadence = if (intervals.size >= 2) (1.0 / intervals.average()).toFloat() else 0f
        val length = stepLength(amplitude.toFloat(), k)
        return State(
            stepCount = stepCount,
            cadenceHz = cadence,
            stepAmplitude = amplitude.toFloat(),
            stepLengthM = if (cadence > 0f) length else 0f,
            speedMs = cadence * length,
            stepDetected = detected,
        )
    }

    companion object {
        /** Частота среза ФНЧ, Гц: выше темпа бега (≈3 Гц), ниже ударов пятки. */
        private const val LOWPASS_HZ = 3.0

        const val MIN_STEP_INTERVAL_S = 0.25
        const val MAX_STEP_INTERVAL_S = 2.0

        private const val MIN_HIGH_THRESHOLD = 0.6
        private const val MIN_LOW_THRESHOLD = 0.3
        private const val HIGH_FRACTION = 0.4
        private const val LOW_FRACTION = 0.2
        private const val INITIAL_ENVELOPE = 1.5
        private const val ENVELOPE_RATE = 0.3
        private const val AMPLITUDE_SMOOTHING = 0.3

        /** По скольким последним интервалам считается темп. */
        private const val CADENCE_STEPS = 4

        const val DEFAULT_K = 0.45f

        /**
         * Коэффициент Вайнберга по положению телефона. Значения — начальное
         * приближение по литературе, их стоит уточнить по своим данным:
         * пройти известное расстояние и подобрать K так, чтобы сошлась длина.
         */
        private fun kFor(placement: PhonePlacement): Float = when (placement) {
            // Бедро добавляет к размаху собственный мах — K меньше.
            PhonePlacement.POCKET -> 0.42f
            // Рука и голова гасят толчки — тот же шаг даёт меньший размах.
            PhonePlacement.IN_HAND -> 0.50f
            PhonePlacement.AT_EAR -> 0.50f
            PhonePlacement.ON_TABLE -> DEFAULT_K
        }

        /** K, взвешенный по вероятностям положения: без скачка при смене решения. */
        fun stepLengthK(placementProbabilities: FloatArray?): Float {
            if (placementProbabilities == null || placementProbabilities.sum() <= 1e-6f) return DEFAULT_K
            var k = 0f
            var total = 0f
            for (p in PhonePlacement.entries) {
                val w = placementProbabilities.getOrElse(p.id) { 0f }
                k += w * kFor(p)
                total += w
            }
            return k / total
        }

        /** Формула Вайнберга. */
        fun stepLength(amplitude: Float, k: Float): Float =
            if (amplitude <= 0f) 0f else k * amplitude.toDouble().pow(0.25).toFloat()
    }
}
