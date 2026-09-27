package com.example.har.ml

/**
 * Временное сглаживание предсказаний.
 *
 * Окна перекрываются на 50 %, решение принимается каждые 1.28 с, и без
 * сглаживания класс дребезжит на переходах: «ходьба — покой — ходьба»
 * за три секунды. Это портит и UI, и журнал, в котором появляются десятки
 * фиктивных интервалов длиной в одно окно.
 *
 * Применяются два механизма:
 *  1. экспоненциальное сглаживание вектора вероятностей — убирает шум;
 *  2. гистерезис — смена класса требует уверенного перевеса, поэтому
 *     короткий всплеск чужого класса не переключает состояние.
 */
class ProbabilitySmoother(
    private val classCount: Int,
    private val alpha: Float = DEFAULT_ALPHA,
    private val switchThreshold: Float = DEFAULT_SWITCH_THRESHOLD,
) {
    private var smoothed: FloatArray? = null
    private var currentClass: Int = -1

    /** Сглаженное распределение после последнего [update]. */
    val smoothedProbabilities: FloatArray
        get() = smoothed?.copyOf() ?: FloatArray(classCount) { 1f / classCount }

    /**
     * Обновляет состояние новым «сырым» распределением.
     *
     * @return индекс класса после сглаживания и гистерезиса
     */
    fun update(raw: FloatArray): Int {
        require(raw.size == classCount) {
            "Ожидалось $classCount классов, получено ${raw.size}"
        }

        val prev = smoothed
        val next = if (prev == null) {
            raw.copyOf()
        } else {
            FloatArray(classCount) { alpha * prev[it] + (1 - alpha) * raw[it] }
        }
        smoothed = next

        val bestIdx = next.indices.maxByOrNull { next[it] } ?: 0
        if (currentClass < 0) {
            currentClass = bestIdx
            return currentClass
        }
        // Гистерезис: уходим от текущего класса только если новый лидер
        // действительно оторвался, а не выиграл доли процента.
        if (bestIdx != currentClass && next[bestIdx] >= next[currentClass] + switchThreshold) {
            currentClass = bestIdx
        }
        return currentClass
    }

    fun confidenceOf(classIndex: Int): Float =
        smoothed?.getOrNull(classIndex) ?: (1f / classCount)

    fun reset() {
        smoothed = null
        currentClass = -1
    }

    companion object {
        /**
         * Вес предыдущего состояния. 0.6 при шаге 1.28 с даёт постоянную
         * времени около 2.5 с: достаточно, чтобы погасить единичный выброс,
         * но реакция на реальную смену активности остаётся в пределах 3–4 с.
         */
        const val DEFAULT_ALPHA = 0.6f

        /** Минимальный перевес нового класса над текущим для переключения. */
        const val DEFAULT_SWITCH_THRESHOLD = 0.15f
    }
}
