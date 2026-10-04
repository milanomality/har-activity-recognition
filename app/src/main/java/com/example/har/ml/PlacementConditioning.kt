package com.example.har.ml

/**
 * Активность с учётом положения телефона — второй этап после [PlacementFusion].
 *
 * Положение определяется первым и по признакам, которые от активности
 * почти не зависят: свет, приближение, ориентация, дрожь руки. Затем оно
 * задаёт, какие активности вообще совместимы с тем, где телефон:
 *  - **на столе** телефон никто не несёт — двигаться с ним нельзя, остаётся покой;
 *  - **у уха** человек разговаривает: стоит или идёт, но не бежит и не крутит педали;
 *  - **в руке** бег возможен, но редок, а велосипед — нет: руки на руле;
 *  - **в кармане** ограничений нет — карман двигается вместе с телом.
 *
 * Положение известно лишь с некоторой вероятностью, поэтому множитель класса —
 * это совместимость, усреднённая по распределению положений:
 * `l(a) = Σₚ P(p) · C[p][a]`. Затем вероятности активности умножаются на `l(a)`
 * и нормируются. Уверенное «в кармане» ничего не меняет, уверенное «на столе»
 * почти не оставляет движению шансов, а спорное положение действует пропорционально.
 * Множители не опускаются до нуля: ошибка классификатора положения ослабляет
 * класс, но не делает его невозможным.
 */
object PlacementConditioning {

    /**
     * Совместимость активности с положением, 0–1: строки в порядке [PhonePlacement],
     * столбцы в порядке [ActivityType].
     */
    private val COMPATIBILITY: Array<FloatArray> = Array(PhonePlacement.COUNT) { FloatArray(ActivityType.COUNT) }.also { m ->
        fun row(p: PhonePlacement, vararg pairs: Pair<ActivityType, Float>) {
            pairs.forEach { (a, v) -> m[p.id][a.id] = v }
        }
        row(
            PhonePlacement.POCKET,
            ActivityType.STILL to 1f, ActivityType.WALKING to 1f, ActivityType.RUNNING to 1f,
            ActivityType.STAIRS_UP to 1f, ActivityType.STAIRS_DOWN to 1f, ActivityType.CYCLING to 1f,
        )
        row(
            PhonePlacement.IN_HAND,
            ActivityType.STILL to 1f, ActivityType.WALKING to 1f, ActivityType.RUNNING to 0.6f,
            ActivityType.STAIRS_UP to 1f, ActivityType.STAIRS_DOWN to 1f, ActivityType.CYCLING to 0.1f,
        )
        row(
            PhonePlacement.AT_EAR,
            ActivityType.STILL to 1f, ActivityType.WALKING to 1f, ActivityType.RUNNING to 0.15f,
            ActivityType.STAIRS_UP to 0.8f, ActivityType.STAIRS_DOWN to 0.8f, ActivityType.CYCLING to 0.05f,
        )
        row(
            PhonePlacement.ON_TABLE,
            ActivityType.STILL to 1f, ActivityType.WALKING to 0.03f, ActivityType.RUNNING to 0.02f,
            ActivityType.STAIRS_UP to 0.03f, ActivityType.STAIRS_DOWN to 0.03f, ActivityType.CYCLING to 0.02f,
        )
    }

    /** Совместимость класса [activity] с положением [placement], 0–1. */
    fun compatibility(placement: PhonePlacement, activity: ActivityType): Float =
        COMPATIBILITY[placement.id][activity.id]

    /** Множители для каждого класса активности при данном распределении положений. */
    fun likelihoods(placementProbabilities: FloatArray): FloatArray {
        val sum = placementProbabilities.sum()
        if (placementProbabilities.size < PhonePlacement.COUNT || sum <= 1e-6f) {
            return FloatArray(ActivityType.COUNT) { 1f }
        }
        return FloatArray(ActivityType.COUNT) { a ->
            var l = 0f
            for (p in 0 until PhonePlacement.COUNT) l += placementProbabilities[p] / sum * COMPATIBILITY[p][a]
            l
        }
    }

    fun apply(probabilities: FloatArray, placementProbabilities: FloatArray): FloatArray {
        val l = likelihoods(placementProbabilities)
        val out = FloatArray(probabilities.size) { i -> probabilities[i] * l.getOrElse(i) { 1f } }
        val sum = out.sum()
        if (sum <= 1e-9f) return probabilities
        for (i in out.indices) out[i] /= sum
        return out
    }
}
