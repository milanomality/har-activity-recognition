package com.example.har.ml

import com.example.har.features.WindowStats

/**
 * Уточнение положения телефона по физическим ограничениям, которые
 * классификатор положения может нарушить.
 *
 * 1. **Карман — это темно и закрыто.** Нейросеть положения видит и движение,
 *    и при энергичном махе рукой относит телефон в карман — хотя освещённость
 *    сотни люкс, а датчик приближения открыт. Если светло и датчик открыт,
 *    вероятность «в кармане» переходит к «в руке»; если датчик открыт,
 *    к «в руке» переходит и «у уха».
 * 2. **Стол не дрожит.** По свету и приближению неподвижный телефон в руке
 *    от телефона на столе не отличить. Отличает дрожь: стол неподвижен
 *    до тысячных долей рад/с, рука даже в покое даёт 0.03–0.06 рад/с
 *    и градус-другой колебаний наклона.
 */
object PlacementFusion {

    /** Выраженность дрожи руки, 0–1. */
    fun tremor(s: WindowStats): Float = maxOf(
        // Вращение: у стола 0.001 рад/с, у руки от 0.03.
        ((s.gyroMagMean - 0.01f) / 0.02f).coerceIn(0f, 1f),
        // Размах наклона за окно: у стола сотые доли градуса, у руки от градуса.
        ((s.tiltSwingDeg - 0.5f) / 1.5f).coerceIn(0f, 1f),
    )

    /** Насколько датчик приближения открыт, 0–1; null — датчика нет. */
    private fun proximityOpen(s: WindowStats): Float? =
        if (s.proximityAvailable) 1f - s.proximityNearRatio else null

    /** Уверенность, что телефон не в кармане: светло и датчик открыт, 0–1. */
    fun notInPocket(s: WindowStats): Float {
        val open = proximityOpen(s)
        val bright = if (s.lightAvailable) ((s.lightLuxMean - 30f) / 70f).coerceIn(0f, 1f) else null
        return when {
            bright != null && open != null -> bright * open
            bright != null -> bright
            // Без датчика света одно открытое приближение — слабое свидетельство.
            open != null -> 0.5f * open
            else -> 0f
        }
    }

    fun apply(probabilities: FloatArray, stats: WindowStats): FloatArray {
        if (probabilities.size < PhonePlacement.COUNT) return probabilities
        val out = probabilities.copyOf()
        val hand = PhonePlacement.IN_HAND.id

        fun move(from: PhonePlacement, share: Float) {
            val moved = out[from.id] * share.coerceIn(0f, 1f)
            out[from.id] -= moved
            out[hand] += moved
        }

        move(PhonePlacement.POCKET, notInPocket(stats))
        proximityOpen(stats)?.let { move(PhonePlacement.AT_EAR, it) }
        move(PhonePlacement.ON_TABLE, tremor(stats))
        return out
    }
}
