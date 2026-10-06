package com.example.har.ml

import com.example.har.features.WindowStats
import kotlin.math.abs

/**
 * Уточнение положения телефона по физическим ограничениям, которые
 * классификатор положения может нарушить.
 *
 * 1. **Карман — это темно и закрыто.** Нейросеть положения видит и движение,
 *    и при энергичном махе рукой относит телефон в карман — хотя освещённость
 *    сотни люкс, а датчик приближения открыт. Если светло и датчик открыт,
 *    вероятность «в кармане» переходит к «в руке»; если датчик открыт,
 *    к «в руке» переходит и «у уха».
 * 2. **Ухо от кармана отличает поза, а не свет.** У уха датчик приближения
 *    закрыт, а световой прикрыт головой — ровно как в кармане, а в тёмной
 *    комнате темно везде. Модель, не видевшая уха в темноте, отдаёт карману 0.99.
 *    Различает геометрия: у уха экран прижат к щеке (стоит вертикально),
 *    верх телефона выше низа, а сам телефон идёт по диагонали от уха ко рту —
 *    заметная доля силы тяжести ложится на поперечную ось X. В кармане стоящего
 *    человека телефон висит отвесно (X ≈ 0, верхом вверх или вниз), у сидящего
 *    лежит плашмя вдоль бедра, у идущего раскачивается вместе с ногой.
 *    Если датчик закрыт и поза «как у уха», вероятность кармана переходит к уху;
 *    перевёрнутый верхом вниз телефон, наоборот, у уха быть не может.
 * 3. **Стол не дрожит.** По свету и приближению неподвижный телефон в руке
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

    /**
     * Насколько поза телефона похожа на разговор у уха, 0–1 — без учёта света
     * и приближения.
     */
    fun earPose(s: WindowStats): Float {
        // Верх телефона выше низа: при отвесном телефоне Y = 1, у уха обычно 0.5–0.9.
        val topUp = ((s.gravityYRatio - 0.25f) / 0.25f).coerceIn(0f, 1f)
        // Экран стоит вертикально: наклон к горизонту 90° ± 35°.
        val screenUpright = ((35f - abs(s.tiltDeg - 90f)) / 20f).coerceIn(0f, 1f)
        // Диагональ от уха ко рту: заметная доля тяжести по оси X.
        // В кармане отвесный телефон даёт X около нуля.
        val diagonal = ((abs(s.gravityXRatio) - 0.2f) / 0.2f).coerceIn(0f, 1f)
        // Не раскачивается, как телефон в кармане идущего: у уха наклон гуляет
        // на градусы, в кармане при шаге — на десятки.
        val steady = 1f - ((s.tiltSwingDeg - 20f) / 20f).coerceIn(0f, 1f)
        return topUp * screenUpright * diagonal * steady
    }

    /** Телефон перевёрнут верхом вниз — так его к уху не подносят, 0–1. */
    private fun upsideDown(s: WindowStats): Float = ((-s.gravityYRatio - 0.3f) / 0.3f).coerceIn(0f, 1f)

    fun apply(probabilities: FloatArray, stats: WindowStats): FloatArray {
        if (probabilities.size < PhonePlacement.COUNT) return probabilities
        val out = probabilities.copyOf()

        fun move(from: PhonePlacement, share: Float, to: PhonePlacement = PhonePlacement.IN_HAND) {
            val moved = out[from.id] * share.coerceIn(0f, 1f)
            out[from.id] -= moved
            out[to.id] += moved
        }

        // Ухо и карман разводятся до правил про свет: оба закрыты и темны.
        val covered = if (stats.proximityAvailable) stats.proximityNearRatio else 0f
        move(PhonePlacement.POCKET, covered * earPose(stats), PhonePlacement.AT_EAR)
        move(PhonePlacement.AT_EAR, upsideDown(stats), PhonePlacement.POCKET)

        move(PhonePlacement.POCKET, notInPocket(stats))
        proximityOpen(stats)?.let { move(PhonePlacement.AT_EAR, it) }
        move(PhonePlacement.ON_TABLE, tremor(stats))
        return out
    }
}
