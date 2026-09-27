package com.example.har.ml

import com.example.har.features.WindowStats

/**
 * Резервный классификатор на явных правилах.
 *
 * Нужен по двум причинам. Во-первых, приложение должно быть работоспособным
 * сразу после сборки, до того как обучена и положена в assets .tflite-модель.
 * Во-вторых, он даёт базовую линию (baseline), с которой честно сравнивать
 * качество нейросети в отчёте.
 *
 * Пороги подобраны по физике сигнала, а не по обучающей выборке, поэтому
 * класс намеренно различает меньше состояний, чем нейросеть: лестницу
 * от ходьбы по одним лишь порогам надёжно не отделить, и правила не делают
 * вид, что умеют это, а отдают ходьбу с невысокой уверенностью.
 */
object HeuristicClassifier {

    /** Классы, которые правила действительно различают. */
    val SUPPORTED_ACTIVITIES = setOf(
        ActivityType.STILL,
        ActivityType.WALKING,
        ActivityType.RUNNING,
        ActivityType.VEHICLE,
        ActivityType.CYCLING,
    )

    fun classifyActivity(s: WindowStats): Pair<ActivityType, FloatArray> {
        val scores = FloatArray(ActivityType.COUNT)

        // Покой: почти нет линейного ускорения и вращения.
        scores[ActivityType.STILL.id] =
            gate(0.40f - s.linAccRms, 0.40f) * gate(0.25f - s.gyroMagStd, 0.25f)

        // Ходьба: выраженный периодический пик в районе 1.2–2.6 Гц.
        scores[ActivityType.WALKING.id] =
            band(s.dominantFreqHz, 1.2f, 2.6f) *
                band(s.linAccRms, 0.6f, 4.5f) *
                gate(s.dominantPower - 0.05f, 0.30f)

        // Бег: тот же периодический характер, но выше темп и амплитуда.
        scores[ActivityType.RUNNING.id] =
            band(s.dominantFreqHz, 2.2f, 4.5f) *
                gate(s.linAccRms - 3.5f, 4.0f)

        // Транспорт: вибрация есть, но она непериодична — высокая спектральная
        // энтропия при низкой мощности главного пика и почти нулевом вращении.
        scores[ActivityType.VEHICLE.id] =
            band(s.linAccRms, 0.15f, 2.0f) *
                gate(s.spectralEntropy - 1.6f, 1.2f) *
                gate(0.20f - s.gyroMagStd, 0.20f)

        // Велосипед: ноги крутят педали примерно 1–1.5 Гц, корпус телефона
        // при этом постоянно слегка качается — заметное вращение.
        scores[ActivityType.CYCLING.id] =
            band(s.dominantFreqHz, 0.8f, 1.6f) *
                band(s.linAccRms, 0.3f, 2.5f) *
                band(s.gyroMagStd, 0.15f, 1.2f)

        // Лестницу правила не различают — отдаём небольшую долю ходьбы,
        // чтобы класс не выглядел невозможным, но и не выигрывал.
        scores[ActivityType.STAIRS_UP.id] = scores[ActivityType.WALKING.id] * 0.15f
        scores[ActivityType.STAIRS_DOWN.id] = scores[ActivityType.WALKING.id] * 0.15f

        val probs = normalise(scores)
        val best = probs.indices.maxByOrNull { probs[it] } ?: ActivityType.STILL.id
        return ActivityType.fromId(best) to probs
    }

    fun classifyPlacement(s: WindowStats): Pair<PhonePlacement, FloatArray> {
        val scores = FloatArray(PhonePlacement.COUNT)

        val dark = if (!s.lightAvailable) 0.5f else gate(30f - s.lightLuxMean, 30f)
        val bright = if (!s.lightAvailable) 0.5f else gate(s.lightLuxMean - 30f, 200f)
        val covered = if (!s.proximityAvailable) 0.0f else s.proximityNearRatio
        val uncovered = if (!s.proximityAvailable) 0.5f else 1f - s.proximityNearRatio

        // У уха: датчик приближения перекрыт, темно, телефон стоит почти
        // вертикально (наклон около 90°) и при этом слегка движется.
        scores[PhonePlacement.AT_EAR.id] =
            covered * dark * band(s.tiltDeg, 55f, 125f) * gate(s.linAccRms - 0.05f, 0.5f)

        // В кармане: тоже темно и перекрыто, но ориентация нестабильна —
        // телефон болтается вместе с ногой.
        scores[PhonePlacement.POCKET.id] =
            covered * dark * gate(s.orientationStd - 0.01f, 0.06f)

        // На столе: ориентация стабильна до долей градуса, движения нет,
        // датчик приближения открыт.
        scores[PhonePlacement.ON_TABLE.id] =
            uncovered * gate(0.02f - s.orientationStd, 0.02f) * gate(0.25f - s.linAccRms, 0.25f)

        // В руке: датчик открыт, светло, ориентация подвижна, но не хаотична.
        // Требование ненулевого движения обязательно: без него неподвижный
        // телефон на столе набирает почти столько же очков, сколько в руке,
        // потому что мягкий нижний край полосы пропускает и нулевой разброс.
        scores[PhonePlacement.IN_HAND.id] =
            uncovered * bright *
                band(s.orientationStd, 0.005f, 0.12f) *
                gate(s.linAccRms - 0.05f, 0.30f)

        val probs = normalise(scores)
        val best = probs.indices.maxByOrNull { probs[it] } ?: PhonePlacement.IN_HAND.id
        return PhonePlacement.fromId(best) to probs
    }

    /**
     * Мягкая ступенька: 0 при value <= 0, 1 при value >= width, линейно между.
     * Плавность важна — жёсткие пороги дают дребезг класса на границе.
     */
    private fun gate(value: Float, width: Float): Float =
        (value / width).coerceIn(0f, 1f)

    /** Трапециевидная принадлежность полосе [lo, hi] с плавными краями. */
    private fun band(value: Float, lo: Float, hi: Float): Float {
        val margin = (hi - lo) * 0.35f
        return when {
            value < lo - margin || value > hi + margin -> 0f
            value < lo -> (value - (lo - margin)) / margin
            value > hi -> ((hi + margin) - value) / margin
            else -> 1f
        }
    }

    /**
     * Приводит оценки к распределению. Если все правила дали ноль
     * (сигнал не похож ни на что известное), отдаём равномерное
     * распределение — это честнее, чем выдавать случайный класс за уверенный.
     */
    private fun normalise(scores: FloatArray): FloatArray {
        val sum = scores.sum()
        if (sum <= 1e-6f) return FloatArray(scores.size) { 1f / scores.size }
        return FloatArray(scores.size) { scores[it] / sum }
    }
}
