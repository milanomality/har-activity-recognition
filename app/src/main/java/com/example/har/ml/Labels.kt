package com.example.har.ml

/**
 * Распознаваемые типы активности.
 *
 * Порядок констант задаёт порядок выходов модели и не должен меняться
 * без переобучения: `ml/train.py` использует ровно этот список.
 */
enum class ActivityType(val id: Int, val title: String, val emoji: String) {
    STILL(0, "Покой", "🧍"),
    WALKING(1, "Ходьба", "🚶"),
    RUNNING(2, "Бег", "🏃"),
    STAIRS_UP(3, "Подъём по лестнице", "🔼"),
    STAIRS_DOWN(4, "Спуск по лестнице", "🔽"),
    VEHICLE(5, "В транспорте", "🚌"),
    CYCLING(6, "Велосипед", "🚴"),
    ;

    companion object {
        val COUNT = entries.size
        fun fromId(id: Int): ActivityType = entries.firstOrNull { it.id == id } ?: STILL
        fun fromName(name: String): ActivityType? = entries.firstOrNull { it.name == name }
    }
}

/**
 * Положение телефона относительно пользователя.
 *
 * Именно этот классификатор оправдывает наличие датчиков приближения
 * и освещённости: по одному акселерометру карман от руки не отличить.
 */
enum class PhonePlacement(val id: Int, val title: String, val emoji: String) {
    POCKET(0, "В кармане", "👖"),
    IN_HAND(1, "В руке", "✋"),
    AT_EAR(2, "У уха", "📞"),
    ON_TABLE(3, "На столе", "🛋"),
    ;

    companion object {
        val COUNT = entries.size
        fun fromId(id: Int): PhonePlacement = entries.firstOrNull { it.id == id } ?: IN_HAND
        fun fromName(name: String): PhonePlacement? = entries.firstOrNull { it.name == name }
    }
}

/** Откуда получено решение — важно для отладки и для честного отчёта в UI. */
enum class InferenceSource(val title: String) {
    NEURAL_NET("нейросеть"),
    HEURISTIC("эвристика"),
}

/**
 * Результат классификации одного окна.
 *
 * @param probabilities распределение по всем классам активности — хранится
 *   целиком, чтобы журнал позволял разобрать спорные случаи, а не только
 *   показать победивший класс.
 */
data class Prediction(
    val activity: ActivityType,
    val activityConfidence: Float,
    val probabilities: FloatArray,
    val placement: PhonePlacement,
    val placementConfidence: Float,
    val placementProbabilities: FloatArray,
    val source: InferenceSource,
) {
    // equals/hashCode переопределены: у data-класса с FloatArray сравнение
    // по ссылке ломает сравнение состояний в Compose.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Prediction) return false
        return activity == other.activity &&
            activityConfidence == other.activityConfidence &&
            placement == other.placement &&
            placementConfidence == other.placementConfidence &&
            source == other.source &&
            probabilities.contentEquals(other.probabilities) &&
            placementProbabilities.contentEquals(other.placementProbabilities)
    }

    override fun hashCode(): Int {
        var result = activity.hashCode()
        result = 31 * result + activityConfidence.hashCode()
        result = 31 * result + probabilities.contentHashCode()
        result = 31 * result + placement.hashCode()
        result = 31 * result + placementConfidence.hashCode()
        result = 31 * result + placementProbabilities.contentHashCode()
        result = 31 * result + source.hashCode()
        return result
    }
}
