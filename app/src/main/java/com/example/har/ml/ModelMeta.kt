package com.example.har.ml

import android.content.Context
import android.util.Log
import com.example.har.features.FeatureExtractor
import com.example.har.sensors.SensorWindow
import org.json.JSONArray
import org.json.JSONObject

/**
 * Описание обученных моделей: что подавать на вход и как это нормировать.
 *
 * Файл `assets/model_meta.json` генерирует `ml/train.py` вместе с .tflite —
 * так статистики нормировки не могут разойтись с весами модели.
 */
data class ModelMeta(
    val version: Int,
    val sampleRateHz: Int,
    val windowSize: Int,
    val channels: List<String>,
    val activityLabels: List<ActivityType>,
    val channelMean: FloatArray,
    val channelStd: FloatArray,
    /** Статистики нормировки контекстного входа. Пусты у одновходовой модели. */
    val contextMean: FloatArray,
    val contextStd: FloatArray,
    val placementLabels: List<PhonePlacement>,
    val placementMean: FloatArray,
    val placementStd: FloatArray,
    val activityAccuracy: Float,
    /** Точность той же модели с обнулённым контекстом — мера вклада датчиков. */
    val activityAccuracyWithoutContext: Float,
    val placementAccuracy: Float,
) {
    /**
     * Принимает ли модель активности второй вход — контекстные признаки,
     * через которые в решение попадают освещённость и приближение.
     */
    val usesContext: Boolean get() = contextMean.isNotEmpty()
    /**
     * Проверка, что модель обучалась на той же геометрии данных, что подаёт
     * приложение. Молча работать с несовпадающим окном — верный способ
     * получить «модель загрузилась, но предсказывает мусор».
     */
    fun validateAgainstApp(): String? = when {
        windowSize != SensorWindow.WINDOW_SIZE ->
            "длина окна модели $windowSize, приложение подаёт ${SensorWindow.WINDOW_SIZE}"
        channelMean.size != channels.size ->
            "в метаданных ${channels.size} каналов, но ${channelMean.size} статистик нормировки"
        channels.any { it !in SensorWindow.ALL_MOTION_CHANNELS } ->
            "неизвестные каналы: " +
                channels.filter { it !in SensorWindow.ALL_MOTION_CHANNELS }.joinToString(", ")
        placementMean.size != FeatureExtractor.PLACEMENT_FEATURE_COUNT ->
            "модель положения ждёт ${placementMean.size} признаков, " +
                "приложение считает ${FeatureExtractor.PLACEMENT_FEATURE_COUNT}"
        usesContext && contextMean.size != FeatureExtractor.PLACEMENT_FEATURE_COUNT ->
            "контекстный вход ждёт ${contextMean.size} признаков, " +
                "приложение считает ${FeatureExtractor.PLACEMENT_FEATURE_COUNT}"
        usesContext && contextStd.size != contextMean.size ->
            "число статистик контекста не совпадает: " +
                "${contextMean.size} средних против ${contextStd.size} СКО"
        else -> null
    }

    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)

    companion object {
        private const val TAG = "ModelMeta"
        const val ASSET_NAME = "model_meta.json"

        /** Читает метаданные из assets. Возвращает null, если файла нет или он битый. */
        fun load(context: Context): ModelMeta? = try {
            val json = context.assets.open(ASSET_NAME).bufferedReader().use { it.readText() }
            parse(JSONObject(json))
        } catch (e: Exception) {
            Log.w(TAG, "Метаданные моделей недоступны, работаем на эвристике", e)
            null
        }

        private fun parse(o: JSONObject): ModelMeta {
            val activityLabels = o.getJSONArray("activity_labels").toStringList()
                .mapNotNull { ActivityType.fromName(it) }
            val placementLabels = o.getJSONArray("placement_labels").toStringList()
                .mapNotNull { PhonePlacement.fromName(it) }

            require(activityLabels.isNotEmpty()) { "activity_labels пуст или содержит неизвестные классы" }
            require(placementLabels.isNotEmpty()) { "placement_labels пуст или содержит неизвестные классы" }

            return ModelMeta(
                version = o.optInt("version", 1),
                sampleRateHz = o.optInt("sample_rate_hz", 50),
                windowSize = o.optInt("window_size", SensorWindow.WINDOW_SIZE),
                channels = o.getJSONArray("channels").toStringList(),
                activityLabels = activityLabels,
                channelMean = o.getJSONArray("channel_mean").toFloatArray(),
                channelStd = o.getJSONArray("channel_std").toFloatArray(),
                // Секции контекста может не быть: модели, обученные более
                // ранней версией скрипта, одновходовые и остаются рабочими.
                contextMean = o.optJSONArray("activity_context_mean").toFloatArrayOrEmpty(),
                contextStd = o.optJSONArray("activity_context_std").toFloatArrayOrEmpty(),
                placementLabels = placementLabels,
                placementMean = o.getJSONArray("placement_mean").toFloatArray(),
                placementStd = o.getJSONArray("placement_std").toFloatArray(),
                activityAccuracy = o.optDouble("activity_accuracy", 0.0).toFloat(),
                activityAccuracyWithoutContext =
                    o.optDouble("activity_accuracy_without_context", 0.0).toFloat(),
                placementAccuracy = o.optDouble("placement_accuracy", 0.0).toFloat(),
            )
        }

        private fun JSONArray.toFloatArray() = FloatArray(length()) { getDouble(it).toFloat() }
        private fun JSONArray.toStringList() = List(length()) { getString(it) }
        private fun JSONArray?.toFloatArrayOrEmpty() =
            this?.let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } } ?: FloatArray(0)
    }
}
