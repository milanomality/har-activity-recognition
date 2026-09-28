package com.example.har.ml

import android.content.Context
import android.util.Log
import com.example.har.features.FeatureExtractor
import com.example.har.features.WindowStats
import com.example.har.sensors.SensorWindow
import java.io.Closeable

/** Что удалось загрузить и в каком режиме работает распознавание. */
data class RecognizerStatus(
    val activityModelLoaded: Boolean,
    /** Каналы, на которых обучена модель активности — состав задаёт она, не приложение. */
    val channels: List<String>,
    val placementModelLoaded: Boolean,
    val activityAccuracy: Float,
    /** Точность без контекстного входа — показывает реальный вклад датчиков. */
    val activityAccuracyWithoutContext: Float,
    val placementAccuracy: Float,
    /** Использует ли модель активности контекстные признаки (освещённость, приближение). */
    val usesContext: Boolean,
    /** Причина отката на эвристику, если он произошёл. */
    val warning: String?,
) {
    val usingNeuralNet: Boolean get() = activityModelLoaded
}

/** Результат обработки одного окна: класс, уверенность и объясняющие признаки. */
data class RecognitionResult(
    val startTimeMs: Long,
    val endTimeMs: Long,
    val prediction: Prediction,
    val stats: WindowStats,
)

/**
 * Полный конвейер распознавания: окно сигналов → признаки → модель → класс.
 *
 * Держит обе модели и оба сглаживателя. Один экземпляр на процесс —
 * создаётся сервисом распознавания и живёт, пока идёт сбор.
 */
class ActivityRecognizer(context: Context) : Closeable {

    private val meta: ModelMeta? = ModelMeta.load(context)
    private var activityModel: TfLiteModel? = null
    private var placementModel: TfLiteModel? = null

    private val activitySmoother = ProbabilitySmoother(ActivityType.COUNT)
    private val placementSmoother = ProbabilitySmoother(PhonePlacement.COUNT)

    val status: RecognizerStatus

    init {
        val metaProblem = meta?.validateAgainstApp()
        var warning: String? = null

        if (meta == null) {
            warning = "Модель не найдена в assets — работает эвристический классификатор. " +
                "Обучите модель скриптом ml/train.py."
        } else if (metaProblem != null) {
            warning = "Модель несовместима с приложением ($metaProblem) — работает эвристика."
            Log.e(TAG, warning)
        } else {
            activityModel = TfLiteModel.fromAsset(context, ACTIVITY_MODEL_ASSET)
            placementModel = TfLiteModel.fromAsset(context, PLACEMENT_MODEL_ASSET)

            if (activityModel == null) {
                warning = "Файл $ACTIVITY_MODEL_ASSET не загрузился — работает эвристика."
            } else if (activityModel?.outputClasses != meta.activityLabels.size) {
                warning = "Выход модели активности (${activityModel?.outputClasses}) не совпадает " +
                    "со списком классов (${meta.activityLabels.size}) — работает эвристика."
                Log.e(TAG, warning!!)
                activityModel?.close()
                activityModel = null
            } else if (activityModel?.usesContext != meta.usesContext) {
                // Метаданные и файл модели должны договориться о числе входов.
                // Расхождение означает, что .tflite и model_meta.json собраны
                // разными запусками train.py — предсказания были бы мусором.
                warning = "Модель активности принимает " +
                    "${activityModel?.inputCount} вход(а), а метаданные описывают " +
                    (if (meta.usesContext) "два" else "один") + " — работает эвристика."
                Log.e(TAG, warning!!)
                activityModel?.close()
                activityModel = null
            }
            if (placementModel != null && placementModel?.outputClasses != meta.placementLabels.size) {
                Log.e(TAG, "Выход модели положения не совпадает со списком классов, отключаем её")
                placementModel?.close()
                placementModel = null
            }

            // Обучение только на публичном датасете — штатный сценарий:
            // разметки положения телефона там нет. Пользователь должен
            // понимать, что эта половина работает на правилах, а не на сети.
            if (warning == null && placementModel == null) {
                warning = "Положение телефона определяется правилами: модель для него " +
                    "не обучена. Запишите свои данные во вкладке «Сбор данных» " +
                    "и переобучите командой ml/train.py --source both."
            }
        }

        status = RecognizerStatus(
            activityModelLoaded = activityModel != null,
            channels = meta?.channels ?: emptyList(),
            placementModelLoaded = placementModel != null,
            activityAccuracy = meta?.activityAccuracy ?: 0f,
            activityAccuracyWithoutContext = meta?.activityAccuracyWithoutContext ?: 0f,
            placementAccuracy = meta?.placementAccuracy ?: 0f,
            usesContext = activityModel?.usesContext == true,
            warning = warning,
        )
    }

    /** Классифицирует окно. Метод синхронный и рассчитан на вызов из фонового потока. */
    fun recognize(window: SensorWindow): RecognitionResult {
        val stats = FeatureExtractor.stats(window)

        val (modelActivity, activitySource) = predictActivity(window, stats)
        // Скорость из инерциальной навигации уточняет решение модели, пока ей можно верить.
        val rawActivity = SpeedFusion.apply(modelActivity, stats)
        val activityIdx = activitySmoother.update(rawActivity)
        val activity = resolveActivity(activityIdx)

        val rawPlacement = predictPlacement(window, stats)
        val placementIdx = placementSmoother.update(rawPlacement)
        val placement = resolvePlacement(placementIdx)

        return RecognitionResult(
            startTimeMs = window.startTimeMs,
            endTimeMs = window.endTimeMs,
            prediction = Prediction(
                activity = activity,
                activityConfidence = activitySmoother.confidenceOf(activityIdx),
                probabilities = activitySmoother.smoothedProbabilities,
                placement = placement,
                placementConfidence = placementSmoother.confidenceOf(placementIdx),
                placementProbabilities = placementSmoother.smoothedProbabilities,
                source = activitySource,
            ),
            stats = stats,
        )
    }

    fun reset() {
        activitySmoother.reset()
        placementSmoother.reset()
    }

    /**
     * Прогон модели активности. При любой ошибке инференса откатываемся
     * на эвристику для этого окна — потеря одного предсказания не должна
     * ронять фоновый сервис, работающий часами.
     */
    private fun predictActivity(
        window: SensorWindow,
        stats: WindowStats,
    ): Pair<FloatArray, InferenceSource> {
        val model = activityModel
        val m = meta
        if (model != null && m != null) {
            try {
                val motion = FeatureExtractor.activityInput(
                    window, m.channels, m.channelMean, m.channelStd,
                )
                val raw = if (m.usesContext) {
                    // Второй вход — те же 16 признаков, что и у классификатора
                    // положения. Через него в решение об активности попадают
                    // освещённость, приближение и ориентация телефона.
                    val context = standardise(
                        FeatureExtractor.placementFeatures(window),
                        m.contextMean,
                        m.contextStd,
                    )
                    model.predict(motion, context)
                } else {
                    model.predict(motion)
                }
                return expandToFullClasses(raw, m.activityLabels.map { it.id }, ActivityType.COUNT) to
                    InferenceSource.NEURAL_NET
            } catch (e: Exception) {
                Log.w(TAG, "Ошибка инференса активности, окно обработано эвристикой", e)
            }
        }
        return HeuristicClassifier.classifyActivity(stats).second to InferenceSource.HEURISTIC
    }

    private fun predictPlacement(window: SensorWindow, stats: WindowStats): FloatArray {
        val model = placementModel
        val m = meta
        if (model != null && m != null) {
            try {
                val features = FeatureExtractor.placementFeatures(window)
                val raw = model.predict(standardise(features, m.placementMean, m.placementStd))
                return expandToFullClasses(raw, m.placementLabels.map { it.id }, PhonePlacement.COUNT)
            } catch (e: Exception) {
                Log.w(TAG, "Ошибка инференса положения, окно обработано эвристикой", e)
            }
        }
        return HeuristicClassifier.classifyPlacement(stats).second
    }

    /**
     * Приводит признаки к нулевому среднему и единичной дисперсии по
     * статистикам обучающей выборки. Нулевое СКО заменяется единицей:
     * постоянный признак не несёт информации, но делить на ноль нельзя.
     */
    private fun standardise(values: FloatArray, mean: FloatArray, std: FloatArray) =
        FloatArray(values.size) { i ->
            val sd = std.getOrElse(i) { 1f }.let { if (it > 1e-6f) it else 1f }
            (values[i] - mean.getOrElse(i) { 0f }) / sd
        }

    /**
     * Раскладывает выход модели по полному перечню классов приложения.
     *
     * Модель может быть обучена на подмножестве классов (например, датасет
     * без велосипеда). Тогда её выход короче перечисления, и сопоставлять
     * индексы напрямую нельзя — нужна карта из `activity_labels`.
     */
    private fun expandToFullClasses(raw: FloatArray, labelIds: List<Int>, total: Int): FloatArray {
        if (raw.size == total && labelIds == (0 until total).toList()) return raw
        val out = FloatArray(total)
        for (i in raw.indices) {
            val target = labelIds.getOrNull(i) ?: continue
            if (target in 0 until total) out[target] = raw[i]
        }
        return out
    }

    private fun resolveActivity(index: Int) = ActivityType.fromId(index)
    private fun resolvePlacement(index: Int) = PhonePlacement.fromId(index)

    override fun close() {
        activityModel?.close()
        placementModel?.close()
    }

    companion object {
        private const val TAG = "ActivityRecognizer"
        const val ACTIVITY_MODEL_ASSET = "activity_model.tflite"
        const val PLACEMENT_MODEL_ASSET = "placement_model.tflite"
    }
}
