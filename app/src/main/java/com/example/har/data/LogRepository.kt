package com.example.har.data

import android.content.Context
import com.example.har.data.db.ActivityIntervalEntity
import com.example.har.data.db.ActivitySummaryRow
import com.example.har.data.db.ActivityWindowEntity
import com.example.har.data.db.AppDatabase
import com.example.har.ml.PhonePlacement
import com.example.har.ml.RecognitionResult
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject
import java.util.Calendar

/**
 * Ведение журнала активности.
 *
 * Пишет два уровня: покадровый (каждое окно) и агрегированный (интервалы).
 * Интервалы дописываются по месту, поэтому запись каждого окна — это
 * «вставка + обновление последней строки», и порядок этих операций важен:
 * доступ сериализован мьютексом, иначе два окна, пришедшие почти одновременно,
 * могут создать два параллельных интервала одного класса.
 */
class LogRepository(context: Context) {

    private val db = AppDatabase.get(context)
    private val windowDao = db.windowDao()
    private val intervalDao = db.intervalDao()
    private val writeLock = Mutex()

    /** Записывает результат распознавания в оба уровня журнала. */
    suspend fun record(result: RecognitionResult) = writeLock.withLock {
        val p = result.prediction
        val s = result.stats

        windowDao.insert(
            ActivityWindowEntity(
                startMs = result.startTimeMs,
                endMs = result.endTimeMs,
                activity = p.activity.name,
                confidence = p.activityConfidence,
                probabilities = p.probabilities.joinToString(",", "[", "]") {
                    String.format(java.util.Locale.US, "%.4f", it)
                },
                placement = p.placement.name,
                placementConfidence = p.placementConfidence,
                source = p.source.name,
                accMagMean = s.accMagMean,
                accMagStd = s.accMagStd,
                linAccRms = s.linAccRms,
                gyroMagMean = s.gyroMagMean,
                gyroMagStd = s.gyroMagStd,
                magMagMean = s.magMagMean,
                lightLux = s.lightLuxMean,
                proximityNearRatio = s.proximityNearRatio,
                dominantFreqHz = s.dominantFreqHz,
                spectralEntropy = s.spectralEntropy,
                tiltDeg = s.tiltDeg,
                speedMs = s.speedMs,
                speedReliable = s.speedReliable,
                verticalAccRms = s.verticalAccRms,
                horizontalAccRms = s.horizontalAccRms,
                jerkRms = s.jerkRms,
                tiltSwingDeg = s.tiltSwingDeg,
                yawRateMean = s.yawRateMean,
                stepsInWindow = s.stepsInWindow,
                cadenceHz = s.cadenceHz,
                stepSpeedMs = s.stepSpeedMs,
                stepRegularity = s.stepRegularity,
                magInclinationDeg = s.magInclinationDeg,
                magDisturbedRatio = s.magDisturbedRatio,
                magGyroMismatchDeg = s.magGyroMismatchDeg,
            )
        )

        mergeIntoInterval(result)
    }

    /**
     * Продлевает последний интервал или открывает новый.
     *
     * Новый интервал открывается, если сменился класс либо в данных был
     * разрыв длиннее [INTERVAL_GAP_MS] — иначе пауза в работе сервиса
     * (телефон в глубоком сне) превратилась бы в фиктивный многочасовой
     * интервал ходьбы.
     */
    private suspend fun mergeIntoInterval(result: RecognitionResult) {
        val p = result.prediction
        // takeIf вместо отдельного флага: так «интервал, который можно продлить»
        // и «его нет» различаются типом, а не парой связанных переменных.
        val extendable = intervalDao.latest()?.takeIf {
            it.activity == p.activity.name &&
                result.startTimeMs - it.endMs <= INTERVAL_GAP_MS
        }

        if (extendable != null) {
            val counts = parsePlacementCounts(extendable.placementCounts).toMutableMap()
            counts[p.placement.name] = (counts[p.placement.name] ?: 0) + 1
            intervalDao.update(
                extendable.copy(
                    endMs = maxOf(extendable.endMs, result.endTimeMs),
                    windowCount = extendable.windowCount + 1,
                    confidenceSum = extendable.confidenceSum + p.activityConfidence,
                    placementCounts = encodePlacementCounts(counts),
                )
            )
        } else {
            intervalDao.insert(
                ActivityIntervalEntity(
                    activity = p.activity.name,
                    startMs = result.startTimeMs,
                    endMs = result.endTimeMs,
                    windowCount = 1,
                    confidenceSum = p.activityConfidence,
                    placementCounts = encodePlacementCounts(mapOf(p.placement.name to 1)),
                )
            )
        }
    }

    fun recentIntervals(limit: Int = 200): Flow<List<ActivityIntervalEntity>> =
        intervalDao.recent(limit)

    fun intervalsForDay(dayStartMs: Long): Flow<List<ActivityIntervalEntity>> =
        intervalDao.betweenFlow(dayStartMs, dayStartMs + DAY_MS)

    fun summaryForDay(dayStartMs: Long): Flow<List<ActivitySummaryRow>> =
        intervalDao.summaryBetween(dayStartMs, dayStartMs + DAY_MS)

    fun recentWindows(limit: Int = 100): Flow<List<ActivityWindowEntity>> =
        windowDao.recent(limit)

    fun windowCount(): Flow<Int> = windowDao.countFlow()

    suspend fun allWindows(): List<ActivityWindowEntity> = windowDao.all()

    suspend fun allIntervals(): List<ActivityIntervalEntity> = intervalDao.all()

    suspend fun clearAll() = writeLock.withLock {
        windowDao.clear()
        intervalDao.clear()
    }

    /** Удаляет записи старше [days] суток. Держит размер базы под контролем. */
    suspend fun prune(days: Int): Int = writeLock.withLock {
        val cutoff = System.currentTimeMillis() - days * DAY_MS
        windowDao.deleteOlderThan(cutoff) + intervalDao.deleteOlderThan(cutoff)
    }

    companion object {
        const val DAY_MS = 24L * 60 * 60 * 1000

        /**
         * Допустимый разрыв внутри одного интервала. Шаг окна — 1.28 с,
         * 10 с дают запас на редкие пропуски, но не склеивают разные сессии.
         */
        private const val INTERVAL_GAP_MS = 10_000L

        /** Начало текущих суток в местном часовом поясе. */
        fun startOfToday(): Long = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis

        fun parsePlacementCounts(json: String): Map<String, Int> = try {
            val o = JSONObject(json)
            buildMap {
                for (key in o.keys()) put(key, o.getInt(key))
            }
        } catch (_: Exception) {
            emptyMap()
        }

        fun encodePlacementCounts(counts: Map<String, Int>): String =
            JSONObject(counts as Map<*, *>).toString()

        /** Самое частое положение телефона за интервал — для строки журнала. */
        fun dominantPlacement(json: String): PhonePlacement? =
            parsePlacementCounts(json)
                .maxByOrNull { it.value }
                ?.key
                ?.let { PhonePlacement.fromName(it) }
    }
}
