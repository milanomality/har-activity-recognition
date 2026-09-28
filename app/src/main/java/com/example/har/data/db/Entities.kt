package com.example.har.data.db

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Журнал по окнам — одна запись на каждое решение классификатора (раз в 1.28 с).
 *
 * Это «сырой» уровень журнала: по нему можно поднять историю целиком,
 * пересчитать метрики и разобрать, почему модель ошиблась на конкретной минуте.
 * Вместе с классом сохраняются признаки окна: без них запись в журнале
 * ничего не объясняет, а с ними решение воспроизводимо.
 */
@Entity(
    tableName = "activity_windows",
    indices = [Index("startMs"), Index("activity")],
)
data class ActivityWindowEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,

    val startMs: Long,
    val endMs: Long,

    /** Имя константы [com.example.har.ml.ActivityType], а не порядковый номер:
     *  добавление класса в середину перечисления не должно портить старый журнал. */
    val activity: String,
    val confidence: Float,
    /** Вероятности всех классов, JSON-массив — для разбора спорных окон. */
    val probabilities: String,

    val placement: String,
    val placementConfidence: Float,

    /** "NEURAL_NET" или "HEURISTIC". */
    val source: String,

    // Признаки окна — объяснение решения.
    @ColumnInfo(name = "acc_mag_mean") val accMagMean: Float,
    @ColumnInfo(name = "acc_mag_std") val accMagStd: Float,
    @ColumnInfo(name = "lin_acc_rms") val linAccRms: Float,
    @ColumnInfo(name = "gyro_mag_mean") val gyroMagMean: Float,
    @ColumnInfo(name = "gyro_mag_std") val gyroMagStd: Float,
    @ColumnInfo(name = "mag_mag_mean") val magMagMean: Float,
    @ColumnInfo(name = "light_lux") val lightLux: Float,
    @ColumnInfo(name = "proximity_near_ratio") val proximityNearRatio: Float,
    @ColumnInfo(name = "dominant_freq_hz") val dominantFreqHz: Float,
    @ColumnInfo(name = "spectral_entropy") val spectralEntropy: Float,
    @ColumnInfo(name = "tilt_deg") val tiltDeg: Float,
    /** Горизонтальная скорость по инерциальной навигации, м/с (с версии БД 2). */
    @ColumnInfo(name = "speed_ms", defaultValue = "0") val speedMs: Float = 0f,
    /** Была ли скорость надёжной — недавно был ZUPT (с версии БД 2). */
    @ColumnInfo(name = "speed_reliable", defaultValue = "0") val speedReliable: Boolean = false,
)

/**
 * Журнал по интервалам — соседние окна одного класса, слитые в один отрезок.
 *
 * Именно это показывается пользователю: «Ходьба, 10:31–10:47, 16 мин».
 * Таблица ведётся инкрементально (открытый интервал дописывается по месту),
 * поэтому журнал переживает перезапуск процесса без пересборки из окон.
 */
@Entity(
    tableName = "activity_intervals",
    indices = [Index("startMs")],
)
data class ActivityIntervalEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,

    val activity: String,
    val startMs: Long,
    val endMs: Long,

    val windowCount: Int,
    /** Сумма уверенностей: среднее считается как sum / windowCount. */
    val confidenceSum: Float,

    /** Счётчики положений телефона внутри интервала, JSON-объект. */
    val placementCounts: String,
) {
    val durationMs: Long get() = endMs - startMs
    val averageConfidence: Float get() = if (windowCount > 0) confidenceSum / windowCount else 0f
}

/** Строка сводки «сколько времени потрачено на каждый вид активности». */
data class ActivitySummaryRow(
    val activity: String,
    val totalMs: Long,
    val intervalCount: Int,
    val avgConfidence: Float,
)
