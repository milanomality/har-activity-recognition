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
    /** СКЗ вертикального ускорения, м/с² (с версии БД 3). */
    @ColumnInfo(name = "vertical_acc_rms", defaultValue = "0") val verticalAccRms: Float = 0f,
    /** СКЗ горизонтального ускорения, м/с² (с версии БД 3). */
    @ColumnInfo(name = "horizontal_acc_rms", defaultValue = "0") val horizontalAccRms: Float = 0f,
    /** СКЗ рывка, м/с³ (с версии БД 3). */
    @ColumnInfo(name = "jerk_rms", defaultValue = "0") val jerkRms: Float = 0f,
    /** Размах наклона телефона за окно, градусы (с версии БД 3). */
    @ColumnInfo(name = "tilt_swing_deg", defaultValue = "0") val tiltSwingDeg: Float = 0f,
    /** Средняя угловая скорость вокруг вертикали, рад/с (с версии БД 3). */
    @ColumnInfo(name = "yaw_rate_mean", defaultValue = "0") val yawRateMean: Float = 0f,
    /** Шагов в окне (с версии БД 3). */
    @ColumnInfo(name = "steps", defaultValue = "0") val stepsInWindow: Int = 0,
    /** Темп шагов, Гц (с версии БД 3). */
    @ColumnInfo(name = "cadence_hz", defaultValue = "0") val cadenceHz: Float = 0f,
    /** Скорость по шагам, м/с (с версии БД 3). */
    @ColumnInfo(name = "step_speed_ms", defaultValue = "0") val stepSpeedMs: Float = 0f,
    /** Регулярность шага, 0–1 (с версии БД 3). */
    @ColumnInfo(name = "step_regularity", defaultValue = "0") val stepRegularity: Float = 0f,
    /** Магнитное наклонение, градусы (с версии БД 3). */
    @ColumnInfo(name = "mag_inclination_deg", defaultValue = "0") val magInclinationDeg: Float = 0f,
    /** Доля кадров с искажённым полем; −1 — нет магнитометра (с версии БД 3). */
    @ColumnInfo(name = "mag_disturbed_ratio", defaultValue = "0") val magDisturbedRatio: Float = 0f,
    /** Расхождение поворота по компасу и гироскопу, градусы (с версии БД 3). */
    @ColumnInfo(name = "mag_gyro_mismatch_deg", defaultValue = "0") val magGyroMismatchDeg: Float = 0f,

    // --- Разбор решения о положении (с версии БД 4) ---
    /** Доля силы тяжести по оси X телефона, −1…1 (с версии БД 4). */
    @ColumnInfo(name = "grav_x", defaultValue = "0") val gravityX: Float = 0f,
    /** Доля силы тяжести по оси Y телефона, −1…1; +1 — верх телефона вверху (с версии БД 4). */
    @ColumnInfo(name = "grav_y", defaultValue = "0") val gravityY: Float = 0f,
    /** Доля силы тяжести по оси Z телефона, −1…1; +1 — экраном вверх (с версии БД 4). */
    @ColumnInfo(name = "grav_z", defaultValue = "0") val gravityZ: Float = 0f,
    /** СКО направления гравитации — устойчивость ориентации (с версии БД 4). */
    @ColumnInfo(name = "orientation_std", defaultValue = "0") val orientationStd: Float = 0f,
    /** Похожесть позы на разговор у уха, 0–1 (с версии БД 4). */
    @ColumnInfo(name = "ear_pose", defaultValue = "0") val earPose: Float = 0f,
    /** Сглаженные вероятности положения, JSON-массив в порядке PhonePlacement (с версии БД 4). */
    @ColumnInfo(name = "placement_probabilities", defaultValue = "''") val placementProbabilities: String = "",
    /** Выход модели положения до поправок, JSON-массив (с версии БД 4). */
    @ColumnInfo(name = "model_placement_probabilities", defaultValue = "''") val modelPlacementProbabilities: String = "",
    /** Выход модели активности до поправок, JSON-массив (с версии БД 4). */
    @ColumnInfo(name = "model_activity_probabilities", defaultValue = "''") val modelActivityProbabilities: String = "",

    // --- Сверка с Google Activity Recognition (с версии БД 5) ---
    /** Класс по Google (имя GoogleActivityType); пусто — свежего ответа не было (с версии БД 5). */
    @ColumnInfo(name = "google_activity", defaultValue = "''") val googleActivity: String = "",
    /** Уверенность Google, 0–100; −1 — ответа не было (с версии БД 5). */
    @ColumnInfo(name = "google_confidence", defaultValue = "-1") val googleConfidence: Int = -1,
    /** Совпадение с Google: 1 — да, 0 — нет, −1 — сравнивать не с чем (с версии БД 5). */
    @ColumnInfo(name = "google_agrees", defaultValue = "-1") val googleAgrees: Int = -1,
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
