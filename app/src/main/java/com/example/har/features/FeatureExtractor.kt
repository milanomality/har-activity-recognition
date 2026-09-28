package com.example.har.features

import com.example.har.motion.InertialSpeedEstimator
import com.example.har.sensors.SensorHub
import com.example.har.sensors.SensorWindow
import kotlin.math.acos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Интерпретируемые характеристики окна.
 *
 * Используются в трёх местах: показываются в UI, пишутся в журнал (чтобы
 * решение классификатора можно было объяснить постфактум) и служат входом
 * эвристического классификатора, работающего до загрузки нейросетевой модели.
 */
data class WindowStats(
    val accMagMean: Float,
    val accMagStd: Float,
    val linAccRms: Float,
    val gyroMagMean: Float,
    val gyroMagStd: Float,
    val magMagMean: Float,
    val magMagStd: Float,
    val lightLuxMean: Float,
    val lightAvailable: Boolean,
    val proximityNearRatio: Float,
    val proximityAvailable: Boolean,
    /** Частота основного пика линейного ускорения в полосе 0.5–5 Гц, Гц. */
    val dominantFreqHz: Float,
    /** Мощность этого пика — насколько движение периодично. */
    val dominantPower: Float,
    /** Спектральная энтропия: низкая у ритмичной ходьбы, высокая у тряски транспорта. */
    val spectralEntropy: Float,
    /** Угол между вектором гравитации и осью Z телефона, градусы (0° — экран вверх). */
    val tiltDeg: Float,
    /** Стабильность ориентации: СКО направления гравитации. Низкая — телефон лежит. */
    val orientationStd: Float,
    /** Частота переходов сигнала через среднее — грубая оценка ритмичности. */
    val zeroCrossingRate: Float,
    /** Средняя горизонтальная скорость за окно по инерциальной навигации, м/с. */
    val speedMs: Float = 0f,
    /** Секунды с последнего ZUPT на конце окна; −1 — скорость не считалась. */
    val secondsSinceZupt: Float = -1f,
) {
    /** Можно ли доверять скорости: после ZUPT прошло не слишком много времени. */
    val speedReliable: Boolean
        get() = secondsSinceZupt in 0f..InertialSpeedEstimator.MAX_RELIABLE_SECONDS
}

/**
 * Превращение окна сигналов в то, что потребляют модели.
 *
 * Все формулы здесь обязаны совпадать с `ml/features.py`: модель обучается
 * на признаках из Python, а применяется к признакам из Kotlin, и любое
 * расхождение проявится как необъяснимое падение точности на устройстве.
 */
object FeatureExtractor {

    /** Названия признаков модели положения телефона — порядок критичен. */
    val PLACEMENT_FEATURE_NAMES = listOf(
        "prox_available", "prox_near_ratio", "prox_transitions",
        "light_available", "light_log_mean", "light_log_std", "light_dark_ratio",
        "grav_x_norm", "grav_y_norm", "grav_z_norm", "grav_dir_std",
        "acc_mag_mean_norm", "acc_mag_std", "lin_acc_rms",
        "gyro_mag_mean", "gyro_mag_std",
    )

    const val PLACEMENT_FEATURE_COUNT = 16

    private const val G = 9.80665f

    /** Верхняя граница нормировки освещённости: прямой солнечный свет около 40 000 лк. */
    private const val LUX_MAX = 40_000f
    private val LOG_LUX_MAX = ln(1f + LUX_MAX)

    /** Ниже этого порога освещённости считаем, что телефон в кармане или сумке. */
    private const val DARK_LUX_THRESHOLD = 10f

    /** Полоса частот, в которой лежит человеческая локомоция. */
    private const val MIN_GAIT_HZ = 0.5f
    private const val MAX_GAIT_HZ = 5.0f

    /**
     * Вход модели активности: окно из каналов [channelNames], нормированное
     * по статистикам обучающей выборки, в раскладке «канал последним» — [t][c].
     *
     * Состав каналов диктует обученная модель, а не приложение: датасет без
     * магнитометра даёт шестиканальную модель, собственные данные — девятиканальную.
     */
    fun activityInput(
        window: SensorWindow,
        channelNames: List<String>,
        channelMean: FloatArray,
        channelStd: FloatArray,
    ): FloatArray {
        val n = window.size
        val c = channelNames.size
        val out = FloatArray(n * c)
        for (ci in channelNames.indices) {
            // Неизвестное имя канала не должно ронять инференс: подаём нули,
            // что для нормированного входа означает «среднее значение».
            val data = window.channelByName(channelNames[ci]) ?: continue
            val mean = channelMean.getOrElse(ci) { 0f }
            val std = channelStd.getOrElse(ci) { 1f }.let { if (it > 1e-6f) it else 1f }
            for (t in 0 until n) {
                out[t * c + ci] = (data[t] - mean) / std
            }
        }
        return out
    }

    /** Вход модели положения телефона: 16 агрегированных признаков. */
    fun placementFeatures(window: SensorWindow): FloatArray {
        val n = window.size
        val proxAvailable = window.proximity.any { it >= 0f }
        val lightAvailable = window.light.any { it >= 0f }

        var nearCount = 0
        var transitions = 0
        for (i in 0 until n) {
            if (window.proximityNear[i]) nearCount++
            if (i > 0 && window.proximityNear[i] != window.proximityNear[i - 1]) transitions++
        }

        val logLux = FloatArray(n)
        var darkCount = 0
        for (i in 0 until n) {
            val lux = max(window.light[i], 0f)
            logLux[i] = ln(1f + lux) / LOG_LUX_MAX
            if (lux < DARK_LUX_THRESHOLD) darkCount++
        }

        val gxm = mean(window.gravX)
        val gym = mean(window.gravY)
        val gzm = mean(window.gravZ)
        val accMag = magnitude(window.accX, window.accY, window.accZ)
        val linAcc = FloatArray(n) { i ->
            val lx = window.accX[i] - window.gravX[i]
            val ly = window.accY[i] - window.gravY[i]
            val lz = window.accZ[i] - window.gravZ[i]
            sqrt(lx * lx + ly * ly + lz * lz)
        }
        val gyroMag = magnitude(window.gyroX, window.gyroY, window.gyroZ)

        return floatArrayOf(
            if (proxAvailable) 1f else 0f,
            nearCount.toFloat() / n,
            transitions.toFloat() / n,
            if (lightAvailable) 1f else 0f,
            mean(logLux),
            std(logLux),
            darkCount.toFloat() / n,
            gxm / G,
            gym / G,
            gzm / G,
            gravityDirectionStd(window),
            mean(accMag) / G,
            std(accMag),
            rms(linAcc),
            mean(gyroMag),
            std(gyroMag),
        )
    }

    /** Интерпретируемая сводка окна для UI, журнала и эвристик. */
    fun stats(window: SensorWindow): WindowStats {
        val n = window.size
        val accMag = magnitude(window.accX, window.accY, window.accZ)
        val gyroMag = magnitude(window.gyroX, window.gyroY, window.gyroZ)
        val magMag = magnitude(window.magX, window.magY, window.magZ)

        val accMean = mean(accMag)
        // Убираем постоянную составляющую (гравитацию): спектр должен описывать
        // движение, а не то, как телефон повёрнут.
        val linAcc = FloatArray(n) { accMag[it] - accMean }

        val spectrum = Fft.magnitudeSpectrum(linAcc)
        var peakIdx = -1
        var peakVal = 0f
        var totalPower = 0f
        for (i in spectrum.indices) {
            val hz = Fft.binToHz(i, n, SensorHub.SAMPLE_RATE_HZ)
            if (hz < MIN_GAIT_HZ || hz > MAX_GAIT_HZ) continue
            totalPower += spectrum[i] * spectrum[i]
            if (spectrum[i] > peakVal) {
                peakVal = spectrum[i]
                peakIdx = i
            }
        }
        val dominantHz = if (peakIdx >= 0) Fft.binToHz(peakIdx, n, SensorHub.SAMPLE_RATE_HZ) else 0f

        // Спектральная энтропия по нормированному спектру мощности в той же полосе.
        var entropy = 0f
        if (totalPower > 1e-9f) {
            for (i in spectrum.indices) {
                val hz = Fft.binToHz(i, n, SensorHub.SAMPLE_RATE_HZ)
                if (hz < MIN_GAIT_HZ || hz > MAX_GAIT_HZ) continue
                val p = spectrum[i] * spectrum[i] / totalPower
                if (p > 1e-9f) entropy -= p * ln(p)
            }
        }

        var crossings = 0
        for (i in 1 until n) {
            if ((linAcc[i] >= 0f) != (linAcc[i - 1] >= 0f)) crossings++
        }

        val gxm = mean(window.gravX)
        val gym = mean(window.gravY)
        val gzm = mean(window.gravZ)
        val gNorm = sqrt(gxm * gxm + gym * gym + gzm * gzm)
        val tiltRad = if (gNorm > 1e-3f) acos((gzm / gNorm).coerceIn(-1f, 1f)) else 0f

        val luxValues = window.light.filter { it >= 0f }

        return WindowStats(
            accMagMean = accMean,
            accMagStd = std(accMag),
            linAccRms = rms(linAcc),
            gyroMagMean = mean(gyroMag),
            gyroMagStd = std(gyroMag),
            magMagMean = mean(magMag),
            magMagStd = std(magMag),
            lightLuxMean = if (luxValues.isEmpty()) -1f else luxValues.average().toFloat(),
            lightAvailable = luxValues.isNotEmpty(),
            proximityNearRatio = window.proximityNear.count { it }.toFloat() / n,
            proximityAvailable = window.proximity.any { it >= 0f },
            dominantFreqHz = dominantHz,
            dominantPower = peakVal,
            spectralEntropy = entropy,
            tiltDeg = Math.toDegrees(tiltRad.toDouble()).toFloat(),
            orientationStd = gravityDirectionStd(window),
            zeroCrossingRate = crossings.toFloat() / n,
            speedMs = mean(window.speed),
            secondsSinceZupt = window.secondsSinceZupt[n - 1],
        )
    }

    /**
     * Разброс направления гравитации внутри окна.
     * Считаем среднее СКО по трём нормированным компонентам: телефон, лежащий
     * на столе, даёт значение около нуля, телефон в руке — заметно больше.
     */
    private fun gravityDirectionStd(window: SensorWindow): Float {
        val n = window.size
        val nx = FloatArray(n)
        val ny = FloatArray(n)
        val nz = FloatArray(n)
        for (i in 0 until n) {
            val x = window.gravX[i]
            val y = window.gravY[i]
            val z = window.gravZ[i]
            val norm = max(sqrt(x * x + y * y + z * z), 1e-3f)
            nx[i] = x / norm
            ny[i] = y / norm
            nz[i] = z / norm
        }
        return (std(nx) + std(ny) + std(nz)) / 3f
    }

    private fun magnitude(x: FloatArray, y: FloatArray, z: FloatArray): FloatArray =
        FloatArray(x.size) { sqrt(x[it] * x[it] + y[it] * y[it] + z[it] * z[it]) }

    private fun mean(a: FloatArray): Float {
        if (a.isEmpty()) return 0f
        var s = 0.0
        for (v in a) s += v
        return (s / a.size).toFloat()
    }

    private fun std(a: FloatArray): Float {
        if (a.size < 2) return 0f
        val m = mean(a)
        var s = 0.0
        for (v in a) {
            val d = v - m
            s += d.toDouble() * d
        }
        return sqrt(s / a.size).toFloat()
    }

    private fun rms(a: FloatArray): Float {
        if (a.isEmpty()) return 0f
        var s = 0.0
        for (v in a) s += v.toDouble() * v
        return sqrt(s / a.size).toFloat()
    }
}
