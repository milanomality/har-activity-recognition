package com.example.har.features

import com.example.har.motion.InertialSpeedEstimator
import com.example.har.sensors.SensorHub
import com.example.har.sensors.SensorWindow
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
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
    /** Доля силы тяжести вдоль оси X телефона (поперёк экрана), −1…1. */
    val gravityXRatio: Float = 0f,
    /** Доля силы тяжести вдоль оси Y телефона (вдоль экрана), −1…1: +1 — верх телефона вверху. */
    val gravityYRatio: Float = 0f,
    /** Доля силы тяжести вдоль оси Z телефона (из экрана), −1…1: +1 — экраном вверх. */
    val gravityZRatio: Float = 0f,
    /** Частота переходов сигнала через среднее — грубая оценка ритмичности. */
    val zeroCrossingRate: Float,
    /** Средняя горизонтальная скорость за окно по инерциальной навигации, м/с. */
    val speedMs: Float = 0f,
    /** Секунды с последнего ZUPT на конце окна; −1 — скорость не считалась. */
    val secondsSinceZupt: Float = -1f,
    /** Поворот телефона с последнего ZUPT на конце окна, рад. */
    val rotationSinceZupt: Float = 0f,

    // --- Движение в земных осях (из инерциальной навигации) ---
    /** СКЗ вертикального ускорения, м/с²: подпрыгивание тела при шаге. */
    val verticalAccRms: Float = 0f,
    /** СКЗ горизонтального ускорения, м/с²: разгон, торможение, мах руки. */
    val horizontalAccRms: Float = 0f,
    /** Доля вертикали в энергии движения, 0–1: у шага высокая, у транспорта низкая. */
    val verticalShare: Float = 0f,
    /** СКЗ рывка (производной модуля ускорения), м/с³: резкость движения. */
    val jerkRms: Float = 0f,
    /** Размах наклона телефона за окно, градусы: мах руки или бедра. */
    val tiltSwingDeg: Float = 0f,
    /** Средний модуль угловой скорости вокруг вертикали, рад/с: повороты корпуса. */
    val yawRateMean: Float = 0f,

    // --- Шаги ---
    /** Шагов в окне. */
    val stepsInWindow: Int = 0,
    /** Темп шагов на конце окна, Гц; 0 — ритма нет. */
    val cadenceHz: Float = 0f,
    /** Размах вертикального ускорения за шаг, м/с². */
    val stepAmplitude: Float = 0f,
    /** Средняя скорость по шагам за окно, м/с (длина шага зависит от положения телефона). */
    val stepSpeedMs: Float = 0f,
    /** Регулярность шага: максимум автокорреляции вертикального ускорения, 0–1. */
    val stepRegularity: Float = 0f,

    // --- Магнитометр ---
    /** Среднее магнитное наклонение, градусы. */
    val magInclinationDeg: Float = 0f,
    /** Доля кадров окна с искажённым полем (металл, ток), 0–1; −1 — магнитометра нет. */
    val magDisturbedRatio: Float = -1f,
    /**
     * Расхождение поворота по компасу и по гироскопу за окно, градусы.
     * Телефон один, поворот один — если компас «повернулся», а гироскоп нет,
     * значит изменилось само поле: рядом едет металл или течёт ток.
     */
    val magGyroMismatchDeg: Float = 0f,
) {
    /** Можно ли доверять скорости: после ZUPT прошло мало времени и телефон мало вращался. */
    val speedReliable: Boolean
        get() = InertialSpeedEstimator.isReliable(secondsSinceZupt, rotationSinceZupt)

    /** Считались ли в этом окне производные величины движения. */
    val motionComputed: Boolean get() = secondsSinceZupt >= 0f

    /** Есть ли в окне устойчивый шаговый ритм. */
    val hasGait: Boolean get() = stepsInWindow >= 2 && cadenceHz > 0f

    /**
     * Скорость, которой можно пользоваться: по инерциальной навигации, пока
     * ей можно верить, иначе по шагам, если человек идёт. null — ни то, ни другое.
     */
    val effectiveSpeedMs: Float?
        get() = when {
            speedReliable -> speedMs
            hasGait -> stepSpeedMs
            else -> null
        }
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

    /** Диапазон лагов автокорреляции: от быстрого шага до медленного двойного шага, с. */
    private const val MIN_STEP_LAG_S = 0.3
    private const val MAX_STEP_LAG_S = 1.2

    /** Ниже этой дисперсии вертикального ускорения (м²/с⁴) регулярность не считается: там шум. */
    private const val MIN_REGULARITY_ENERGY = 0.05

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
        val dominantHz = if (peakIdx >= 0) {
            Fft.binToHz(peakIdx, n, SensorHub.SAMPLE_RATE_HZ) * refinePeak(spectrum, peakIdx)
        } else {
            0f
        }

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

        val motion = window.secondsSinceZupt[n - 1] >= 0f
        val vRms = rms(window.verticalAcc)
        val hRms = rms(window.horizontalAcc)

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
            gravityXRatio = if (gNorm > 1e-3f) gxm / gNorm else 0f,
            gravityYRatio = if (gNorm > 1e-3f) gym / gNorm else 0f,
            gravityZRatio = if (gNorm > 1e-3f) gzm / gNorm else 0f,
            zeroCrossingRate = crossings.toFloat() / n,
            speedMs = mean(window.speed),
            secondsSinceZupt = window.secondsSinceZupt[n - 1],
            rotationSinceZupt = window.rotationSinceZupt[n - 1],
            verticalAccRms = vRms,
            horizontalAccRms = hRms,
            verticalShare = if (vRms + hRms > 1e-6f) vRms * vRms / (vRms * vRms + hRms * hRms) else 0f,
            jerkRms = jerkRms(accMag),
            tiltSwingDeg = if (motion) window.insTiltDeg.max() - window.insTiltDeg.min() else 0f,
            yawRateMean = meanAbs(window.yawRate),
            stepsInWindow = max(window.stepCount[n - 1] - window.stepCount[0], 0),
            cadenceHz = window.cadenceHz[n - 1],
            stepAmplitude = window.stepAmplitude[n - 1],
            stepSpeedMs = mean(window.stepSpeed),
            stepRegularity = stepRegularity(window.verticalAcc),
            magInclinationDeg = magInclination(window),
            magDisturbedRatio = magDisturbedRatio(window),
            magGyroMismatchDeg = magGyroMismatch(window),
        )
    }

    /**
     * Уточнение положения пика между бинами БПФ параболой по трём точкам.
     * Возвращает множитель к частоте бина.
     *
     * Без этого частота шага квантуется с шагом 50/128 = 0.39 Гц: ходьба
     * в 100 и в 110 шагов в минуту выглядела бы одинаково.
     */
    private fun refinePeak(spectrum: FloatArray, idx: Int): Float {
        if (idx <= 0 || idx >= spectrum.size - 1) return 1f
        val a = spectrum[idx - 1]
        val b = spectrum[idx]
        val c = spectrum[idx + 1]
        val denom = a - 2 * b + c
        if (abs(denom) < 1e-9f) return 1f
        val delta = (0.5f * (a - c) / denom).coerceIn(-0.5f, 0.5f)
        return (idx + delta) / idx
    }

    /** СКЗ производной модуля ускорения, м/с³. */
    private fun jerkRms(accMag: FloatArray): Float {
        if (accMag.size < 2) return 0f
        var s = 0.0
        for (i in 1 until accMag.size) {
            val d = (accMag[i] - accMag[i - 1]) * SensorHub.SAMPLE_RATE_HZ
            s += d.toDouble() * d
        }
        return sqrt(s / (accMag.size - 1)).toFloat()
    }

    /**
     * Регулярность шага по Мое-Нильссену: максимум нормированной автокорреляции
     * вертикального ускорения на лагах шага или двойного шага (0.3–1.2 с).
     * У ровной ходьбы около 0.8, у тряски в транспорте и жестов рукой — около
     * нуля: там нет повторяющегося рисунка.
     */
    fun stepRegularity(signal: FloatArray): Float {
        val n = signal.size
        val m = mean(signal)
        var energy = 0.0
        for (v in signal) energy += (v - m).toDouble() * (v - m)
        energy /= n
        if (energy < MIN_REGULARITY_ENERGY) return 0f
        val minLag = (MIN_STEP_LAG_S * SensorHub.SAMPLE_RATE_HZ).toInt()
        val maxLag = min((MAX_STEP_LAG_S * SensorHub.SAMPLE_RATE_HZ).toInt(), n - 2)
        var best = 0.0
        for (lag in minLag..maxLag) {
            var s = 0.0
            for (i in 0 until n - lag) s += (signal[i] - m).toDouble() * (signal[i + lag] - m)
            // Несмещённая оценка: на больших лагах слагаемых меньше.
            val r = s / (n - lag) / energy
            if (r > best) best = r
        }
        return best.toFloat().coerceIn(0f, 1f)
    }

    private fun magInclination(window: SensorWindow): Float {
        var s = 0.0
        var count = 0
        for (i in 0 until window.size) {
            if (window.magNorm[i] > 0f) {
                s += window.magInclinationDeg[i]
                count++
            }
        }
        return if (count > 0) (s / count).toFloat() else 0f
    }

    private fun magDisturbedRatio(window: SensorWindow): Float {
        var present = 0
        var disturbed = 0
        for (i in 0 until window.size) {
            if (window.magNorm[i] <= 0f) continue
            present++
            if (!window.magTrusted[i]) disturbed++
        }
        return if (present > 0) disturbed.toFloat() / present else -1f
    }

    /**
     * Поворот по компасу минус поворот по гироскопу за окно, по модулю, градусы.
     * Курс по компасу растёт по часовой стрелке, угловая скорость вокруг
     * вертикали положительна против часовой — отсюда знак минус.
     */
    private fun magGyroMismatch(window: SensorWindow): Float {
        val dt = 1.0 / SensorHub.SAMPLE_RATE_HZ
        var compass = 0.0
        var gyro = 0.0
        var pairs = 0
        for (i in 1 until window.size) {
            val h0 = window.headingDeg[i - 1]
            val h1 = window.headingDeg[i]
            if (h0.isNaN() || h1.isNaN()) continue
            var d = (h1 - h0).toDouble()
            if (d > 180) d -= 360
            if (d < -180) d += 360
            compass += d
            gyro += -Math.toDegrees(window.yawRate[i] * dt)
            pairs++
        }
        // Если курс определён меньше чем в половине окна, сравнивать нечего.
        if (pairs < window.size / 2) return 0f
        return abs(compass - gyro).toFloat()
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

    private fun meanAbs(a: FloatArray): Float {
        if (a.isEmpty()) return 0f
        var s = 0.0
        for (v in a) s += abs(v)
        return (s / a.size).toFloat()
    }

    private fun rms(a: FloatArray): Float {
        if (a.isEmpty()) return 0f
        var s = 0.0
        for (v in a) s += v.toDouble() * v
        return sqrt(s / a.size).toFloat()
    }
}
