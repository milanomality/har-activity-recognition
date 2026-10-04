package com.example.har.sensors

/**
 * Окно сигналов фиксированной длины — единица, на которой работает классификатор.
 *
 * Каналы хранятся «по столбцам» (структура массивов, а не массив структур):
 * так признаки и нормировка считаются без аллокаций на каждый отсчёт.
 */
class SensorWindow(
    val startTimeMs: Long,
    val endTimeMs: Long,
    val size: Int,
    val accX: FloatArray, val accY: FloatArray, val accZ: FloatArray,
    val gyroX: FloatArray, val gyroY: FloatArray, val gyroZ: FloatArray,
    val magX: FloatArray, val magY: FloatArray, val magZ: FloatArray,
    val proximity: FloatArray,
    val proximityNear: BooleanArray,
    val light: FloatArray,
    val gravX: FloatArray, val gravY: FloatArray, val gravZ: FloatArray,
    /** Горизонтальная скорость по инерциальной навигации, м/с. В модель не подаётся. */
    val speed: FloatArray = FloatArray(size),
    /** Секунды с последнего ZUPT; −1 — скорость не считалась. */
    val secondsSinceZupt: FloatArray = FloatArray(size) { -1f },
    // Производные покадровые величины (см. поля SensorFrame). В модель не подаются.
    val rotationSinceZupt: FloatArray = FloatArray(size),
    val verticalAcc: FloatArray = FloatArray(size),
    val horizontalAcc: FloatArray = FloatArray(size),
    val yawRate: FloatArray = FloatArray(size),
    val insTiltDeg: FloatArray = FloatArray(size),
    val magNorm: FloatArray = FloatArray(size),
    val magInclinationDeg: FloatArray = FloatArray(size),
    val headingDeg: FloatArray = FloatArray(size) { Float.NaN },
    val magTrusted: BooleanArray = BooleanArray(size),
    val stepCount: IntArray = IntArray(size),
    val cadenceHz: FloatArray = FloatArray(size),
    val stepAmplitude: FloatArray = FloatArray(size),
    val stepSpeed: FloatArray = FloatArray(size),
) {
    /**
     * Канал по имени из `model_meta.json`.
     *
     * Модель может быть обучена на подмножестве каналов: в публичных
     * датасетах HAR магнитометра нет вообще, и заставлять сеть принимать
     * три нулевых канала — значит тратить параметры впустую. Поэтому
     * набор входных каналов задаёт обученная модель, а не приложение.
     */
    fun channelByName(name: String): FloatArray? = when (name) {
        "acc_x" -> accX
        "acc_y" -> accY
        "acc_z" -> accZ
        "gyro_x" -> gyroX
        "gyro_y" -> gyroY
        "gyro_z" -> gyroZ
        "mag_x" -> magX
        "mag_y" -> magY
        "mag_z" -> magZ
        else -> null
    }

    companion object {
        /** Полный набор каналов движения в каноническом порядке. */
        val ALL_MOTION_CHANNELS = listOf(
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
        )

        /** Длина окна в отсчётах: 128 при 50 Гц = 2.56 с (стандарт датасетов HAR). */
        const val WINDOW_SIZE = 128

        /** Сдвиг окна: 64 отсчёта = перекрытие 50 %, предсказание раз в 1.28 с. */
        const val WINDOW_STRIDE = 64
    }
}

/**
 * Кольцевой буфер, нарезающий поток кадров на перекрывающиеся окна.
 *
 * Не потокобезопасен: предполагается вызов из одной корутины-потребителя.
 */
class SlidingWindowBuffer(
    private val windowSize: Int = SensorWindow.WINDOW_SIZE,
    private val stride: Int = SensorWindow.WINDOW_STRIDE,
) {
    private val buffer = arrayOfNulls<SensorFrame>(windowSize)
    private var writeIndex = 0
    private var filled = 0
    private var sinceLastWindow = 0

    /**
     * Добавляет кадр и возвращает готовое окно, если накопилось достаточно данных.
     * В остальных случаях возвращает null.
     */
    fun push(frame: SensorFrame): SensorWindow? {
        buffer[writeIndex] = frame
        writeIndex = (writeIndex + 1) % windowSize
        if (filled < windowSize) filled++
        sinceLastWindow++

        if (filled < windowSize || sinceLastWindow < stride) return null
        sinceLastWindow = 0
        return snapshot()
    }

    fun reset() {
        writeIndex = 0
        filled = 0
        sinceLastWindow = 0
        buffer.fill(null)
    }

    /** Разворачивает кольцевой буфер в линейное окно, начиная с самого старого кадра. */
    private fun snapshot(): SensorWindow {
        val n = windowSize
        val accX = FloatArray(n); val accY = FloatArray(n); val accZ = FloatArray(n)
        val gyroX = FloatArray(n); val gyroY = FloatArray(n); val gyroZ = FloatArray(n)
        val magX = FloatArray(n); val magY = FloatArray(n); val magZ = FloatArray(n)
        val prox = FloatArray(n); val near = BooleanArray(n); val lux = FloatArray(n)
        val grX = FloatArray(n); val grY = FloatArray(n); val grZ = FloatArray(n)
        val speed = FloatArray(n); val sinceZupt = FloatArray(n) { -1f }
        val rotZupt = FloatArray(n)
        val vert = FloatArray(n); val horiz = FloatArray(n); val yaw = FloatArray(n)
        val tilt = FloatArray(n); val mNorm = FloatArray(n); val mDip = FloatArray(n)
        val heading = FloatArray(n) { Float.NaN }; val mTrusted = BooleanArray(n)
        val steps = IntArray(n); val cadence = FloatArray(n)
        val stepAmp = FloatArray(n); val stepSpeed = FloatArray(n)

        for (i in 0 until n) {
            val f = buffer[(writeIndex + i) % n] ?: continue
            accX[i] = f.ax; accY[i] = f.ay; accZ[i] = f.az
            gyroX[i] = f.gx; gyroY[i] = f.gy; gyroZ[i] = f.gz
            magX[i] = f.mx; magY[i] = f.my; magZ[i] = f.mz
            prox[i] = f.proximityCm; near[i] = f.proximityNear; lux[i] = f.lightLux
            grX[i] = f.gravX; grY[i] = f.gravY; grZ[i] = f.gravZ
            speed[i] = f.speedMs; sinceZupt[i] = f.secondsSinceZupt
            rotZupt[i] = f.rotationSinceZupt
            vert[i] = f.verticalAcc; horiz[i] = f.horizontalAcc; yaw[i] = f.yawRate
            tilt[i] = f.insTiltDeg; mNorm[i] = f.magNorm; mDip[i] = f.magInclinationDeg
            heading[i] = f.headingDeg; mTrusted[i] = f.magTrusted
            steps[i] = f.stepCount; cadence[i] = f.cadenceHz
            stepAmp[i] = f.stepAmplitude; stepSpeed[i] = f.stepSpeedMs
        }

        val oldest = buffer[writeIndex % n]
        val newest = buffer[(writeIndex + n - 1) % n]
        return SensorWindow(
            startTimeMs = oldest?.timestampMs ?: 0L,
            endTimeMs = newest?.timestampMs ?: 0L,
            size = n,
            accX = accX, accY = accY, accZ = accZ,
            gyroX = gyroX, gyroY = gyroY, gyroZ = gyroZ,
            magX = magX, magY = magY, magZ = magZ,
            proximity = prox, proximityNear = near, light = lux,
            gravX = grX, gravY = grY, gravZ = grZ,
            speed = speed, secondsSinceZupt = sinceZupt, rotationSinceZupt = rotZupt,
            verticalAcc = vert, horizontalAcc = horiz, yawRate = yaw,
            insTiltDeg = tilt, magNorm = mNorm, magInclinationDeg = mDip,
            headingDeg = heading, magTrusted = mTrusted,
            stepCount = steps, cadenceHz = cadence,
            stepAmplitude = stepAmp, stepSpeed = stepSpeed,
        )
    }
}
