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

        for (i in 0 until n) {
            val f = buffer[(writeIndex + i) % n] ?: continue
            accX[i] = f.ax; accY[i] = f.ay; accZ[i] = f.az
            gyroX[i] = f.gx; gyroY[i] = f.gy; gyroZ[i] = f.gz
            magX[i] = f.mx; magY[i] = f.my; magZ[i] = f.mz
            prox[i] = f.proximityCm; near[i] = f.proximityNear; lux[i] = f.lightLux
            grX[i] = f.gravX; grY[i] = f.gravY; grZ[i] = f.gravZ
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
        )
    }
}
