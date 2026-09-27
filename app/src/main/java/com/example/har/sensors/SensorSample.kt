package com.example.har.sensors

/**
 * Один кадр синхронизированных показаний всех датчиков.
 *
 * Физические датчики Android выдают события с разной и плавающей частотой,
 * поэтому [SensorHub] приводит их к единой сетке 50 Гц: акселерометр выступает
 * источником тактов, остальные датчики подставляются последним известным значением
 * (sample-and-hold). Для датчиков приближения и освещённости это корректно —
 * они событийные и меняются редко.
 */
data class SensorFrame(
    /** Время кадра, мс от начала эпохи (для журнала). */
    val timestampMs: Long,
    /** Время кадра, нс от загрузки устройства (монотонное, для расчёта дельт). */
    val elapsedNs: Long,

    // Акселерометр, м/с² (включая гравитацию)
    val ax: Float, val ay: Float, val az: Float,
    // Гироскоп, рад/с
    val gx: Float, val gy: Float, val gz: Float,
    // Магнитометр, мкТл
    val mx: Float, val my: Float, val mz: Float,

    // Датчик приближения, см (у большинства телефонов бинарный: 0 или maxRange)
    val proximityCm: Float,
    /** true, если датчик приближения перекрыт (телефон в кармане / у уха). */
    val proximityNear: Boolean,
    // Датчик освещённости, лк
    val lightLux: Float,

    // Оценка вектора гравитации (ФНЧ по акселерометру), м/с²
    val gravX: Float, val gravY: Float, val gravZ: Float,
) {
    /** Линейное ускорение — акселерометр за вычетом гравитации. */
    val lx: Float get() = ax - gravX
    val ly: Float get() = ay - gravY
    val lz: Float get() = az - gravZ

    fun toCsvRow(): String = buildString {
        append(timestampMs); append(',')
        append(ax); append(','); append(ay); append(','); append(az); append(',')
        append(gx); append(','); append(gy); append(','); append(gz); append(',')
        append(mx); append(','); append(my); append(','); append(mz); append(',')
        append(proximityCm); append(',')
        append(if (proximityNear) 1 else 0); append(',')
        append(lightLux); append(',')
        append(gravX); append(','); append(gravY); append(','); append(gravZ)
    }

    companion object {
        const val CSV_HEADER =
            "timestamp_ms,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z," +
                "mag_x,mag_y,mag_z,proximity_cm,proximity_near,light_lux,grav_x,grav_y,grav_z"
    }
}

/** Какие из пяти требуемых датчиков реально есть на устройстве. */
data class SensorAvailability(
    val accelerometer: Boolean,
    val gyroscope: Boolean,
    val magnetometer: Boolean,
    val proximity: Boolean,
    val light: Boolean,
) {
    val missing: List<String>
        get() = buildList {
            if (!accelerometer) add("акселерометр")
            if (!gyroscope) add("гироскоп")
            if (!magnetometer) add("магнитометр")
            if (!proximity) add("приближение")
            if (!light) add("освещённость")
        }

    /** Без акселерометра распознавание движения невозможно в принципе. */
    val canRecognize: Boolean get() = accelerometer
}
