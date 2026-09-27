package com.example.har.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow

/**
 * Единая точка доступа ко всем пяти датчикам.
 *
 * Задачи класса:
 *  1. подписаться на акселерометр, гироскоп, магнитометр, датчик приближения и освещённости;
 *  2. привести разнородные потоки событий к равномерной сетке [SAMPLE_RATE_HZ];
 *  3. оценить вектор гравитации фильтром нижних частот, чтобы отделить
 *     ориентацию телефона от собственно движения.
 */
class SensorHub(context: Context) {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val proximity = sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY)
    private val light = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)

    val availability = SensorAvailability(
        accelerometer = accelerometer != null,
        gyroscope = gyroscope != null,
        magnetometer = magnetometer != null,
        proximity = proximity != null,
        light = light != null,
    )

    /** Максимальная дальность датчика приближения, см (нужна для нормировки). */
    val proximityMaxRange: Float = proximity?.maximumRange ?: DEFAULT_PROXIMITY_RANGE_CM

    /**
     * Поток кадров с частотой [SAMPLE_RATE_HZ].
     *
     * Реализован как [callbackFlow] с буфером и политикой DROP_OLDEST: если
     * потребитель (инференс) отстаёт, мы теряем старые кадры, а не копим их в памяти.
     */
    fun frames(): Flow<SensorFrame> = rawFrames()
        .buffer(capacity = SAMPLE_RATE_HZ * 4, onBufferOverflow = BufferOverflow.DROP_OLDEST)

    private fun rawFrames(): Flow<SensorFrame> = callbackFlow {
        if (accelerometer == null) {
            close(IllegalStateException("На устройстве нет акселерометра"))
            return@callbackFlow
        }

        // Последние известные значения каждого датчика (sample-and-hold).
        var ax = 0f; var ay = 0f; var az = 0f
        var gx = 0f; var gy = 0f; var gz = 0f
        var mx = 0f; var my = 0f; var mz = 0f
        // Отрицательное значение = «датчика нет». Ноль здесь недопустим:
        // у датчика приближения ноль означает «перекрыт», и отсутствующий
        // датчик выглядел бы как постоянно закрытый.
        var proximityCm = if (proximity != null) proximityMaxRange else -1f
        var lightLux = -1f

        // ФНЧ для выделения гравитации. При 50 Гц alpha = 0.9 даёт
        // постоянную времени около 0.2 с — этого хватает, чтобы гравитация
        // следила за поворотом телефона, но не реагировала на шаги.
        var gravX = 0f; var gravY = 0f; var gravZ = 0f
        var gravityInitialised = false

        var nextEmitNs = 0L
        val periodNs = 1_000_000_000L / SAMPLE_RATE_HZ

        val listener = object : SensorEventListener {
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_GYROSCOPE -> {
                        gx = event.values[0]; gy = event.values[1]; gz = event.values[2]
                        return
                    }
                    Sensor.TYPE_MAGNETIC_FIELD -> {
                        mx = event.values[0]; my = event.values[1]; mz = event.values[2]
                        return
                    }
                    Sensor.TYPE_PROXIMITY -> {
                        proximityCm = event.values[0]
                        return
                    }
                    Sensor.TYPE_LIGHT -> {
                        lightLux = event.values[0]
                        return
                    }
                    Sensor.TYPE_ACCELEROMETER -> {
                        ax = event.values[0]; ay = event.values[1]; az = event.values[2]
                    }
                    else -> return
                }

                // Дальше — только для событий акселерометра: он задаёт такт.
                if (!gravityInitialised) {
                    gravX = ax; gravY = ay; gravZ = az
                    gravityInitialised = true
                } else {
                    gravX = GRAVITY_ALPHA * gravX + (1 - GRAVITY_ALPHA) * ax
                    gravY = GRAVITY_ALPHA * gravY + (1 - GRAVITY_ALPHA) * ay
                    gravZ = GRAVITY_ALPHA * gravZ + (1 - GRAVITY_ALPHA) * az
                }

                val eventNs = event.timestamp
                if (nextEmitNs == 0L) nextEmitNs = eventNs

                // Догоняем сетку. Ограничиваем число повторов, чтобы при
                // длинной паузе (например, после глубокого сна) не выплюнуть
                // тысячи дублирующих кадров.
                var emitted = 0
                while (eventNs >= nextEmitNs && emitted < MAX_CATCH_UP) {
                    val nowMs = System.currentTimeMillis() -
                        (SystemClock.elapsedRealtimeNanos() - eventNs) / 1_000_000L
                    trySend(
                        SensorFrame(
                            timestampMs = nowMs,
                            elapsedNs = eventNs,
                            ax = ax, ay = ay, az = az,
                            gx = gx, gy = gy, gz = gz,
                            mx = mx, my = my, mz = mz,
                            proximityCm = proximityCm,
                            proximityNear = proximityCm >= 0f &&
                                proximityCm < proximityMaxRange * PROXIMITY_NEAR_FRACTION,
                            lightLux = lightLux,
                            gravX = gravX, gravY = gravY, gravZ = gravZ,
                        )
                    )
                    nextEmitNs += periodNs
                    emitted++
                }
                if (emitted == MAX_CATCH_UP) {
                    // Слишком большой разрыв — пересинхронизируемся с текущим временем.
                    nextEmitNs = eventNs + periodNs
                }
            }
        }

        // Акселерометр опрашиваем вдвое чаще целевой частоты: так сетка 50 Гц
        // набирается без пропусков даже при джиттере драйвера.
        sensorManager.registerListener(listener, accelerometer, ACCEL_PERIOD_US)
        gyroscope?.let { sensorManager.registerListener(listener, it, ACCEL_PERIOD_US) }
        magnetometer?.let { sensorManager.registerListener(listener, it, SLOW_PERIOD_US) }
        // Приближение и освещённость — событийные датчики, задаём редкий опрос.
        proximity?.let { sensorManager.registerListener(listener, it, SensorManager.SENSOR_DELAY_NORMAL) }
        light?.let { sensorManager.registerListener(listener, it, SensorManager.SENSOR_DELAY_NORMAL) }

        awaitClose { sensorManager.unregisterListener(listener) }
    }

    companion object {
        /** Частота единой сетки, Гц. Совпадает с частотой обучающего датасета. */
        const val SAMPLE_RATE_HZ = 50

        private const val ACCEL_PERIOD_US = 10_000   // 100 Гц
        private const val SLOW_PERIOD_US = 20_000    // 50 Гц
        private const val GRAVITY_ALPHA = 0.9f
        private const val MAX_CATCH_UP = 4
        private const val DEFAULT_PROXIMITY_RANGE_CM = 5f

        /**
         * Доля от maximumRange, ниже которой считаем датчик перекрытым.
         * У большинства телефонов датчик бинарный (0 / maxRange), у части —
         * аналоговый, поэтому используем порог, а не сравнение с нулём.
         */
        private const val PROXIMITY_NEAR_FRACTION = 0.5f
    }
}
