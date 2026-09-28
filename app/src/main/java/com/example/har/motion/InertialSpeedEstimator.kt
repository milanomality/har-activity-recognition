package com.example.har.motion

import com.example.har.sensors.SensorHub
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Оценка скорости методом инерциальной навигации (INS) по акселерометру и гироскопу.
 *
 * Конвейер на каждом кадре сетки 50 Гц:
 *  1. **Ориентация.** Угловая скорость интегрируется в кватернион «телефон → Земля».
 *     Кватернион, а не углы Эйлера, — чтобы не было шарнирного замка. Дрейф
 *     гироскопа по наклону гасится комплементарной поправкой по акселерометру
 *     (фильтр Махони): когда модуль ускорения близок к g, акселерометр показывает,
 *     где верх, и ориентация подтягивается к нему.
 *  2. **Компенсация гравитации.** Ускорение поворачивается в земную систему
 *     координат, и из вертикальной оси вычитается g: a_чист = R·a_сыр − (0, 0, g).
 *  3. **Интегрирование.** v ← v + a_чист·Δt.
 *  4. **ZUPT (Zero Velocity Update).** Если телефон неподвижен (модуль ускорения
 *     равен g, вращения нет) дольше [ZUPT_MIN_SECONDS], скорость принудительно
 *     обнуляется. Это единственное, что останавливает накопление ошибки.
 *
 * Ограничение, которое нельзя обойти: между остановками ошибка скорости растёт
 * без предела. Погрешность наклона в 1° даёт паразитное ускорение 0.17 м/с²,
 * то есть 1.7 м/с ошибки уже через 10 с. У телефона в кармане при ходьбе
 * моментов нулевой скорости нет — бедро движется вместе с телом, — поэтому
 * оценка считается надёжной только [MAX_RELIABLE_SECONDS] после последнего ZUPT.
 *
 * Класс не зависит от Android и тестируется на JVM.
 */
class InertialSpeedEstimator(
    sampleRateHz: Int = SensorHub.SAMPLE_RATE_HZ,
) {
    /** Результат обработки одного кадра. */
    data class Estimate(
        /** Горизонтальная скорость, м/с. Вертикальная не входит: по ней не различить активность. */
        val horizontalSpeed: Float,
        /** Сколько секунд прошло с последнего обнуления скорости. */
        val secondsSinceZupt: Float,
        /** Считается ли сейчас телефон неподвижным. */
        val stationary: Boolean,
    ) {
        val reliable: Boolean get() = secondsSinceZupt <= MAX_RELIABLE_SECONDS
    }

    // Кадры приходят на равномерной сетке, поэтому шаг берём номинальный:
    // метки времени кадров, досланных при догоне сетки, совпадают, и Δt по ним был бы нулём.
    private val dt = 1.0 / sampleRateHz

    // Кватернион ориентации «телефон → Земля» (w, x, y, z).
    private var qw = 1.0
    private var qx = 0.0
    private var qy = 0.0
    private var qz = 0.0
    private var initialised = false

    // Скорость в земной системе координат, м/с (ось Z — вверх).
    private var vx = 0.0
    private var vy = 0.0
    private var vz = 0.0

    /** Опорное значение g: уточняется в моменты покоя, у каждого датчика оно своё. */
    private var gRef = STANDARD_G

    private var stillSamples = 0
    private var samplesSinceZupt = 0
    private val zuptMinSamples = (ZUPT_MIN_SECONDS * sampleRateHz).toInt()

    fun reset() {
        qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0
        vx = 0.0; vy = 0.0; vz = 0.0
        gRef = STANDARD_G
        initialised = false
        stillSamples = 0
        samplesSinceZupt = 0
    }

    /**
     * Обрабатывает один кадр.
     *
     * @param ax, ay, az показания акселерометра в осях телефона, м/с² (с гравитацией)
     * @param gx, gy, gz показания гироскопа в осях телефона, рад/с
     */
    fun update(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
    ): Estimate {
        val aNorm = sqrt(ax.toDouble() * ax + ay.toDouble() * ay + az.toDouble() * az)
        if (!initialised) {
            if (aNorm < 1e-3) return Estimate(0f, 0f, false)
            initOrientation(ax / aNorm, ay / aNorm, az / aNorm)
            initialised = true
        }

        // --- детектор покоя для ZUPT ---
        val gyroNorm = sqrt(gx.toDouble() * gx + gy.toDouble() * gy + gz.toDouble() * gz)
        val stillNow = abs(aNorm - gRef) < STILL_ACC_TOLERANCE && gyroNorm < STILL_GYRO_TOLERANCE
        stillSamples = if (stillNow) stillSamples + 1 else 0
        val stationary = stillSamples >= zuptMinSamples

        // --- 1. ориентация: гироскоп + комплементарная поправка по акселерометру ---
        var wx = gx.toDouble()
        var wy = gy.toDouble()
        var wz = gz.toDouble()
        // Акселерометру как указателю «верха» верим, только когда он не сильно
        // отличается от g: во время удара пятки в нём больше движения, чем гравитации.
        if (aNorm > 1e-3 && abs(aNorm - gRef) < TRUST_ACC_TOLERANCE) {
            val kp = if (stationary) KP_STATIONARY else KP_MOVING
            // Куда, по текущей ориентации, должен смотреть «верх» в осях телефона:
            // третья строка матрицы поворота R (R^T · (0, 0, 1)).
            val ux = 2 * (qx * qz - qw * qy)
            val uy = 2 * (qy * qz + qw * qx)
            val uz = qw * qw - qx * qx - qy * qy + qz * qz
            val mx = ax / aNorm
            val my = ay / aNorm
            val mz = az / aNorm
            // Ошибка — векторное произведение измеренного и ожидаемого «верха».
            wx += kp * (my * uz - mz * uy)
            wy += kp * (mz * ux - mx * uz)
            wz += kp * (mx * uy - my * ux)
        }
        integrateGyro(wx, wy, wz)

        // --- 2. компенсация гравитации в земных координатах ---
        val (ex, ey, ez) = rotateToEarth(ax.toDouble(), ay.toDouble(), az.toDouble())
        val linX = ex
        val linY = ey
        val linZ = ez - gRef

        // --- 3–4. интегрирование и ZUPT ---
        if (stationary) {
            vx = 0.0; vy = 0.0; vz = 0.0
            samplesSinceZupt = 0
            // В покое модуль ускорения и есть g этого датчика: калибровка у всех
            // телефонов разная, и разница в 0.1 м/с² сразу ушла бы в дрейф.
            gRef += G_ADAPT_RATE * (aNorm - gRef)
        } else {
            vx += linX * dt
            vy += linY * dt
            vz += linZ * dt
            samplesSinceZupt++
        }

        val horizontal = sqrt(vx * vx + vy * vy).toFloat()
        return Estimate(
            horizontalSpeed = horizontal,
            secondsSinceZupt = (samplesSinceZupt * dt).toFloat(),
            stationary = stationary,
        )
    }

    /** Начальная ориентация: кратчайший поворот, совмещающий измеренный «верх» с осью Z Земли. */
    private fun initOrientation(ux: Double, uy: Double, uz: Double) {
        // Поворот вектора u в (0, 0, 1): q = (1 + u·z, u × z), затем нормировка.
        val dot = uz
        if (dot < -0.9999) {
            // Телефон экраном вниз: поворот на 180° вокруг оси X.
            qw = 0.0; qx = 1.0; qy = 0.0; qz = 0.0
            return
        }
        qw = 1.0 + dot
        qx = uy
        qy = -ux
        qz = 0.0
        normalise()
    }

    /** q ← q + ½ · q ⊗ (0, ω) · Δt, затем нормировка. */
    private fun integrateGyro(wx: Double, wy: Double, wz: Double) {
        val h = 0.5 * dt
        val dw = -qx * wx - qy * wy - qz * wz
        val dx = qw * wx + qy * wz - qz * wy
        val dy = qw * wy - qx * wz + qz * wx
        val dz = qw * wz + qx * wy - qy * wx
        qw += dw * h
        qx += dx * h
        qy += dy * h
        qz += dz * h
        normalise()
    }

    /** Поворот вектора из осей телефона в земные: R(q) · v. */
    private fun rotateToEarth(x: Double, y: Double, z: Double): Triple<Double, Double, Double> {
        val ww = qw * qw; val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
        val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
        val wx = qw * qx; val wy = qw * qy; val wz = qw * qz
        return Triple(
            (ww + xx - yy - zz) * x + 2 * (xy - wz) * y + 2 * (xz + wy) * z,
            2 * (xy + wz) * x + (ww - xx + yy - zz) * y + 2 * (yz - wx) * z,
            2 * (xz - wy) * x + 2 * (yz + wx) * y + (ww - xx - yy + zz) * z,
        )
    }

    private fun normalise() {
        val n = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if (n < 1e-12) { qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0; return }
        qw /= n; qx /= n; qy /= n; qz /= n
    }

    companion object {
        const val STANDARD_G = 9.80665

        /** Сколько секунд после последнего ZUPT скорости ещё можно доверять. */
        const val MAX_RELIABLE_SECONDS = 10f

        /** Минимальная длительность покоя для обнуления скорости, с. */
        const val ZUPT_MIN_SECONDS = 0.2

        /** Допуск модуля ускорения от g для признания покоя, м/с². */
        private const val STILL_ACC_TOLERANCE = 0.3

        /** Допуск угловой скорости для признания покоя, рад/с. */
        private const val STILL_GYRO_TOLERANCE = 0.1

        /**
         * Отклонение |a| от g, при котором акселерометр ещё годится как указатель «верха», м/с².
         * Узкий допуск важен: постоянное ускорение вперёд по одному акселерометру неотличимо
         * от наклона, и при широком допуске фильтр «списал» бы разгон в поворот телефона.
         */
        private const val TRUST_ACC_TOLERANCE = 0.5

        /** Вес поправки по акселерометру в покое: ориентация быстро выравнивается по гравитации. */
        private const val KP_STATIONARY = 2.0

        /** Вес поправки в движении: в основном доверяем гироскопу. */
        private const val KP_MOVING = 0.3

        /** Скорость подстройки опорного g в покое (доля за кадр). */
        private const val G_ADAPT_RATE = 0.01
    }
}
