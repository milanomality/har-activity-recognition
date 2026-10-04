package com.example.har.motion

import com.example.har.sensors.SensorHub
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Оценка скорости методом инерциальной навигации (INS) по акселерометру, гироскопу
 * и магнитометру.
 *
 * Конвейер на каждом кадре сетки 50 Гц:
 *  1. **Ориентация.** Угловая скорость интегрируется в кватернион «телефон → Земля».
 *     Кватернион, а не углы Эйлера, — чтобы не было шарнирного замка. Дрейф
 *     гироскопа по наклону гасится комплементарной поправкой по акселерометру
 *     (фильтр Махони): когда модуль ускорения близок к g, акселерометр показывает,
 *     где верх, и ориентация подтягивается к нему.
 *  2. **Курс по магнитометру.** Акселерометр ничего не говорит о повороте вокруг
 *     вертикали — рыскание держится только на гироскопе и уплывает. Магнитное поле
 *     Земли даёт недостающую опору: пока его модуль и наклонение совпадают
 *     с опорными (поле не искажено металлом или током), рыскание подтягивается
 *     к нему. Искажённое поле в поправку не идёт, зато само искажение —
 *     полезный признак транспорта.
 *  3. **Компенсация гравитации.** Ускорение поворачивается в земную систему
 *     координат, и из вертикальной оси вычитается g: a_чист = R·a_сыр − (0, 0, g).
 *  4. **Интегрирование.** v ← v + a_чист·Δt.
 *  5. **ZUPT (Zero Velocity Update).** Если телефон неподвижен (модуль ускорения
 *     равен g, вращения нет) дольше [ZUPT_MIN_SECONDS], скорость принудительно
 *     обнуляется. Это единственное, что останавливает накопление ошибки.
 *
 * Попутно, раз ориентация уже известна, класс раскладывает движение по земным
 * осям: вертикальное и горизонтальное ускорение, угловая скорость вокруг
 * вертикали, наклон телефона, курс и наклонение магнитного поля. Эти величины
 * не зависят от того, как телефон повёрнут в кармане или в руке, — в отличие
 * от сырых осей датчиков.
 *
 * Ограничение, которое нельзя обойти: между остановками ошибка скорости растёт
 * без предела. Погрешность наклона в 1° даёт паразитное ускорение 0.17 м/с²,
 * то есть 1.7 м/с ошибки уже через 10 с. У телефона в кармане при ходьбе
 * моментов нулевой скорости нет — бедро движется вместе с телом, — поэтому
 * оценка считается надёжной только [MAX_RELIABLE_SECONDS] после последнего ZUPT.
 * Для ходьбы есть вторая, независимая оценка — по шагам ([StepDetector]).
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
        /** Ускорение по вертикали Земли без гравитации, м/с², вверх — плюс. */
        val verticalAcc: Float = 0f,
        /** Модуль ускорения в горизонтальной плоскости, м/с². */
        val horizontalAcc: Float = 0f,
        /** Угловая скорость вокруг вертикали Земли (поворот корпуса), рад/с, против часовой — плюс. */
        val yawRate: Float = 0f,
        /** Угол между осью Z телефона и вертикалью по оценке ориентации, градусы. */
        val tiltDeg: Float = 0f,
        /** Модуль магнитного поля, мкТл; 0 — магнитометра нет. */
        val magNorm: Float = 0f,
        /** Магнитное наклонение: на сколько градусов поле уходит ниже горизонта. */
        val magInclinationDeg: Float = 0f,
        /** Курс оси Y телефона по компасу, 0–360°; NaN — нет магнитометра или ось Y вертикальна. */
        val headingDeg: Float = Float.NaN,
        /** Поле совпадает с опорным и используется для поправки рыскания. */
        val magTrusted: Boolean = false,
        /** На сколько радиан телефон повернулся с последнего ZUPT (интеграл модуля угловой скорости). */
        val rotationSinceZupt: Float = 0f,
    ) {
        val reliable: Boolean get() = isReliable(secondsSinceZupt, rotationSinceZupt)
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
    private var rotationSinceZupt = 0.0
    private val zuptMinSamples = (ZUPT_MIN_SECONDS * sampleRateHz).toInt()

    // Опорное магнитное поле: модуль и наклонение, к которым оно сходится на
    // открытом месте, и горизонтальное направление в земных осях — к нему
    // подтягивается рыскание. Направление запоминается заново после каждого
    // долгого искажения: в другом помещении поле может смотреть иначе.
    private var magRefNorm = 0.0
    private var magRefDip = 0.0
    private var magRefHx = 0.0
    private var magRefHy = 0.0
    private var magRefValid = false
    private var magUntrustedSamples = 0
    private val magRelockSamples = (MAG_RELOCK_SECONDS * sampleRateHz).toInt()

    fun reset() {
        qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0
        vx = 0.0; vy = 0.0; vz = 0.0
        gRef = STANDARD_G
        initialised = false
        stillSamples = 0
        samplesSinceZupt = 0
        rotationSinceZupt = 0.0
        magRefNorm = 0.0
        magRefDip = 0.0
        magRefValid = false
        magUntrustedSamples = 0
    }

    /** Направление оси Y телефона в земных осях фильтра (без привязки к северу), градусы. */
    fun yawDeg(): Double {
        val (fx, fy, _) = rotateToEarth(0.0, 1.0, 0.0)
        return Math.toDegrees(atan2(fy, fx))
    }

    /**
     * Обрабатывает один кадр.
     *
     * @param ax, ay, az показания акселерометра в осях телефона, м/с² (с гравитацией)
     * @param gx, gy, gz показания гироскопа в осях телефона, рад/с
     * @param mx, my, mz показания магнитометра, мкТл; нули — магнитометра нет
     */
    fun update(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        mx: Float = 0f, my: Float = 0f, mz: Float = 0f,
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
            // Куда, по текущей ориентации, должен смотреть «верх» в осях телефона.
            val (ux, uy, uz) = bodyUp()
            val mx0 = ax / aNorm
            val my0 = ay / aNorm
            val mz0 = az / aNorm
            // Ошибка — векторное произведение измеренного и ожидаемого «верха».
            wx += kp * (my0 * uz - mz0 * uy)
            wy += kp * (mz0 * ux - mx0 * uz)
            wz += kp * (mx0 * uy - my0 * ux)
        }

        // --- 2. поправка рыскания по магнитному полю ---
        val mag = magnetic(mx.toDouble(), my.toDouble(), mz.toDouble())
        if (mag.trusted) {
            // Синус угла от текущего горизонтального направления поля до опорного
            // (в земных осях) — Z-компонента их векторного произведения.
            val hn = sqrt(mag.hx * mag.hx + mag.hy * mag.hy)
            val rn = sqrt(magRefHx * magRefHx + magRefHy * magRefHy)
            if (hn > 1e-6 && rn > 1e-6) {
                val err = (mag.hx * magRefHy - mag.hy * magRefHx) / (hn * rn)
                // Поворот вокруг земной вертикали в осях телефона — поворот
                // вокруг вектора «верх», записанного в осях телефона.
                val (ux, uy, uz) = bodyUp()
                wx += KP_MAG * err * ux
                wy += KP_MAG * err * uy
                wz += KP_MAG * err * uz
            }
        }
        integrateGyro(wx, wy, wz)

        // --- 3. компенсация гравитации в земных координатах ---
        val (ex, ey, ez) = rotateToEarth(ax.toDouble(), ay.toDouble(), az.toDouble())
        val linX = ex
        val linY = ey
        val linZ = ez - gRef

        // --- 4–5. интегрирование и ZUPT ---
        if (stationary) {
            vx = 0.0; vy = 0.0; vz = 0.0
            samplesSinceZupt = 0
            rotationSinceZupt = 0.0
            // В покое модуль ускорения и есть g этого датчика: калибровка у всех
            // телефонов разная, и разница в 0.1 м/с² сразу ушла бы в дрейф.
            gRef += G_ADAPT_RATE * (aNorm - gRef)
        } else {
            vx += linX * dt
            vy += linY * dt
            vz += linZ * dt
            samplesSinceZupt++
            rotationSinceZupt += gyroNorm * dt
        }

        val horizontal = sqrt(vx * vx + vy * vy).toFloat()
        val (ux, uy, uz) = bodyUp()
        return Estimate(
            horizontalSpeed = horizontal,
            secondsSinceZupt = (samplesSinceZupt * dt).toFloat(),
            stationary = stationary,
            verticalAcc = linZ.toFloat(),
            horizontalAcc = sqrt(linX * linX + linY * linY).toFloat(),
            // Проекция угловой скорости на вертикаль: насколько корпус поворачивает
            // «по курсу», как бы телефон ни лежал в кармане.
            yawRate = (gx * ux + gy * uy + gz * uz).toFloat(),
            tiltDeg = Math.toDegrees(acos(uz.coerceIn(-1.0, 1.0))).toFloat(),
            magNorm = mag.norm.toFloat(),
            magInclinationDeg = mag.dipDeg.toFloat(),
            headingDeg = mag.headingDeg.toFloat(),
            magTrusted = mag.trusted,
            rotationSinceZupt = rotationSinceZupt.toFloat(),
        )
    }

    /** Магнитное поле в земных осях и решение, можно ли ему верить. */
    private class Magnetic(
        val norm: Double,
        val dipDeg: Double,
        val hx: Double,
        val hy: Double,
        val headingDeg: Double,
        val trusted: Boolean,
    )

    private fun magnetic(mx: Double, my: Double, mz: Double): Magnetic {
        val norm = sqrt(mx * mx + my * my + mz * mz)
        if (norm < MAG_MIN_UT) return Magnetic(0.0, 0.0, 0.0, 0.0, Double.NaN, false)

        val (hx, hy, hz) = rotateToEarth(mx, my, mz)
        val hNorm = sqrt(hx * hx + hy * hy)
        // Наклонение: насколько поле уходит вниз от горизонта (в северном полушарии — плюс).
        val dip = Math.toDegrees(atan2(-hz, hNorm))

        // Курс: угол от горизонтальной проекции поля до проекции оси Y телефона,
        // по часовой стрелке, как у компаса. От рыскания фильтра не зависит:
        // поле и ось телефона поворачиваются вместе.
        val (fx, fy, _) = rotateToEarth(0.0, 1.0, 0.0)
        val fNorm = sqrt(fx * fx + fy * fy)
        val heading = if (hNorm > 1e-6 && fNorm > FORWARD_MIN_HORIZONTAL) {
            val cross = hx * fy - hy * fx
            val dot = hx * fx + hy * fy
            (Math.toDegrees(atan2(-cross, dot)) + 360.0) % 360.0
        } else {
            Double.NaN
        }

        if (magRefNorm == 0.0) {
            magRefNorm = norm
            magRefDip = dip
        }
        val cleanNow = norm in MAG_EARTH_MIN_UT..MAG_EARTH_MAX_UT &&
            abs(norm - magRefNorm) < MAG_NORM_TOLERANCE_UT &&
            abs(dip - magRefDip) < MAG_DIP_TOLERANCE_DEG

        // Опорные модуль и наклонение ползут за полем медленно (десятки секунд):
        // так они переживают смену места, но не подстраиваются под короткое искажение.
        magRefNorm += MAG_REF_ADAPT_RATE * (norm - magRefNorm)
        magRefDip += MAG_REF_ADAPT_RATE * (dip - magRefDip)

        var trusted = false
        if (cleanNow) {
            if (!magRefValid || magUntrustedSamples > magRelockSamples) {
                // Первое чистое поле или чистое поле после долгого искажения:
                // запоминаем направление, а не тянем ориентацию к старому.
                magRefHx = hx
                magRefHy = hy
                magRefValid = true
            } else {
                trusted = true
            }
            magUntrustedSamples = 0
        } else {
            magUntrustedSamples++
        }
        return Magnetic(norm, dip, hx, hy, heading, trusted)
    }

    /** Вертикаль Земли в осях телефона: третья строка матрицы поворота, R^T · (0, 0, 1). */
    private fun bodyUp(): Triple<Double, Double, Double> = Triple(
        2 * (qx * qz - qw * qy),
        2 * (qy * qz + qw * qx),
        qw * qw - qx * qx - qy * qy + qz * qz,
    )

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

        /**
         * Сколько телефон может повернуться с последнего ZUPT, пока скорости
         * можно доверять, рад (≈ один оборот). Ошибка ориентации растёт с каждым
         * поворотом: неверный масштаб гироскопа в 1 % на обороте даёт 3.6°
         * наклона, то есть 0.6 м/с² ложного ускорения. При махе рукой телефон
         * поворачивается на 2 рад/с — и за 10 с навигация «разгонялась» до 25 км/ч.
         */
        const val MAX_RELIABLE_ROTATION_RAD = 6f

        fun isReliable(secondsSinceZupt: Float, rotationSinceZupt: Float): Boolean =
            secondsSinceZupt in 0f..MAX_RELIABLE_SECONDS && rotationSinceZupt <= MAX_RELIABLE_ROTATION_RAD

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

        /** Вес поправки рыскания по магнитному полю. */
        private const val KP_MAG = 0.5

        /** Ниже этого модуля считаем, что магнитометра нет (нули от SensorHub), мкТл. */
        private const val MAG_MIN_UT = 1.0

        /** Диапазон модуля поля Земли: от экватора до полюсов, мкТл. */
        const val MAG_EARTH_MIN_UT = 20.0
        const val MAG_EARTH_MAX_UT = 70.0

        /** Допуск модуля поля от опорного, мкТл: больше — рядом металл или ток. */
        private const val MAG_NORM_TOLERANCE_UT = 6.0

        /** Допуск наклонения от опорного, градусы. */
        private const val MAG_DIP_TOLERANCE_DEG = 8.0

        /** Подстройка опорного поля: постоянная времени около 40 с при 50 Гц. */
        private const val MAG_REF_ADAPT_RATE = 0.0005

        /** После искажения дольше этого опорное направление запоминается заново, с. */
        private const val MAG_RELOCK_SECONDS = 2.0

        /** Если ось Y смотрит почти вертикально, курс не определён. */
        private const val FORWARD_MIN_HORIZONTAL = 0.2
    }
}
