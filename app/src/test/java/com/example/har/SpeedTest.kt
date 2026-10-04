package com.example.har

import com.example.har.features.WindowStats
import com.example.har.ml.ActivityType
import com.example.har.ml.SpeedFusion
import com.example.har.motion.InertialSpeedEstimator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class SpeedTest {

    private val g = InertialSpeedEstimator.STANDARD_G.toFloat()
    private val rate = 50

    @Test
    fun `лежащий телефон — скорость ноль и срабатывает ZUPT`() {
        val ins = InertialSpeedEstimator(rate)
        var last: InertialSpeedEstimator.Estimate? = null
        repeat(rate * 3) { last = ins.update(0f, 0f, g, 0f, 0f, 0f) }
        val e = requireNotNull(last)
        assertTrue(e.stationary)
        assertEquals(0f, e.horizontalSpeed, 1e-4f)
        assertEquals(0f, e.secondsSinceZupt, 1e-4f)
    }

    @Test
    fun `постоянное ускорение вперёд интегрируется в скорость`() {
        val ins = InertialSpeedEstimator(rate)
        repeat(rate) { ins.update(0f, 0f, g, 0f, 0f, 0f) }
        // 1 с с ускорением 4 м/с² вдоль оси X: ожидаем 4 м/с. Слабый разгон
        // (|a| в пределах 0.5 м/с² от g) фильтр частично принимает за наклон —
        // по одному акселерометру эти случаи неразличимы.
        var last: InertialSpeedEstimator.Estimate? = null
        repeat(rate) { last = ins.update(4f, 0f, g, 0f, 0f, 0f) }
        val e = requireNotNull(last)
        assertEquals(4f, e.horizontalSpeed, 0.1f)
        assertFalse(e.stationary)
        assertTrue(e.reliable)
    }

    @Test
    fun `наклонённый телефон — гравитация вычитается и не даёт скорости`() {
        val ins = InertialSpeedEstimator(rate)
        // Телефон неподвижен, но наклонён на 40° вокруг оси X: g распределена по Y и Z.
        val a = Math.toRadians(40.0)
        val ay = (g * sin(a)).toFloat()
        val az = (g * cos(a)).toFloat()
        var last: InertialSpeedEstimator.Estimate? = null
        repeat(rate * 5) { last = ins.update(0f, ay, az, 0f, 0f, 0f) }
        assertEquals(0f, requireNotNull(last).horizontalSpeed, 0.05f)
    }

    @Test
    fun `после сильного вращения скорость ненадёжна`() {
        val ins = InertialSpeedEstimator(rate)
        repeat(rate) { ins.update(0f, 0f, g, 0f, 0f, 0f) }
        // 4 с вращения 2 рад/с (мах рукой) — 8 рад, больше допустимого оборота.
        var last: InertialSpeedEstimator.Estimate? = null
        repeat(rate * 4) { last = ins.update(1f, 0f, g, 0f, 0f, 2f) }
        val e = requireNotNull(last)
        assertTrue(e.secondsSinceZupt < InertialSpeedEstimator.MAX_RELIABLE_SECONDS)
        assertFalse("повернулся на ${e.rotationSinceZupt} рад", e.reliable)
    }

    @Test
    fun `невозможная скорость почти запрещает класс`() {
        // 25 км/ч — ходьба и лестница физически невозможны.
        assertEquals(SpeedFusion.IMPOSSIBLE, SpeedFusion.likelihood(ActivityType.WALKING, 7f), 0f)
        assertEquals(SpeedFusion.IMPOSSIBLE, SpeedFusion.likelihood(ActivityType.STAIRS_UP, 7f), 0f)
        assertEquals(1f, SpeedFusion.likelihood(ActivityType.CYCLING, 7f), 0f)
    }

    @Test
    fun `остановка обнуляет накопленную скорость`() {
        val ins = InertialSpeedEstimator(rate)
        repeat(rate) { ins.update(0f, 0f, g, 0f, 0f, 0f) }
        repeat(rate) { ins.update(3f, 0f, g, 0f, 0f, 0f) }
        var last: InertialSpeedEstimator.Estimate? = null
        repeat(rate / 2) { last = ins.update(0f, 0f, g, 0f, 0f, 0f) }
        val e = requireNotNull(last)
        assertTrue(e.stationary)
        assertEquals(0f, e.horizontalSpeed, 1e-4f)
    }

    @Test
    fun `без остановок оценка перестаёт считаться надёжной`() {
        val ins = InertialSpeedEstimator(rate)
        var last: InertialSpeedEstimator.Estimate? = null
        // Непрерывная тряска: ZUPT не срабатывает ни разу.
        repeat(rate * 15) { i -> last = ins.update(0f, 0f, g + if (i % 2 == 0) 2f else -2f, 0f, 0f, 0f) }
        assertFalse(requireNotNull(last).reliable)
    }

    @Test
    fun `скорость бега поднимает бег над ходьбой`() {
        val probs = FloatArray(ActivityType.COUNT).apply {
            this[ActivityType.WALKING.id] = 0.55f
            this[ActivityType.RUNNING.id] = 0.45f
        }
        val out = SpeedFusion.apply(probs, stats(speed = 3.5f, sinceZupt = 2f))
        assertTrue(out[ActivityType.RUNNING.id] > out[ActivityType.WALKING.id])
        assertEquals(1f, out.sum(), 1e-4f)
    }

    @Test
    fun `ненадёжная скорость не меняет решение`() {
        val probs = FloatArray(ActivityType.COUNT).apply {
            this[ActivityType.WALKING.id] = 0.55f
            this[ActivityType.RUNNING.id] = 0.45f
        }
        val out = SpeedFusion.apply(probs, stats(speed = 3.5f, sinceZupt = 60f))
        assertEquals(0.55f, out[ActivityType.WALKING.id], 1e-6f)
    }

    @Test
    fun `скорость ослабляет класс, но не запрещает его`() {
        // 11 км/ч — быстро для ходьбы, но не вдвое выше её предела: ослабить, не запретить.
        assertEquals(SpeedFusion.FLOOR, SpeedFusion.likelihood(ActivityType.WALKING, 3f), 1e-6f)
        assertEquals(1f, SpeedFusion.likelihood(ActivityType.WALKING, 1.4f), 1e-6f)
    }

    private fun stats(speed: Float, sinceZupt: Float) = WindowStats(
        accMagMean = g, accMagStd = 0f, linAccRms = 0f,
        gyroMagMean = 0f, gyroMagStd = 0f, magMagMean = 45f, magMagStd = 0f,
        lightLuxMean = 200f, lightAvailable = true,
        proximityNearRatio = 0f, proximityAvailable = true,
        dominantFreqHz = 0f, dominantPower = 0f, spectralEntropy = 0f,
        tiltDeg = 0f, orientationStd = 0f, zeroCrossingRate = 0f,
        speedMs = speed, secondsSinceZupt = sinceZupt,
    )
}
