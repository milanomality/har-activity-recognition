package com.example.har

import com.example.har.features.WindowStats
import com.example.har.ml.ActivityType
import com.example.har.ml.HeuristicClassifier
import com.example.har.ml.PhonePlacement
import com.example.har.ml.ProbabilitySmoother
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Тесты резервного классификатора и сглаживания.
 *
 * Эвристика — не просто заглушка: она работает, пока не обучена модель,
 * и служит базовой линией в отчёте. Её пороги должны давать ожидаемые
 * ответы на характерных наборах признаков.
 */
class ClassifierTest {

    @Test
    fun `покой распознаётся по отсутствию движения`() {
        val (activity, _) = HeuristicClassifier.classifyActivity(
            stats(linAccRms = 0.05f, gyroMagStd = 0.02f, dominantFreqHz = 0f)
        )
        assertEquals(ActivityType.STILL, activity)
    }

    @Test
    fun `ходьба распознаётся по пику около двух герц`() {
        val (activity, probs) = HeuristicClassifier.classifyActivity(
            stats(
                linAccRms = 2.0f,
                gyroMagStd = 0.4f,
                dominantFreqHz = 1.8f,
                dominantPower = 0.5f,
                spectralEntropy = 1.0f,
            )
        )
        assertEquals(ActivityType.WALKING, activity)
        assertTrue("ходьба должна выигрывать уверенно", probs[ActivityType.WALKING.id] > 0.4f)
    }

    @Test
    fun `бег отличается от ходьбы темпом и амплитудой`() {
        val (activity, _) = HeuristicClassifier.classifyActivity(
            stats(
                linAccRms = 8.0f,
                gyroMagStd = 1.0f,
                dominantFreqHz = 3.0f,
                dominantPower = 0.8f,
                spectralEntropy = 1.0f,
            )
        )
        assertEquals(ActivityType.RUNNING, activity)
    }

    @Test
    fun `транспорт распознаётся по непериодичной тряске`() {
        val (activity, _) = HeuristicClassifier.classifyActivity(
            stats(
                linAccRms = 0.6f,
                gyroMagStd = 0.03f,
                dominantFreqHz = 3.7f,
                dominantPower = 0.02f,
                spectralEntropy = 2.6f,
            )
        )
        assertEquals(ActivityType.VEHICLE, activity)
    }

    @Test
    fun `распределение эвристики всегда нормировано`() {
        val samples = listOf(
            stats(linAccRms = 0f, gyroMagStd = 0f),
            stats(linAccRms = 2f, dominantFreqHz = 1.8f, dominantPower = 0.5f),
            // Набор, не похожий ни на что: все правила дадут ноль.
            stats(linAccRms = 50f, gyroMagStd = 40f, dominantFreqHz = 12f),
        )
        for (s in samples) {
            val (_, probs) = HeuristicClassifier.classifyActivity(s)
            assertEquals("сумма вероятностей", 1.0f, probs.sum(), 1e-4f)
            assertTrue("нет отрицательных вероятностей", probs.all { it >= 0f })
        }
    }

    @Test
    fun `телефон у уха определяется по перекрытому датчику и вертикали`() {
        val (placement, _) = HeuristicClassifier.classifyPlacement(
            stats(
                linAccRms = 0.3f,
                proximityNearRatio = 1.0f,
                proximityAvailable = true,
                lightLuxMean = 2f,
                lightAvailable = true,
                tiltDeg = 85f,
                orientationStd = 0.02f,
            )
        )
        assertEquals(PhonePlacement.AT_EAR, placement)
    }

    @Test
    fun `телефон на столе определяется по неподвижной ориентации`() {
        val (placement, _) = HeuristicClassifier.classifyPlacement(
            stats(
                linAccRms = 0.02f,
                proximityNearRatio = 0f,
                proximityAvailable = true,
                lightLuxMean = 300f,
                lightAvailable = true,
                tiltDeg = 5f,
                orientationStd = 0.0005f,
            )
        )
        assertEquals(PhonePlacement.ON_TABLE, placement)
    }

    @Test
    fun `сглаживание не переключает класс из-за одиночного выброса`() {
        val smoother = ProbabilitySmoother(classCount = 3)
        val walking = floatArrayOf(0.8f, 0.1f, 0.1f)
        val spike = floatArrayOf(0.1f, 0.8f, 0.1f)

        repeat(5) { smoother.update(walking) }
        assertEquals(0, smoother.update(walking))

        // Одно «чужое» окно не должно менять состояние.
        assertEquals(0, smoother.update(spike))
    }

    @Test
    fun `сглаживание переключает класс при устойчивой смене активности`() {
        val smoother = ProbabilitySmoother(classCount = 3)
        val walking = floatArrayOf(0.8f, 0.1f, 0.1f)
        val running = floatArrayOf(0.05f, 0.9f, 0.05f)

        repeat(5) { smoother.update(walking) }
        var result = 0
        repeat(6) { result = smoother.update(running) }

        assertEquals("после нескольких окон бега класс обязан смениться", 1, result)
        assertNotEquals(0, result)
    }

    @Test
    fun `сброс возвращает сглаживатель в исходное состояние`() {
        val smoother = ProbabilitySmoother(classCount = 3)
        repeat(5) { smoother.update(floatArrayOf(0.9f, 0.05f, 0.05f)) }
        smoother.reset()

        // После сброса первое же распределение принимается без гистерезиса.
        assertEquals(2, smoother.update(floatArrayOf(0.1f, 0.2f, 0.7f)))
    }

    private fun stats(
        accMagMean: Float = 9.81f,
        accMagStd: Float = 0f,
        linAccRms: Float = 0f,
        gyroMagMean: Float = 0f,
        gyroMagStd: Float = 0f,
        magMagMean: Float = 45f,
        magMagStd: Float = 0f,
        lightLuxMean: Float = 200f,
        lightAvailable: Boolean = true,
        proximityNearRatio: Float = 0f,
        proximityAvailable: Boolean = true,
        dominantFreqHz: Float = 0f,
        dominantPower: Float = 0f,
        spectralEntropy: Float = 0f,
        tiltDeg: Float = 0f,
        orientationStd: Float = 0.01f,
        zeroCrossingRate: Float = 0f,
    ) = WindowStats(
        accMagMean = accMagMean,
        accMagStd = accMagStd,
        linAccRms = linAccRms,
        gyroMagMean = gyroMagMean,
        gyroMagStd = gyroMagStd,
        magMagMean = magMagMean,
        magMagStd = magMagStd,
        lightLuxMean = lightLuxMean,
        lightAvailable = lightAvailable,
        proximityNearRatio = proximityNearRatio,
        proximityAvailable = proximityAvailable,
        dominantFreqHz = dominantFreqHz,
        dominantPower = dominantPower,
        spectralEntropy = spectralEntropy,
        tiltDeg = tiltDeg,
        orientationStd = orientationStd,
        zeroCrossingRate = zeroCrossingRate,
    )
}
