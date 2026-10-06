package com.example.har

import com.example.har.features.FeatureExtractor
import com.example.har.features.WindowStats
import com.example.har.ml.ActivityType
import com.example.har.ml.MotionFusion
import com.example.har.ml.PhonePlacement
import com.example.har.motion.InertialSpeedEstimator
import com.example.har.motion.StepDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Промежуточные величины движения: шаги, магнитное поле, учёт положения телефона.
 * Проверяются на синтетике, где правильный ответ известен заранее.
 */
class MotionTest {

    private val rate = 50
    private val g = InertialSpeedEstimator.STANDARD_G.toFloat()

    // ------------------------------------------------------------ шаги

    @Test
    fun `шаги считаются по вертикальному ускорению и дают темп`() {
        val det = StepDetector(rate)
        val hz = 1.8
        var last: StepDetector.State? = null
        repeat(rate * 10) { i ->
            last = det.update((3.0 * sin(2 * PI * hz * i / rate)).toFloat())
        }
        val s = requireNotNull(last)
        // За 10 с при 1.8 шага/с — 18 шагов; первый может не набрать порог.
        assertTrue("шагов ${s.stepCount}", s.stepCount in 16..19)
        assertEquals(hz.toFloat(), s.cadenceHz, 0.1f)
        // Размах 6 м/с² → длина по Вайнбергу 0.45 · 6^¼ ≈ 0.7 м, скорость ≈ 1.3 м/с.
        assertEquals(1.26f, s.speedMs, 0.25f)
    }

    @Test
    fun `шум лежащего телефона шагами не считается`() {
        val det = StepDetector(rate)
        val rnd = Random(1)
        var last: StepDetector.State? = null
        repeat(rate * 10) { last = det.update((rnd.nextFloat() - 0.5f) * 0.3f) }
        assertEquals(0, requireNotNull(last).stepCount)
        assertEquals(0f, last!!.cadenceHz, 0f)
    }

    @Test
    fun `длина шага зависит от положения телефона`() {
        val pocket = FloatArray(PhonePlacement.COUNT).also { it[PhonePlacement.POCKET.id] = 1f }
        val hand = FloatArray(PhonePlacement.COUNT).also { it[PhonePlacement.IN_HAND.id] = 1f }
        // В руке тот же шаг даёт меньший размах, поэтому коэффициент больше.
        assertTrue(StepDetector.stepLengthK(hand) > StepDetector.stepLengthK(pocket))
        assertEquals(StepDetector.DEFAULT_K, StepDetector.stepLengthK(null), 0f)
    }

    @Test
    fun `регулярность высока у ходьбы и низка у случайной тряски`() {
        val n = 128
        val walk = FloatArray(n) { (2.5 * sin(2 * PI * 1.8 * it / rate)).toFloat() }
        val rnd = Random(7)
        val shake = FloatArray(n) { (rnd.nextFloat() - 0.5f) * 3f }
        assertTrue(FeatureExtractor.stepRegularity(walk) > 0.7f)
        assertTrue(FeatureExtractor.stepRegularity(shake) < 0.3f)
    }

    // ------------------------------------------------------------ навигация и магнитометр

    @Test
    fun `вертикальное ускорение не зависит от наклона телефона`() {
        // Телефон наклонён на 60°, тело подпрыгивает на 2 м/с² по вертикали Земли.
        val ins = InertialSpeedEstimator(rate)
        val a = Math.toRadians(60.0)
        repeat(rate) { ins.update(0f, (g * sin(a)).toFloat(), (g * cos(a)).toFloat(), 0f, 0f, 0f) }
        val up = 2f
        val e = ins.update(0f, ((g + up) * sin(a)).toFloat(), ((g + up) * cos(a)).toFloat(), 0f, 0f, 0f)
        assertEquals(up, e.verticalAcc, 0.1f)
        assertEquals(0f, e.horizontalAcc, 0.1f)
        assertEquals(60f, e.tiltDeg, 1f)
    }

    @Test
    fun `курс по компасу`() {
        val ins = InertialSpeedEstimator(rate)
        // Телефон лежит экраном вверх; север — вдоль оси Y телефона, поле уходит вниз.
        var e = ins.update(0f, 0f, g, 0f, 0f, 0f, 0f, 20f, -45f)
        assertEquals(0f, e.headingDeg, 1f)
        assertEquals(66f, e.magInclinationDeg, 1f)
        // Север справа от телефона (вдоль +X) — телефон смотрит на запад, курс 270°.
        val ins2 = InertialSpeedEstimator(rate)
        e = ins2.update(0f, 0f, g, 0f, 0f, 0f, 20f, 0f, -45f)
        assertEquals(270f, e.headingDeg, 1f)
    }

    @Test
    fun `магнитометр гасит дрейф рыскания от смещения гироскопа`() {
        val bias = 0.02f // рад/с по оси Z — типичное смещение недорогого гироскопа
        fun drift(withMag: Boolean): Double {
            val ins = InertialSpeedEstimator(rate)
            // Телефон в руке слегка движется, чтобы ZUPT не мешал: ускорение чуть не равно g.
            val start = ins.update(0f, 0f, g, 0f, 0f, 0f).let { ins.yawDeg() }
            repeat(rate * 60) {
                if (withMag) ins.update(0f, 0f, g, 0f, 0f, bias, 0f, 20f, -45f)
                else ins.update(0f, 0f, g, 0f, 0f, bias)
            }
            var d = ins.yawDeg() - start
            if (d > 180) d -= 360
            if (d < -180) d += 360
            return abs(d)
        }
        // Без магнитометра за минуту уплывает на ≈69°, с ним — остаётся у опоры.
        assertTrue("без поля ${drift(false)}", drift(false) > 50.0)
        assertTrue("с полем ${drift(true)}", drift(true) < 5.0)
    }

    @Test
    fun `искажённое поле в поправку не идёт`() {
        val ins = InertialSpeedEstimator(rate)
        repeat(rate) { ins.update(0f, 0f, g, 0f, 0f, 0f, 0f, 20f, -45f) }
        // Рядом металл: модуль поля вырос вдвое.
        val e = ins.update(0f, 0f, g, 0f, 0f, 0f, 0f, 40f, -90f)
        assertTrue(!e.magTrusted)
    }

    // ------------------------------------------------------------ учёт положения телефона

    private fun placement(p: PhonePlacement) = FloatArray(PhonePlacement.COUNT).also { it[p.id] = 1f }

    @Test
    fun `жест рукой без шагов ослабляет ходьбу и бег, но не покой`() {
        // Телефон в руке, им двигают: большая амплитуда и вращение, шагов нет.
        val s = stats(accStd = 3f, gyro = 1.5f, steps = 0, cadence = 0f, regularity = 0.1f)
        val l = MotionFusion.likelihoods(s, placement(PhonePlacement.IN_HAND))
        assertEquals(1f, l[ActivityType.STILL.id], 1e-4f)
        assertEquals(MotionFusion.FLOOR, l[ActivityType.WALKING.id], 1e-4f)
        assertEquals(MotionFusion.FLOOR, l[ActivityType.RUNNING.id], 1e-4f)

        // Та же модель выбрала бы бег — после поправки побеждает покой.
        val model = FloatArray(ActivityType.COUNT).also {
            it[ActivityType.RUNNING.id] = 0.5f; it[ActivityType.STILL.id] = 0.3f; it[ActivityType.WALKING.id] = 0.2f
        }
        val fused = MotionFusion.apply(model, s, placement(PhonePlacement.IN_HAND))
        assertEquals(ActivityType.STILL.id, fused.indices.maxByOrNull { fused[it] })
    }

    @Test
    fun `мах руки при ходьбе не превращает ходьбу в бег`() {
        // Амплитуду раздула рука, но темп — ходьба (1.8 шага/с) и шаги регулярны.
        val s = stats(accStd = 6f, gyro = 2f, steps = 4, cadence = 1.8f, regularity = 0.8f)
        val model = FloatArray(ActivityType.COUNT).also {
            it[ActivityType.RUNNING.id] = 0.55f; it[ActivityType.WALKING.id] = 0.45f
        }
        val fused = MotionFusion.apply(model, s, placement(PhonePlacement.IN_HAND))
        assertTrue(fused[ActivityType.WALKING.id] > fused[ActivityType.RUNNING.id])
    }

    @Test
    fun `в кармане темп весит меньше, чем в руке`() {
        val s = stats(accStd = 6f, gyro = 2f, steps = 4, cadence = 1.8f, regularity = 0.8f)
        val pocket = MotionFusion.likelihoods(s, placement(PhonePlacement.POCKET))
        val hand = MotionFusion.likelihoods(s, placement(PhonePlacement.IN_HAND))
        assertTrue(pocket[ActivityType.RUNNING.id] > hand[ActivityType.RUNNING.id])
    }

    @Test
    fun `шаги исключают покой`() {
        val s = stats(accStd = 2.5f, gyro = 1f, steps = 4, cadence = 1.8f, regularity = 0.8f)
        val l = MotionFusion.likelihoods(s, placement(PhonePlacement.POCKET))
        assertEquals(MotionFusion.FLOOR, l[ActivityType.STILL.id], 1e-4f)
        assertEquals(1f, l[ActivityType.WALKING.id], 1e-4f)
    }

    @Test
    fun `неподвижный телефон в руке — не велосипед, даже если модель уверена`() {
        // Случай с устройства: телефон держат почти вертикально и не двигаются,
        // классификатор положения ошибся («на столе»), а сеть отдала велосипеду 0.9999.
        val s = stats(accStd = 0.1f, gyro = 0.06f, steps = 0, cadence = 0f, regularity = 0f, tiltSwing = 2.4f)
        val model = FloatArray(ActivityType.COUNT).also {
            it[ActivityType.CYCLING.id] = 0.9999f; it[ActivityType.STILL.id] = 0.0001f
        }
        val fused = MotionFusion.apply(model, s, placement(PhonePlacement.ON_TABLE))
        assertEquals(ActivityType.STILL.id, fused.indices.maxByOrNull { fused[it] })
        assertTrue(fused[ActivityType.STILL.id] > 0.9f)
    }

    @Test
    fun `качающийся телефон «на столе» считается телефоном в руке`() {
        val s = stats(accStd = 0.3f, gyro = 0.25f, steps = 0, cadence = 0f, regularity = 0f, tiltSwing = 10f)
        val f = MotionFusion.factors(s, placement(PhonePlacement.ON_TABLE))
        assertEquals(1f, f.handHeld, 1e-4f)
    }

    @Test
    fun `дрожь руки переводит «на столе» в «в руке», а неподвижный стол не трогает`() {
        val model = placement(PhonePlacement.ON_TABLE)
        // Данные с устройства: в руке вращение 0.03–0.06 рад/с, на столе 0.001.
        val held = stats(accStd = 0.07f, gyro = 0.05f, steps = 0, cadence = 0f, regularity = 0f, tiltSwing = 1.2f)
        val onTable = stats(accStd = 0.01f, gyro = 0.001f, steps = 0, cadence = 0f, regularity = 0f, tiltSwing = 0.02f)
        val p1 = com.example.har.ml.PlacementFusion.apply(model, held)
        assertEquals(1f, p1[PhonePlacement.IN_HAND.id], 1e-4f)
        val p2 = com.example.har.ml.PlacementFusion.apply(model, onTable)
        assertEquals(1f, p2[PhonePlacement.ON_TABLE.id], 1e-4f)
    }

    @Test
    fun `мах рукой — не велосипед`() {
        // Окно с устройства: вращение 1.4 рад/с, размах наклона 60°, «шаги» с темпом 0.8 Гц.
        val s = stats(accStd = 2.7f, gyro = 1.4f, steps = 2, cadence = 0.79f, regularity = 0.26f, tiltSwing = 60f)
        val model = FloatArray(ActivityType.COUNT).also {
            it[ActivityType.CYCLING.id] = 0.99f; it[ActivityType.STILL.id] = 0.01f
        }
        val f = MotionFusion.factors(s, placement(PhonePlacement.IN_HAND))
        assertEquals("редкие толчки — не шаговый ритм", 0f, f.gait, 1e-4f)
        val fused = MotionFusion.apply(model, s, placement(PhonePlacement.IN_HAND))
        assertEquals(ActivityType.STILL.id, fused.indices.maxByOrNull { fused[it] })
        assertTrue("велосипед ${fused[ActivityType.CYCLING.id]}", fused[ActivityType.CYCLING.id] < 0.15f)
    }

    @Test
    fun `в кармане велосипед не ослабляется`() {
        val s = stats(accStd = 1.5f, gyro = 1.2f, steps = 0, cadence = 0f, regularity = 0.1f, tiltSwing = 40f)
        val l = MotionFusion.likelihoods(s, placement(PhonePlacement.POCKET))
        assertEquals(1f, l[ActivityType.CYCLING.id], 1e-4f)
    }

    @Test
    fun `в темноте у уха — не карман, отличает поза`() {
        // Окно с устройства: тёмная комната (0 лк), датчик закрыт, телефон
        // почти неподвижен у уха, а нейросеть положения отдала карману 0.99.
        val model = placement(PhonePlacement.POCKET)
        val dark = stats(accStd = 0.03f, gyro = 0.05f, steps = 0, cadence = 0f, regularity = 0f, lux = 0f, near = 1f)
        // Экран вертикален, верх выше низа, телефон по диагонали от уха ко рту.
        val ear = dark.copy(tiltDeg = 88f, gravityXRatio = -0.6f, gravityYRatio = 0.75f, tiltSwingDeg = 3f)
        val p = com.example.har.ml.PlacementFusion.apply(model, ear)
        assertTrue("у уха ${p[PhonePlacement.AT_EAR.id]}", p[PhonePlacement.AT_EAR.id] > 0.9f)

        // Отвесный телефон в кармане стоящего человека: оси X нет.
        val standing = dark.copy(tiltDeg = 90f, gravityXRatio = 0.05f, gravityYRatio = 0.99f)
        assertEquals(1f, com.example.har.ml.PlacementFusion.apply(model, standing)[PhonePlacement.POCKET.id], 1e-4f)
        // Плашмя вдоль бедра сидящего.
        val sitting = dark.copy(tiltDeg = 170f, gravityXRatio = 0.4f, gravityYRatio = 0.1f)
        assertEquals(1f, com.example.har.ml.PlacementFusion.apply(model, sitting)[PhonePlacement.POCKET.id], 1e-4f)
        // Раскачивается вместе с ногой при ходьбе.
        val walking = ear.copy(tiltSwingDeg = 45f)
        assertEquals(1f, com.example.har.ml.PlacementFusion.apply(model, walking)[PhonePlacement.POCKET.id], 1e-4f)
    }

    @Test
    fun `перевёрнутый верхом вниз телефон — не у уха`() {
        val model = placement(PhonePlacement.AT_EAR)
        val s = stats(accStd = 0.03f, gyro = 0.05f, steps = 0, cadence = 0f, regularity = 0f, lux = 0f, near = 1f)
            .copy(tiltDeg = 90f, gravityYRatio = -0.95f)
        val p = com.example.har.ml.PlacementFusion.apply(model, s)
        assertEquals(1f, p[PhonePlacement.POCKET.id], 1e-4f)
    }

    @Test
    fun `свет и открытый датчик — не карман`() {
        // Окно с устройства: энергичный мах, 480 лк, датчик приближения открыт,
        // а нейросеть положения отдала карману 0.98.
        val model = placement(PhonePlacement.POCKET)
        val s = stats(accStd = 3.7f, gyro = 1.8f, steps = 4, cadence = 1.5f, regularity = 0.9f, lux = 480f)
        val p = com.example.har.ml.PlacementFusion.apply(model, s)
        assertEquals(1f, p[PhonePlacement.IN_HAND.id], 1e-4f)
        // В темноте с закрытым датчиком карман остаётся карманом.
        val dark = stats(accStd = 3.7f, gyro = 1.8f, steps = 4, cadence = 1.5f, regularity = 0.9f, lux = 2f, near = 1f)
        assertEquals(1f, com.example.har.ml.PlacementFusion.apply(model, dark)[PhonePlacement.POCKET.id], 1e-4f)
    }

    @Test
    fun `ритмичный мах рукой — не шаги`() {
        // Окно с устройства: 4 «шага» с темпом 1.5 Гц и регулярностью 0.9, но вращение
        // 2.3 рад/с, размах наклона 47°, горизонталь 8.6 против вертикали 3.6 м/с².
        val s = stats(
            accStd = 4.4f, gyro = 2.3f, steps = 4, cadence = 1.52f, regularity = 0.95f,
            tiltSwing = 47f, vertical = 3.6f, horizontal = 8.6f,
        )
        val f = MotionFusion.factors(s, placement(PhonePlacement.IN_HAND))
        assertTrue("шаговый ритм ${f.gait}", f.gait < 0.05f)
        val model = FloatArray(ActivityType.COUNT).also {
            it[ActivityType.STAIRS_UP.id] = 0.95f; it[ActivityType.STILL.id] = 0.05f
        }
        val fused = MotionFusion.apply(model, s, placement(PhonePlacement.IN_HAND))
        assertEquals(ActivityType.STILL.id, fused.indices.maxByOrNull { fused[it] })
    }

    @Test
    fun `ходьба с телефоном в руке остаётся ходьбой`() {
        // Телефон держат перед собой: вращение небольшое, преобладает вертикаль.
        val s = stats(
            accStd = 2.0f, gyro = 0.5f, steps = 4, cadence = 1.8f, regularity = 0.8f,
            tiltSwing = 10f, vertical = 2.2f, horizontal = 1.2f,
        )
        val f = MotionFusion.factors(s, placement(PhonePlacement.IN_HAND))
        assertEquals(1f, f.gait, 1e-4f)
        assertEquals(0f, f.stillOverride, 1e-4f)
    }

    @Test
    fun `при ходьбе физический запрет не срабатывает`() {
        val s = stats(accStd = 2.5f, gyro = 1f, steps = 4, cadence = 1.8f, regularity = 0.8f)
        assertEquals(0f, MotionFusion.factors(s, placement(PhonePlacement.POCKET)).stillOverride, 1e-4f)
    }

    @Test
    fun `без навигации поправка не применяется`() {
        val s = stats(accStd = 3f, gyro = 1.5f, steps = 0, cadence = 0f, regularity = 0f, computed = false)
        val model = FloatArray(ActivityType.COUNT) { 1f / ActivityType.COUNT }
        val fused = MotionFusion.apply(model, s, placement(PhonePlacement.IN_HAND))
        for (i in model.indices) assertEquals(model[i], fused[i], 1e-6f)
    }

    private fun stats(
        accStd: Float,
        gyro: Float,
        steps: Int,
        cadence: Float,
        regularity: Float,
        magStd: Float = 0.5f,
        magDisturbed: Float = 0f,
        mismatch: Float = 0f,
        computed: Boolean = true,
        tiltSwing: Float = 0f,
        lux: Float = 200f,
        near: Float = 0f,
        vertical: Float = 0f,
        horizontal: Float = 0f,
    ) = WindowStats(
        accMagMean = g, accMagStd = accStd, linAccRms = accStd,
        gyroMagMean = gyro, gyroMagStd = gyro / 2,
        magMagMean = 45f, magMagStd = magStd,
        lightLuxMean = lux, lightAvailable = true,
        proximityNearRatio = near, proximityAvailable = true,
        dominantFreqHz = cadence, dominantPower = 0.5f, spectralEntropy = 1.5f,
        tiltDeg = 60f, orientationStd = 0.05f, zeroCrossingRate = 0.1f,
        speedMs = 0f, secondsSinceZupt = if (computed) 30f else -1f,
        stepsInWindow = steps, cadenceHz = cadence, stepRegularity = regularity,
        magDisturbedRatio = magDisturbed, magGyroMismatchDeg = mismatch,
        tiltSwingDeg = tiltSwing,
        verticalAccRms = vertical, horizontalAccRms = horizontal,
        verticalShare = if (vertical + horizontal > 0f) {
            vertical * vertical / (vertical * vertical + horizontal * horizontal)
        } else {
            0f
        },
    )
}
