package com.example.har

import com.example.har.features.Fft
import com.example.har.features.FeatureExtractor
import com.example.har.sensors.SensorFrame
import com.example.har.sensors.SensorHub
import com.example.har.sensors.SensorWindow
import com.example.har.sensors.SlidingWindowBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

/**
 * Тесты обработки сигнала — той части, где ошибка не видна глазами.
 *
 * Частотные признаки — основа распознавания походки, и неверная привязка
 * бина БПФ к герцам сместила бы все пороги сразу. Проверяем на синтетике,
 * где правильный ответ известен заранее.
 */
class FeatureExtractorTest {

    private val rate = SensorHub.SAMPLE_RATE_HZ
    private val n = SensorWindow.WINDOW_SIZE

    @Test
    fun `БПФ находит частоту чистой синусоиды`() {
        val hz = 2.0f
        val signal = FloatArray(n) { sin(2.0 * PI * hz * it / rate).toFloat() }

        val spectrum = Fft.magnitudeSpectrum(signal)
        val peak = spectrum.indices.maxByOrNull { spectrum[it] } ?: -1
        val peakHz = Fft.binToHz(peak, n, rate)

        // Разрешение по частоте при 128 отсчётах и 50 Гц — 0.39 Гц,
        // поэтому допуск в половину бина.
        assertEquals(hz, peakHz, 0.25f)
    }

    @Test
    fun `частота главного пика соответствует темпу шагов`() {
        val stepHz = 1.8f
        val window = syntheticWindow { i ->
            // Модель ходьбы: гравитация по Z плюс вертикальные толчки шагов.
            9.81f + 2.5f * sin(2.0 * PI * stepHz * i / rate).toFloat()
        }

        val stats = FeatureExtractor.stats(window)

        assertEquals(stepHz, stats.dominantFreqHz, 0.25f)
        assertTrue("периодический сигнал должен давать выраженный пик",
            stats.dominantPower > 0.1f)
        assertTrue("СКЗ линейного ускорения должно быть заметным",
            stats.linAccRms > 1.0f)
    }

    @Test
    fun `покой даёт околонулевое линейное ускорение`() {
        val window = syntheticWindow { 9.81f }
        val stats = FeatureExtractor.stats(window)

        assertEquals(0f, stats.linAccRms, 0.01f)
        assertEquals(0f, stats.accMagStd, 0.01f)
        // Телефон неподвижен — гравитация не гуляет.
        assertEquals(0f, stats.orientationStd, 1e-4f)
    }

    @Test
    fun `шум даёт более высокую спектральную энтропию чем ритмичный сигнал`() {
        val rhythmic = syntheticWindow { i ->
            9.81f + 2.5f * sin(2.0 * PI * 1.8 * i / rate).toFloat()
        }
        val random = java.util.Random(42)
        val noisy = syntheticWindow { 9.81f + random.nextGaussian().toFloat() * 2.5f }

        val rhythmicEntropy = FeatureExtractor.stats(rhythmic).spectralEntropy
        val noisyEntropy = FeatureExtractor.stats(noisy).spectralEntropy

        assertTrue(
            "энтропия шума ($noisyEntropy) должна превышать энтропию ритма ($rhythmicEntropy)",
            noisyEntropy > rhythmicEntropy,
        )
    }

    @Test
    fun `вектор признаков положения имеет объявленную длину`() {
        val window = syntheticWindow { 9.81f }
        val features = FeatureExtractor.placementFeatures(window)

        assertEquals(FeatureExtractor.PLACEMENT_FEATURE_COUNT, features.size)
        assertEquals(FeatureExtractor.PLACEMENT_FEATURE_NAMES.size, features.size)
        assertTrue("признаки не должны содержать NaN", features.none { it.isNaN() })
    }

    @Test
    fun `вход модели раскладывается по каналам в правильном порядке`() {
        val window = syntheticWindow { 9.81f }
        val channels = listOf("acc_z", "gyro_x")
        val mean = floatArrayOf(0f, 0f)
        val std = floatArrayOf(1f, 1f)

        val input = FeatureExtractor.activityInput(window, channels, mean, std)

        assertEquals(window.size * channels.size, input.size)
        // Раскладка «канал последним»: [t0c0, t0c1, t1c0, t1c1, …].
        // acc_z задан значением 9.81, gyro_x — нулями.
        assertEquals(9.81f, input[0], 1e-4f)
        assertEquals(0f, input[1], 1e-4f)
        assertEquals(9.81f, input[2], 1e-4f)
    }

    @Test
    fun `неизвестный канал не ломает инференс`() {
        val window = syntheticWindow { 1f }
        val input = FeatureExtractor.activityInput(
            window, listOf("acc_x", "нет_такого"), floatArrayOf(0f, 0f), floatArrayOf(1f, 1f),
        )
        assertEquals(window.size * 2, input.size)
        // Отсутствующий канал заполняется нулями, а не мусором.
        assertEquals(0f, input[1], 1e-6f)
    }

    @Test
    fun `кольцевой буфер выдаёт окно каждые STRIDE кадров`() {
        val buffer = SlidingWindowBuffer()
        var emitted = 0

        // Первое окно появляется только когда буфер заполнен целиком.
        repeat(SensorWindow.WINDOW_SIZE - 1) { i ->
            assertNull("окно не должно появляться раньше времени", buffer.push(frame(i)))
        }
        assertNotNull("окно должно появиться на WINDOW_SIZE-м кадре",
            buffer.push(frame(SensorWindow.WINDOW_SIZE)))

        // Дальше — ровно одно окно на каждые STRIDE кадров.
        repeat(SensorWindow.WINDOW_STRIDE * 3) { i ->
            if (buffer.push(frame(i)) != null) emitted++
        }
        assertEquals(3, emitted)
    }

    @Test
    fun `окно хранит кадры в порядке поступления`() {
        val buffer = SlidingWindowBuffer()
        var window: SensorWindow? = null
        // Заполняем буфер дважды, чтобы проверить разворот кольца.
        repeat(SensorWindow.WINDOW_SIZE + SensorWindow.WINDOW_STRIDE) { i ->
            buffer.push(frame(i, ax = i.toFloat()))?.let { window = it }
        }

        val w = requireNotNull(window)
        // Последнее окно заканчивается последним поданным кадром.
        val last = SensorWindow.WINDOW_SIZE + SensorWindow.WINDOW_STRIDE - 1
        assertEquals(last.toFloat(), w.accX[w.size - 1], 1e-4f)
        assertEquals((last - w.size + 1).toFloat(), w.accX[0], 1e-4f)
    }

    // --- вспомогательное ---

    private fun frame(index: Int, ax: Float = 0f) = SensorFrame(
        timestampMs = 1_000L + index * 20L,
        elapsedNs = index * 20_000_000L,
        ax = ax, ay = 0f, az = 0f,
        gx = 0f, gy = 0f, gz = 0f,
        mx = 0f, my = 0f, mz = 0f,
        proximityCm = 5f, proximityNear = false, lightLux = 300f,
        gravX = 0f, gravY = 0f, gravZ = 9.81f,
    )

    /** Окно, у которого ось Z акселерометра задаётся функцией от номера отсчёта. */
    private fun syntheticWindow(az: (Int) -> Float): SensorWindow {
        val zeros = { FloatArray(n) }
        return SensorWindow(
            startTimeMs = 0,
            endTimeMs = n * 1000L / rate,
            size = n,
            accX = zeros(), accY = zeros(), accZ = FloatArray(n) { az(it) },
            gyroX = zeros(), gyroY = zeros(), gyroZ = zeros(),
            magX = zeros(), magY = zeros(), magZ = zeros(),
            proximity = FloatArray(n) { 5f },
            proximityNear = BooleanArray(n),
            light = FloatArray(n) { 300f },
            gravX = zeros(), gravY = zeros(), gravZ = FloatArray(n) { 9.81f },
        )
    }
}
