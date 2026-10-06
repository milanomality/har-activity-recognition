package com.example.har.service

import android.content.Context
import android.util.Log
import com.example.har.collect.DatasetRecorder
import com.example.har.data.LogRepository
import com.example.har.google.GoogleActivityTracker
import com.example.har.ml.ActivityRecognizer
import com.example.har.ml.RecognitionResult
import com.example.har.ml.RecognizerStatus
import com.example.har.motion.InertialSpeedEstimator
import com.example.har.motion.StepDetector
import com.example.har.sensors.SensorAvailability
import com.example.har.sensors.SensorFrame
import com.example.har.sensors.SensorHub
import com.example.har.sensors.SlidingWindowBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Наблюдаемое состояние конвейера — источник истины для UI и уведомления. */
data class EngineState(
    val running: Boolean = false,
    val startedAtMs: Long = 0L,
    val framesProcessed: Long = 0L,
    val windowsProcessed: Long = 0L,
    val status: RecognizerStatus? = null,
    val availability: SensorAvailability? = null,
    val error: String? = null,
    /** Окон за сессию, где решение Google можно сравнить с нашим. */
    val googleCompared: Long = 0L,
    /** Из них совпало с Google. */
    val googleAgreed: Long = 0L,
)

/**
 * Связывает датчики, классификатор и журнал в один работающий конвейер.
 *
 * Живёт в объекте Application, а не в сервисе: и экран, и сервис должны
 * видеть одно и то же состояние, а пересоздание распознавателя при каждом
 * повороте экрана означало бы перезагрузку моделей и сброс сглаживания.
 * Сервис при этом остаётся нужен — он удерживает процесс живым,
 * когда экран погашен.
 */
class RecognitionEngine(
    private val context: Context,
    private val repository: LogRepository,
    val datasetRecorder: DatasetRecorder,
) {
    private val sensorHub = SensorHub(context)

    private val _state = MutableStateFlow(EngineState(availability = sensorHub.availability))
    val state: StateFlow<EngineState> = _state.asStateFlow()

    private val _latest = MutableStateFlow<RecognitionResult?>(null)
    val latest: StateFlow<RecognitionResult?> = _latest.asStateFlow()

    /** Последний кадр — для «живых» показаний датчиков на экране. */
    private val _liveFrame = MutableStateFlow<SensorFrame?>(null)
    val liveFrame: StateFlow<SensorFrame?> = _liveFrame.asStateFlow()

    /** Последние [HISTORY_SECONDS] секунд кадров — для графиков датчиков. */
    private val _history = MutableStateFlow<List<SensorFrame>>(emptyList())
    val history: StateFlow<List<SensorFrame>> = _history.asStateFlow()

    private var recognizer: ActivityRecognizer? = null
    private var job: Job? = null

    /**
     * Запускает сбор и распознавание в переданной области видимости
     * (обычно — жизненный цикл сервиса).
     *
     * Повторный вызов при уже запущенном конвейере игнорируется.
     */
    fun start(scope: CoroutineScope) {
        if (job?.isActive == true) return

        if (!sensorHub.availability.canRecognize) {
            _state.update {
                it.copy(error = "На устройстве нет акселерометра — распознавание невозможно")
            }
            return
        }

        val r = recognizer ?: ActivityRecognizer(context).also { recognizer = it }
        r.reset()

        _state.update {
            it.copy(
                running = true,
                startedAtMs = System.currentTimeMillis(),
                framesProcessed = 0,
                windowsProcessed = 0,
                googleCompared = 0,
                googleAgreed = 0,
                status = r.status,
                availability = sensorHub.availability,
                error = null,
            )
        }

        // Инференс и запись в БД идут на Dispatchers.Default: БПФ на окне
        // из 128 отсчётов и прогон сети нельзя выполнять на главном потоке.
        GoogleActivityTracker.start(context)

        job = scope.launch(Dispatchers.Default) {
            val windowBuffer = SlidingWindowBuffer()
            val speedEstimator = InertialSpeedEstimator()
            val stepDetector = StepDetector()
            val historyBuffer = ArrayDeque<SensorFrame>(HISTORY_FRAMES)
            _history.value = emptyList()
            var frames = 0L
            var lastLiveFrameMs = 0L
            var lastHistoryMs = 0L
            try {
                sensorHub.frames().collect { raw ->
                    frames++
                    // В датасет — сырые показания: скорость производная и пересчитывается.
                    datasetRecorder.write(raw)

                    // Скорость считается покадрово: интегрирование требует каждого отсчёта,
                    // окна для него слишком редкие.
                    val est = speedEstimator.update(
                        raw.ax, raw.ay, raw.az, raw.gx, raw.gy, raw.gz, raw.mx, raw.my, raw.mz,
                    )
                    // Длина шага зависит от того, где телефон: берём последнее
                    // решение классификатора положения (оно обновляется раз в окно).
                    val k = StepDetector.stepLengthK(_latest.value?.prediction?.placementProbabilities)
                    val step = stepDetector.update(est.verticalAcc, k)
                    val frame = raw.copy(
                        speedMs = est.horizontalSpeed,
                        secondsSinceZupt = est.secondsSinceZupt,
                        rotationSinceZupt = est.rotationSinceZupt,
                        verticalAcc = est.verticalAcc,
                        horizontalAcc = est.horizontalAcc,
                        yawRate = est.yawRate,
                        insTiltDeg = est.tiltDeg,
                        magNorm = est.magNorm,
                        magInclinationDeg = est.magInclinationDeg,
                        headingDeg = est.headingDeg,
                        magTrusted = est.magTrusted,
                        stepCount = step.stepCount,
                        cadenceHz = step.cadenceHz,
                        stepAmplitude = step.stepAmplitude,
                        stepSpeedMs = step.speedMs,
                    )

                    historyBuffer.addLast(frame)
                    if (historyBuffer.size > HISTORY_FRAMES) historyBuffer.removeFirst()

                    // Кадры приходят 50 раз в секунду. Публиковать каждый в
                    // StateFlow нельзя: экран перерисовывался бы 50 раз в секунду
                    // ради чисел, которые глаз всё равно не успевает прочесть.
                    val now = System.currentTimeMillis()
                    if (now - lastLiveFrameMs >= LIVE_FRAME_INTERVAL_MS) {
                        lastLiveFrameMs = now
                        _liveFrame.value = frame
                    }
                    // Графикам нужна плавность, поэтому чаще, чем числам, — но
                    // всё равно не на каждом кадре: копия буфера стоит памяти.
                    if (now - lastHistoryMs >= HISTORY_INTERVAL_MS) {
                        lastHistoryMs = now
                        _history.value = historyBuffer.toList()
                    }

                    val window = windowBuffer.push(frame) ?: return@collect

                    val result = r.recognize(window).copy(google = GoogleActivityTracker.current())
                    _latest.value = result
                    repository.record(result)
                    val agrees = result.google?.type?.agreesWith(result.prediction.activity)
                    // Счётчики обновляем раз в окно, а не раз в кадр — по той же причине.
                    _state.update {
                        it.copy(
                            framesProcessed = frames,
                            windowsProcessed = it.windowsProcessed + 1,
                            googleCompared = it.googleCompared + if (agrees != null) 1 else 0,
                            googleAgreed = it.googleAgreed + if (agrees == true) 1 else 0,
                        )
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Конвейер распознавания остановлен из-за ошибки", e)
                _state.update { it.copy(running = false, error = e.message ?: "Неизвестная ошибка") }
            }
        }
    }

    fun stop() {
        GoogleActivityTracker.stop(context)
        job?.cancel()
        job = null
        datasetRecorder.stop()
        recognizer?.reset()
        _state.update { it.copy(running = false) }
    }

    /** Освобождает модели. Вызывается, когда процесс завершает работу. */
    fun shutdown() {
        stop()
        recognizer?.close()
        recognizer = null
    }

    companion object {
        private const val TAG = "RecognitionEngine"

        /** Как часто показания датчиков обновляются на экране (5 раз в секунду). */
        private const val LIVE_FRAME_INTERVAL_MS = 200L

        /** Сколько секунд сигнала показывают графики. */
        const val HISTORY_SECONDS = 10

        private const val HISTORY_FRAMES = HISTORY_SECONDS * SensorHub.SAMPLE_RATE_HZ

        /** Как часто обновляются графики (10 раз в секунду). */
        private const val HISTORY_INTERVAL_MS = 100L
    }
}
