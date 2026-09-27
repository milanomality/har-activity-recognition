package com.example.har.service

import android.content.Context
import android.util.Log
import com.example.har.collect.DatasetRecorder
import com.example.har.data.LogRepository
import com.example.har.ml.ActivityRecognizer
import com.example.har.ml.RecognitionResult
import com.example.har.ml.RecognizerStatus
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
                status = r.status,
                availability = sensorHub.availability,
                error = null,
            )
        }

        // Инференс и запись в БД идут на Dispatchers.Default: БПФ на окне
        // из 128 отсчётов и прогон сети нельзя выполнять на главном потоке.
        job = scope.launch(Dispatchers.Default) {
            val windowBuffer = SlidingWindowBuffer()
            var frames = 0L
            var lastLiveFrameMs = 0L
            try {
                sensorHub.frames().collect { frame ->
                    frames++
                    datasetRecorder.write(frame)

                    // Кадры приходят 50 раз в секунду. Публиковать каждый в
                    // StateFlow нельзя: экран перерисовывался бы 50 раз в секунду
                    // ради чисел, которые глаз всё равно не успевает прочесть.
                    val now = System.currentTimeMillis()
                    if (now - lastLiveFrameMs >= LIVE_FRAME_INTERVAL_MS) {
                        lastLiveFrameMs = now
                        _liveFrame.value = frame
                    }

                    val window = windowBuffer.push(frame) ?: return@collect

                    val result = r.recognize(window)
                    _latest.value = result
                    repository.record(result)
                    // Счётчики обновляем раз в окно, а не раз в кадр — по той же причине.
                    _state.update {
                        it.copy(
                            framesProcessed = frames,
                            windowsProcessed = it.windowsProcessed + 1,
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
    }
}
