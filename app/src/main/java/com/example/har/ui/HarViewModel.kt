package com.example.har.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.har.HarApplication
import com.example.har.data.CsvExporter
import com.example.har.data.LogRepository
import com.example.har.data.db.ActivityIntervalEntity
import com.example.har.data.db.ActivitySummaryRow
import com.example.har.ml.ActivityType
import com.example.har.ml.PhonePlacement
import com.example.har.service.RecognitionService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** Состояние режима сбора размеченного датасета. */
data class CollectionState(
    val recording: Boolean = false,
    val activity: ActivityType = ActivityType.WALKING,
    val placement: PhonePlacement = PhonePlacement.POCKET,
    val samples: Int = 0,
    val startedAtMs: Long = 0L,
    val files: List<File> = emptyList(),
    val totalBytes: Long = 0L,
)

/** Одноразовое сообщение для снекбара. */
data class UiMessage(val text: String, val shareFile: File? = null, val shareTitle: String = "")

class HarViewModel(app: Application) : AndroidViewModel(app) {

    private val harApp = app as HarApplication
    private val repository = harApp.repository
    private val engine = harApp.engine
    private val recorder = harApp.datasetRecorder

    val engineState = engine.state
    val latest = engine.latest
    val liveFrame = engine.liveFrame

    /** День, который показывает журнал (начало суток в мс). */
    private val _selectedDay = MutableStateFlow(LogRepository.startOfToday())
    val selectedDay: StateFlow<Long> = _selectedDay.asStateFlow()

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    val intervals: StateFlow<List<ActivityIntervalEntity>> = _selectedDay
        .flatMapLatest { repository.intervalsForDay(it) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    val summary: StateFlow<List<ActivitySummaryRow>> = _selectedDay
        .flatMapLatest { repository.summaryForDay(it) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val windowCount: StateFlow<Int> = repository.windowCount()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), 0)

    private val _collection = MutableStateFlow(CollectionState())
    val collection: StateFlow<CollectionState> = _collection.asStateFlow()

    private val _message = MutableStateFlow<UiMessage?>(null)
    val message: StateFlow<UiMessage?> = _message.asStateFlow()

    init {
        refreshDatasetFiles()
    }

    // --- Управление распознаванием ---

    fun startRecognition() = RecognitionService.start(getApplication())

    fun stopRecognition() {
        // Запись датасета без работающих датчиков бессмысленна — останавливаем вместе.
        if (_collection.value.recording) stopCollection()
        RecognitionService.stop(getApplication())
    }

    fun toggleRecognition() {
        if (engineState.value.running) stopRecognition() else startRecognition()
    }

    // --- Журнал ---

    fun shiftDay(days: Int) {
        _selectedDay.value += days * LogRepository.DAY_MS
    }

    fun goToToday() {
        _selectedDay.value = LogRepository.startOfToday()
    }

    fun clearJournal() = viewModelScope.launch {
        repository.clearAll()
        _message.value = UiMessage("Журнал очищен")
    }

    fun exportIntervals() = export("Журнал интервалов") {
        CsvExporter.exportIntervals(getApplication(), repository)
    }

    fun exportWindows() = export("Покадровый журнал") {
        CsvExporter.exportWindows(getApplication(), repository)
    }

    private fun export(title: String, block: suspend () -> File) = viewModelScope.launch {
        try {
            val file = withContext(Dispatchers.IO) { block() }
            val kb = file.length() / 1024
            _message.value = UiMessage(
                text = "$title: ${file.name} ($kb КБ)",
                shareFile = file,
                shareTitle = title,
            )
        } catch (e: Exception) {
            _message.value = UiMessage("Не удалось выгрузить: ${e.message}")
        }
    }

    // --- Сбор датасета ---

    fun setCollectionActivity(activity: ActivityType) {
        if (_collection.value.recording) return
        _collection.value = _collection.value.copy(activity = activity)
    }

    fun setCollectionPlacement(placement: PhonePlacement) {
        if (_collection.value.recording) return
        _collection.value = _collection.value.copy(placement = placement)
    }

    /**
     * Начинает запись размеченных данных. Требует работающего конвейера:
     * кадры в файл пишет [com.example.har.service.RecognitionEngine],
     * поэтому при остановленных датчиках файл остался бы пустым.
     */
    fun startCollection() {
        if (!engineState.value.running) {
            startRecognition()
        }
        val state = _collection.value
        val session = recorder.start(state.activity, state.placement)
        _collection.value = state.copy(
            recording = true,
            samples = 0,
            startedAtMs = session.startedAtMs,
        )
        _message.value = UiMessage("Запись начата: ${session.file.name}")
    }

    fun stopCollection() {
        val session = recorder.stop()
        _collection.value = _collection.value.copy(recording = false, samples = session?.samples ?: 0)
        refreshDatasetFiles()
        _message.value = if (session == null) {
            UiMessage("Запись остановлена, данных не получено")
        } else {
            val seconds = session.samples / 50
            UiMessage("Записано ${session.samples} отсчётов (~$seconds с) в ${session.file.name}")
        }
    }

    /** Обновляет счётчик отсчётов в UI — вызывается из экрана по таймеру. */
    fun refreshCollectionProgress() {
        val session = recorder.currentSession ?: return
        _collection.value = _collection.value.copy(samples = session.samples)
    }

    fun refreshDatasetFiles() {
        _collection.value = _collection.value.copy(
            files = recorder.listFiles(),
            totalBytes = recorder.totalSizeBytes(),
        )
    }

    fun shareDataset(file: File) {
        _message.value = UiMessage("Файл датасета: ${file.name}", file, "Файл датасета")
    }

    fun deleteDatasets() = viewModelScope.launch {
        val n = recorder.deleteAll()
        refreshDatasetFiles()
        _collection.value = _collection.value.copy(recording = false)
        _message.value = UiMessage("Удалено файлов: $n")
    }

    fun consumeMessage() {
        _message.value = null
    }
}
