package com.example.har

import android.app.Application
import com.example.har.collect.DatasetRecorder
import com.example.har.data.LogRepository
import com.example.har.service.RecognitionEngine

/**
 * Простейший контейнер зависимостей на уровне процесса.
 *
 * Полноценный DI-фреймворк здесь был бы избыточен: зависимостей три,
 * все — синглтоны на процесс, и важно как раз то, что и экран, и фоновый
 * сервис получают одни и те же экземпляры.
 */
class HarApplication : Application() {

    val repository: LogRepository by lazy { LogRepository(this) }
    val datasetRecorder: DatasetRecorder by lazy { DatasetRecorder(this) }
    val engine: RecognitionEngine by lazy {
        RecognitionEngine(this, repository, datasetRecorder)
    }

    override fun onTerminate() {
        // На реальных устройствах вызывается редко, но в эмуляторе
        // и в тестах позволяет корректно освободить интерпретаторы TFLite.
        engine.shutdown()
        super.onTerminate()
    }
}
