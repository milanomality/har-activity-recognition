package com.example.har.collect

import android.content.Context
import android.util.Log
import com.example.har.ml.ActivityType
import com.example.har.ml.PhonePlacement
import com.example.har.sensors.SensorFrame
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Запись размеченного датасета прямо с телефона.
 *
 * Публичные датасеты (UCI HAR, WISDM) не содержат ни магнитометра,
 * ни датчиков приближения и освещённости, и уж точно не содержат разметки
 * «где лежит телефон». Поэтому модель положения телефона обучать не на чем,
 * пока данные не собраны этим режимом: пользователь заранее сообщает,
 * что он сейчас будет делать и где находится телефон, и приложение пишет
 * сырые кадры 50 Гц в CSV с этой меткой.
 *
 * Пишем сырые кадры, а не признаки: набор признаков ещё будет меняться,
 * а переснять двадцать минут ходьбы по лестнице заново — дорого.
 */
class DatasetRecorder(private val context: Context) {

    data class Session(
        val file: File,
        val activity: ActivityType,
        val placement: PhonePlacement,
        val startedAtMs: Long,
        var samples: Int = 0,
    )

    private var writer: BufferedWriter? = null
    private var session: Session? = null

    val isRecording: Boolean get() = session != null
    val currentSession: Session? get() = session

    /**
     * Начинает запись. Если запись уже идёт, предыдущая сессия корректно
     * закрывается — потерять уже записанные данные из-за двойного нажатия нельзя.
     */
    fun start(activity: ActivityType, placement: PhonePlacement): Session {
        if (isRecording) stop()

        val dir = File(context.filesDir, DATASET_DIR).apply { mkdirs() }
        val name = "har_${activity.name}_${placement.name}_${stamp.format(Date())}.csv"
        val file = File(dir, name)

        val w = file.bufferedWriter()
        w.appendLine("${SensorFrame.CSV_HEADER},activity,placement")
        writer = w

        return Session(file, activity, placement, System.currentTimeMillis()).also {
            session = it
            Log.i(TAG, "Начата запись датасета: ${file.name}")
        }
    }

    /** Дописывает кадр. Вызывается 50 раз в секунду, поэтому строка собирается вручную. */
    fun write(frame: SensorFrame) {
        val s = session ?: return
        val w = writer ?: return
        try {
            w.append(frame.toCsvRow())
            w.append(',').append(s.activity.name)
            w.append(',').append(s.placement.name)
            w.newLine()
            s.samples++
        } catch (e: Exception) {
            Log.e(TAG, "Ошибка записи датасета, сессия остановлена", e)
            stop()
        }
    }

    /** Завершает запись и возвращает закрытую сессию (null, если записи не было). */
    fun stop(): Session? {
        val s = session ?: return null
        try {
            writer?.flush()
            writer?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Ошибка закрытия файла датасета", e)
        }
        writer = null
        session = null
        Log.i(TAG, "Запись завершена: ${s.file.name}, ${s.samples} отсчётов")

        // Файл без данных только мешает при обучении — удаляем.
        if (s.samples == 0) {
            s.file.delete()
            return null
        }
        return s
    }

    /** Все ранее записанные файлы датасета, свежие сверху. */
    fun listFiles(): List<File> {
        val dir = File(context.filesDir, DATASET_DIR)
        if (!dir.isDirectory) return emptyList()
        return dir.listFiles { f -> f.isFile && f.extension == "csv" }
            ?.sortedByDescending { it.lastModified() }
            ?: emptyList()
    }

    fun totalSizeBytes(): Long = listFiles().sumOf { it.length() }

    fun deleteAll(): Int {
        if (isRecording) stop()
        return listFiles().count { it.delete() }
    }

    companion object {
        private const val TAG = "DatasetRecorder"
        const val DATASET_DIR = "datasets"
        private val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    }
}
