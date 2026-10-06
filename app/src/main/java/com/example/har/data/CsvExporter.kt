package com.example.har.data

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.example.har.ml.ActivityType
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Выгрузка журнала в CSV.
 *
 * Файлы кладутся в cacheDir/exports и отдаются наружу через FileProvider:
 * прямой путь к файлу приложения другому приложению отдать нельзя,
 * а запрашивать разрешение на внешнее хранилище ради выгрузки отчёта — избыточно.
 */
object CsvExporter {

    private const val EXPORT_DIR = "exports"
    private val fileStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    private val isoTime = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

    /** Покадровый журнал со всеми признаками — то, что нужно для анализа. */
    suspend fun exportWindows(context: Context, repository: LogRepository): File {
        val rows = repository.allWindows()
        val file = newFile(context, "har_windows")
        file.bufferedWriter().use { w ->
            w.appendLine(
                "id,start_time,end_time,start_ms,end_ms,activity,confidence,source," +
                    "placement,placement_confidence,acc_mag_mean,acc_mag_std,lin_acc_rms," +
                    "gyro_mag_mean,gyro_mag_std,mag_mag_mean,light_lux,proximity_near_ratio," +
                    "dominant_freq_hz,spectral_entropy,tilt_deg,speed_ms,speed_reliable," +
                    "vertical_acc_rms,horizontal_acc_rms,jerk_rms,tilt_swing_deg,yaw_rate_mean,steps,cadence_hz,step_speed_ms,step_regularity,mag_inclination_deg,mag_disturbed_ratio,mag_gyro_mismatch_deg," +
                    "grav_x,grav_y,grav_z,orientation_std,ear_pose,google_activity,google_confidence,google_agrees," +
                    "probabilities,placement_probabilities,model_placement_probabilities,model_activity_probabilities"
            )
            for (r in rows) {
                w.append(r.id.toString()).append(',')
                w.append(isoTime.format(Date(r.startMs))).append(',')
                w.append(isoTime.format(Date(r.endMs))).append(',')
                w.append(r.startMs.toString()).append(',')
                w.append(r.endMs.toString()).append(',')
                w.append(r.activity).append(',')
                w.append(fmt(r.confidence)).append(',')
                w.append(r.source).append(',')
                w.append(r.placement).append(',')
                w.append(fmt(r.placementConfidence)).append(',')
                w.append(fmt(r.accMagMean)).append(',')
                w.append(fmt(r.accMagStd)).append(',')
                w.append(fmt(r.linAccRms)).append(',')
                w.append(fmt(r.gyroMagMean)).append(',')
                w.append(fmt(r.gyroMagStd)).append(',')
                w.append(fmt(r.magMagMean)).append(',')
                w.append(fmt(r.lightLux)).append(',')
                w.append(fmt(r.proximityNearRatio)).append(',')
                w.append(fmt(r.dominantFreqHz)).append(',')
                w.append(fmt(r.spectralEntropy)).append(',')
                w.append(fmt(r.tiltDeg)).append(',')
                w.append(fmt(r.speedMs)).append(',')
                w.append(if (r.speedReliable) "1" else "0").append(',')
                w.append(fmt(r.verticalAccRms)).append(',')
                w.append(fmt(r.horizontalAccRms)).append(',')
                w.append(fmt(r.jerkRms)).append(',')
                w.append(fmt(r.tiltSwingDeg)).append(',')
                w.append(fmt(r.yawRateMean)).append(',')
                w.append(r.stepsInWindow.toString()).append(',')
                w.append(fmt(r.cadenceHz)).append(',')
                w.append(fmt(r.stepSpeedMs)).append(',')
                w.append(fmt(r.stepRegularity)).append(',')
                w.append(fmt(r.magInclinationDeg)).append(',')
                w.append(fmt(r.magDisturbedRatio)).append(',')
                w.append(fmt(r.magGyroMismatchDeg)).append(',')
                w.append(fmt(r.gravityX)).append(',')
                w.append(fmt(r.gravityY)).append(',')
                w.append(fmt(r.gravityZ)).append(',')
                w.append(fmt(r.orientationStd)).append(',')
                w.append(fmt(r.earPose)).append(',')
                w.append(r.googleActivity).append(',')
                w.append(r.googleConfidence.toString()).append(',')
                w.append(r.googleAgrees.toString()).append(',')
                // Массивы вероятностей содержат запятые — каждый в кавычках.
                listOf(
                    r.probabilities, r.placementProbabilities,
                    r.modelPlacementProbabilities, r.modelActivityProbabilities,
                ).forEachIndexed { i, json ->
                    if (i > 0) w.append(',')
                    w.append('"').append(json).append('"')
                }
                w.appendLine()
            }
        }
        return file
    }

    /** Человекочитаемый журнал интервалов — то, что показывается на экране. */
    suspend fun exportIntervals(context: Context, repository: LogRepository): File {
        val rows = repository.allIntervals()
        val file = newFile(context, "har_intervals")
        file.bufferedWriter().use { w ->
            w.appendLine(
                "id,activity,activity_ru,start_time,end_time,duration_sec," +
                    "windows,avg_confidence,dominant_placement,dominant_placement_ru"
            )
            for (r in rows) {
                val placement = LogRepository.dominantPlacement(r.placementCounts)
                w.append(r.id.toString()).append(',')
                w.append(r.activity).append(',')
                w.append(ActivityType.fromName(r.activity)?.title ?: r.activity).append(',')
                w.append(isoTime.format(Date(r.startMs))).append(',')
                w.append(isoTime.format(Date(r.endMs))).append(',')
                w.append((r.durationMs / 1000).toString()).append(',')
                w.append(r.windowCount.toString()).append(',')
                w.append(fmt(r.averageConfidence)).append(',')
                w.append(placement?.name ?: "").append(',')
                w.append(placement?.title ?: "")
                w.appendLine()
            }
        }
        return file
    }

    /** Intent «Поделиться» для готового файла. */
    fun shareIntent(context: Context, file: File, title: String): Intent {
        val uri: Uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file,
        )
        return Intent.createChooser(
            Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, file.name)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            },
            title,
        )
    }

    private fun newFile(context: Context, prefix: String): File {
        val dir = File(context.cacheDir, EXPORT_DIR).apply { mkdirs() }
        return File(dir, "${prefix}_${fileStamp.format(Date())}.csv")
    }

    private fun fmt(v: Float): String =
        if (v.isNaN() || v.isInfinite()) "" else String.format(Locale.US, "%.4f", v)
}
