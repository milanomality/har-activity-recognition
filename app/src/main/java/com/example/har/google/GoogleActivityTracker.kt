package com.example.har.google

import android.Manifest
import android.annotation.SuppressLint
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import com.example.har.ml.ActivityType
import com.google.android.gms.location.ActivityRecognition
import com.google.android.gms.location.ActivityRecognitionResult
import com.google.android.gms.location.DetectedActivity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Класс активности по версии Google Activity Recognition API. */
enum class GoogleActivityType(val title: String) {
    STILL("покой"),
    WALKING("ходьба"),
    ON_FOOT("пешком"),
    RUNNING("бег"),
    ON_BICYCLE("велосипед"),
    IN_VEHICLE("транспорт"),
    TILTING("наклон телефона"),
    UNKNOWN("неизвестно"),
    ;

    /**
     * Совпадает ли с нашим классом. null — сравнивать не с чем: транспорта
     * у нас нет, а «наклон» и «неизвестно» — не активность.
     * Лестницу Google не различает и относит к ходьбе, поэтому подъём и спуск
     * считаются совпадением с WALKING и ON_FOOT. ON_FOOT — «идёт или бежит».
     */
    fun agreesWith(ours: ActivityType): Boolean? = when (this) {
        STILL -> ours == ActivityType.STILL
        WALKING -> ours in WALKING_LIKE
        ON_FOOT -> ours in WALKING_LIKE || ours == ActivityType.RUNNING
        RUNNING -> ours == ActivityType.RUNNING
        ON_BICYCLE -> ours == ActivityType.CYCLING
        IN_VEHICLE, TILTING, UNKNOWN -> null
    }

    companion object {
        private val WALKING_LIKE = setOf(ActivityType.WALKING, ActivityType.STAIRS_UP, ActivityType.STAIRS_DOWN)

        fun fromDetected(type: Int): GoogleActivityType = when (type) {
            DetectedActivity.STILL -> STILL
            DetectedActivity.WALKING -> WALKING
            DetectedActivity.ON_FOOT -> ON_FOOT
            DetectedActivity.RUNNING -> RUNNING
            DetectedActivity.ON_BICYCLE -> ON_BICYCLE
            DetectedActivity.IN_VEHICLE -> IN_VEHICLE
            DetectedActivity.TILTING -> TILTING
            else -> UNKNOWN
        }

        fun fromName(name: String): GoogleActivityType? = entries.firstOrNull { it.name == name }
    }
}

/** Одно решение Google: самый вероятный класс и уверенность 0–100. */
data class GoogleActivity(
    val type: GoogleActivityType,
    val confidence: Int,
    val timeMs: Long,
)

/**
 * Подписка на Google Activity Recognition API — эталон для сверки.
 *
 * Google решает по своим моделям в Play Services, без положения телефона
 * и без лестницы, а обновления присылает не чаще, чем разрешит система
 * (обычно раз в 5–30 с, реже при неподвижности). Поэтому его ответ
 * не участвует в нашем решении: он только записывается рядом с ним,
 * чтобы видеть, где мы расходимся.
 */
object GoogleActivityTracker {

    private const val TAG = "GoogleActivity"
    private const val INTERVAL_MS = 5_000L

    /** Ответ старше этого к окну не прикрепляется — он уже не про это окно. */
    private const val MAX_AGE_MS = 60_000L

    private val _latest = MutableStateFlow<GoogleActivity?>(null)
    val latest: StateFlow<GoogleActivity?> = _latest.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    /** Почему сверка не работает (нет разрешения, нет Play Services), null — работает. */
    val error: StateFlow<String?> = _error.asStateFlow()

    private var pendingIntent: PendingIntent? = null

    fun hasPermission(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) ==
            PackageManager.PERMISSION_GRANTED

    /** Последний ответ, если он достаточно свежий для окна, закончившегося в [nowMs]. */
    fun current(nowMs: Long = System.currentTimeMillis()): GoogleActivity? =
        _latest.value?.takeIf { nowMs - it.timeMs <= MAX_AGE_MS }

    @SuppressLint("MissingPermission") // проверяется в hasPermission
    fun start(context: Context) {
        if (pendingIntent != null) return
        if (!hasPermission(context)) {
            _error.value = "Нет разрешения «Физическая активность» — сверка с Google выключена"
            return
        }
        val app = context.applicationContext
        // FLAG_MUTABLE обязателен: Play Services дописывает результат в extras интента.
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or
            (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0)
        val pi = PendingIntent.getBroadcast(app, 0, Intent(app, GoogleActivityReceiver::class.java), flags)
        ActivityRecognition.getClient(app).requestActivityUpdates(INTERVAL_MS, pi)
            .addOnSuccessListener {
                pendingIntent = pi
                _error.value = null
            }
            .addOnFailureListener { e ->
                Log.w(TAG, "Подписка на Google Activity Recognition не удалась", e)
                _error.value = "Google Activity Recognition недоступен: ${e.message}"
            }
    }

    @SuppressLint("MissingPermission")
    fun stop(context: Context) {
        val pi = pendingIntent ?: return
        pendingIntent = null
        ActivityRecognition.getClient(context.applicationContext).removeActivityUpdates(pi)
        pi.cancel()
    }

    internal fun publish(result: ActivityRecognitionResult) {
        val best = result.mostProbableActivity
        _latest.value = GoogleActivity(
            type = GoogleActivityType.fromDetected(best.type),
            confidence = best.confidence,
            timeMs = System.currentTimeMillis(),
        )
    }
}

/** Принимает решения Google, присланные через PendingIntent. */
class GoogleActivityReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (!ActivityRecognitionResult.hasResult(intent)) return
        ActivityRecognitionResult.extractResult(intent)?.let { GoogleActivityTracker.publish(it) }
    }
}
