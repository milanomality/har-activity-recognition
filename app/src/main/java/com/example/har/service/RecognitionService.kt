package com.example.har.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.LifecycleService
import com.example.har.HarApplication
import com.example.har.MainActivity
import com.example.har.R
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/**
 * Передний сервис, удерживающий распознавание во время погашенного экрана.
 *
 * Без него Android усыпляет процесс через несколько минут, и журнал
 * получается дырявым — а смысл приложения именно в непрерывном наблюдении.
 * Собственно вычисления живут в [RecognitionEngine]; сервис отвечает только
 * за жизненный цикл и за уведомление, показывающее текущую активность.
 */
class RecognitionService : LifecycleService() {

    private val engine: RecognitionEngine
        get() = (application as HarApplication).engine

    override fun onCreate() {
        super.onCreate()
        createChannel()

        // Уведомление обновляем по последнему предсказанию: пользователь
        // должен видеть, что именно приложение сейчас про него думает,
        // не открывая экран.
        lifecycleScope.launch {
            engine.latest.collectLatest { result ->
                if (result == null) return@collectLatest
                val p = result.prediction
                val percent = (p.activityConfidence * 100).toInt()
                notify(
                    title = "${p.activity.emoji} ${p.activity.title} · $percent %",
                    text = "${p.placement.emoji} ${p.placement.title} · ${p.source.title}",
                )
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)

        when (intent?.action) {
            ACTION_STOP -> {
                stopRecognition()
                return START_NOT_STICKY
            }
        }

        startForegroundCompat(buildNotification("Запуск распознавания…", "Накопление первого окна"))
        engine.start(lifecycleScope)

        // START_STICKY: если система убила процесс из-за нехватки памяти,
        // сервис поднимется сам и сбор продолжится.
        return START_STICKY
    }

    private fun stopRecognition() {
        engine.stop()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        engine.stop()
        super.onDestroy()
    }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun notify(title: String, text: String) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, buildNotification(title, text))
    }

    private fun buildNotification(title: String, text: String): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, RecognitionService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setContentIntent(openApp)
            .addAction(0, "Остановить", stop)
            .setOngoing(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.channel_name),
                // IMPORTANCE_LOW: уведомление должно висеть, но не звенеть
                // каждые несколько секунд при обновлении текста.
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = getString(R.string.channel_desc)
                setShowBadge(false)
            }
        )
    }

    companion object {
        private const val CHANNEL_ID = "recognition"
        private const val NOTIFICATION_ID = 1001
        const val ACTION_STOP = "com.example.har.STOP"

        fun start(context: Context) {
            val intent = Intent(context, RecognitionService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, RecognitionService::class.java).setAction(ACTION_STOP)
            )
        }
    }
}
