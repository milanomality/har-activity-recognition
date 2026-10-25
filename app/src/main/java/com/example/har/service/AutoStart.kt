package com.example.har.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Автоматический запуск распознавания.
 *
 * Смысл приложения — непрерывный журнал дня, поэтому распознавание не должно
 * ждать нажатия кнопки: оно включается само при включении телефона, после
 * обновления приложения и при его открытии.
 *
 * Ручная остановка уважается: если пользователь нажал «Остановить», открытие
 * приложения распознавание не перезапускает — до следующей перезагрузки
 * телефона или до нажатия «Начать». Автозапуск можно выключить совсем
 * переключателем на экране «Сейчас».
 */
object AutoStart {

    private const val TAG = "AutoStart"
    private const val PREFS = "autostart"
    private const val KEY_ENABLED = "enabled"
    private const val KEY_STOPPED_BY_USER = "stopped_by_user"

    private val _enabled = MutableStateFlow(true)
    /** Включён ли автозапуск — для переключателя на экране. */
    val enabled: StateFlow<Boolean> = _enabled.asStateFlow()

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun isEnabled(context: Context): Boolean =
        prefs(context).getBoolean(KEY_ENABLED, true).also { _enabled.value = it }

    fun setEnabled(context: Context, value: Boolean) {
        prefs(context).edit().putBoolean(KEY_ENABLED, value).apply()
        _enabled.value = value
    }

    /** Пользователь остановил распознавание сам или запустил его снова. */
    fun setStoppedByUser(context: Context, value: Boolean) {
        prefs(context).edit().putBoolean(KEY_STOPPED_BY_USER, value).apply()
    }

    private fun stoppedByUser(context: Context) = prefs(context).getBoolean(KEY_STOPPED_BY_USER, false)

    /** Телефон включился: ручная остановка прошлой сессии больше не действует. */
    fun onBoot(context: Context) {
        if (!isEnabled(context)) return
        setStoppedByUser(context, false)
        start(context, "включение телефона")
    }

    /** Приложение обновили: продолжаем, если его не останавливали вручную. */
    fun onAppUpdated(context: Context) {
        if (isEnabled(context) && !stoppedByUser(context)) start(context, "обновление приложения")
    }

    /** Приложение открыли: запускаем, если не работает и его не останавливали вручную. */
    fun onAppOpened(context: Context, running: Boolean) {
        if (!running && isEnabled(context) && !stoppedByUser(context)) start(context, "открытие приложения")
    }

    private fun start(context: Context, reason: String) {
        try {
            RecognitionService.start(context.applicationContext)
            Log.i(TAG, "Распознавание запущено автоматически: $reason")
        } catch (e: Exception) {
            // Система может запретить запуск переднего сервиса из фона
            // (ForegroundServiceStartNotAllowedException) — приложение
            // при этом не должно падать: распознавание запустится при открытии.
            Log.w(TAG, "Автозапуск не удался ($reason)", e)
        }
    }
}

/** Включение телефона и обновление приложения. */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED -> AutoStart.onBoot(context)
            Intent.ACTION_MY_PACKAGE_REPLACED -> AutoStart.onAppUpdated(context)
        }
    }
}
