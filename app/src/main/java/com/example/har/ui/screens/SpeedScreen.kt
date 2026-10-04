package com.example.har.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.har.motion.InertialSpeedEstimator
import com.example.har.ui.HarViewModel

/**
 * Экран «Скорость»: только текущая горизонтальная скорость крупным числом.
 *
 * Берётся из последнего кадра (обновляется 5 раз в секунду), а не из окна:
 * окно даёт среднее за 2.56 с и запаздывает. Скорость считается только
 * пока идёт распознавание — отдельного сбора для неё нет.
 */
@Composable
fun SpeedScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val state by vm.engineState.collectAsStateWithLifecycle()
    val frame by vm.liveFrame.collectAsStateWithLifecycle()

    val f = frame
    val computed = state.running && f != null && f.secondsSinceZupt >= 0f
    val reliable = computed && InertialSpeedEstimator.isReliable(f!!.secondsSinceZupt, f.rotationSinceZupt)

    Column(
        modifier = modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically),
    ) {
        Text(
            text = if (computed) formatFloat(f!!.speedMs * 3.6f, 1) else "—",
            fontSize = 96.sp,
            fontWeight = FontWeight.Bold,
            color = if (reliable || !computed) MaterialTheme.colorScheme.onBackground
            else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
        )
        Text(
            text = "км/ч",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (computed) {
            Text(
                text = "${formatFloat(f!!.speedMs, 2)} м/с",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            // Вторая оценка — по шагам: не накапливает ошибку и потому полезна
            // именно тогда, когда навигация уже «ненадёжна» (ходьба с телефоном в кармане).
            Text(
                text = if (f.cadenceHz > 0f) {
                    "По шагам: ${formatFloat(f.stepSpeedMs * 3.6f, 1)} км/ч · " +
                        "${formatFloat(f.cadenceHz * 60f, 0)} шаг/мин · всего ${f.stepCount}"
                } else {
                    "По шагам: нет ритма · всего ${f.stepCount}"
                },
                style = MaterialTheme.typography.bodyLarge,
                color = if (!reliable && f.cadenceHz > 0f) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Text(
            text = when {
                !state.running -> "Скорость считается во время распознавания"
                !computed -> "Ожидание первых показаний…"
                // Покой засекается с перерывами: полсекунды после ZUPT — ещё «стоит».
                f!!.secondsSinceZupt < 0.5f -> "Телефон неподвижен"
                // Без остановок ошибка интегрирования растёт: число показываем
                // бледным, чтобы его не приняли за точное.
                !reliable && f.rotationSinceZupt > InertialSpeedEstimator.MAX_RELIABLE_ROTATION_RAD ->
                    "Ненадёжно: телефон сильно вращался (${formatFloat(f.rotationSinceZupt, 0)} рад без остановки)"
                !reliable -> "Ненадёжно: ${formatFloat(f.secondsSinceZupt, 0)} с без остановки"
                else -> "С последней остановки ${formatFloat(f.secondsSinceZupt, 0)} с"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = if (computed && !reliable) MaterialTheme.colorScheme.error
            else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 8.dp),
        )

        Button(
            onClick = vm::toggleRecognition,
            modifier = Modifier.fillMaxWidth().padding(top = 24.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (state.running) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.primary
            ),
        ) {
            Icon(
                imageVector = if (state.running) Icons.Default.Stop else Icons.Default.PlayArrow,
                contentDescription = null,
            )
            Text(
                text = if (state.running) "  Остановить" else "  Начать распознавание",
                modifier = Modifier.padding(start = 4.dp),
            )
        }
    }
}
