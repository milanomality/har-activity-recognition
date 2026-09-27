package com.example.har.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.har.ml.ActivityRecognizer
import com.example.har.sensors.SensorHub
import com.example.har.sensors.SensorWindow
import com.example.har.ui.HarViewModel

/**
 * Служебный экран: что за модель работает, сколько накоплено данных
 * и как всё это выгрузить или стереть.
 */
@Composable
fun AboutScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val state by vm.engineState.collectAsStateWithLifecycle()
    val windows by vm.windowCount.collectAsStateWithLifecycle()
    var confirmClear by remember { mutableStateOf(false) }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Очистить журнал?") },
            text = {
                Text(
                    "Будут удалены все записи активности за всё время " +
                        "($windows окон). Файлы собранного датасета не затрагиваются."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    vm.clearJournal()
                    confirmClear = false
                }) { Text("Удалить") }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) { Text("Отмена") }
            },
        )
    }

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            SectionCard("Модель") {
                val status = state.status
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    ModelRow(
                        "Классификатор активности",
                        status?.activityModelLoaded == true,
                        ActivityRecognizer.ACTIVITY_MODEL_ASSET,
                    )
                    ModelRow(
                        "Классификатор положения",
                        status?.placementModelLoaded == true,
                        ActivityRecognizer.PLACEMENT_MODEL_ASSET,
                    )
                    if (status != null && status.activityAccuracy > 0f) {
                        StatRow(
                            "Точность на тесте (активность)",
                            "${(status.activityAccuracy * 100).toInt()} %",
                        )
                    }
                    if (status != null && status.placementAccuracy > 0f) {
                        StatRow(
                            "Точность на тесте (положение)",
                            "${(status.placementAccuracy * 100).toInt()} %",
                        )
                    }
                    if (status != null && status.usesContext) {
                        // Показываем измеренный вклад, а не утверждение «датчики
                        // используются». Нулевой вклад — тоже результат, и скрывать
                        // его значит выдавать архитектуру за пользу.
                        val delta = status.activityAccuracy - status.activityAccuracyWithoutContext
                        StatRow(
                            "Вклад освещённости и приближения",
                            if (status.activityAccuracyWithoutContext <= 0f) "не измерен"
                            else String.format(java.util.Locale.US, "%+.1f п.п.", delta * 100),
                        )
                    }
                }

                status?.warning?.let {
                    Card(
                        Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer
                        ),
                    ) {
                        Text(
                            text = it,
                            modifier = Modifier.padding(12.dp),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onTertiaryContainer,
                        )
                    }
                }
            }
        }

        item {
            SectionCard("Параметры обработки") {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatRow("Частота дискретизации", "${SensorHub.SAMPLE_RATE_HZ} Гц")
                    StatRow(
                        "Длина окна",
                        "${SensorWindow.WINDOW_SIZE} отсчётов " +
                            "(${SensorWindow.WINDOW_SIZE / SensorHub.SAMPLE_RATE_HZ.toFloat()} с)",
                    )
                    StatRow(
                        "Сдвиг окна",
                        "${SensorWindow.WINDOW_STRIDE} отсчётов " +
                            "(перекрытие ${100 - SensorWindow.WINDOW_STRIDE * 100 / SensorWindow.WINDOW_SIZE} %)",
                    )
                    val channels = state.status?.channels.orEmpty()
                    StatRow(
                        "Каналы на входе модели",
                        if (channels.isEmpty()) "—" else "${channels.size}",
                    )
                    if (channels.isNotEmpty()) {
                        Text(
                            text = channels.joinToString(", "),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }

        item {
            SectionCard("Данные") {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatRow("Записей в журнале (окон)", windows.toString())
                }
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    OutlinedButton(onClick = vm::exportWindows, modifier = Modifier.weight(1f)) {
                        Text("Выгрузить всё")
                    }
                    OutlinedButton(
                        onClick = { confirmClear = true },
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Очистить")
                    }
                }
            }
        }

        item {
            SectionCard("Как это работает") {
                Text(
                    text = "Показания пяти датчиков приводятся к общей сетке 50 Гц и " +
                        "нарезаются на окна по 2.56 с с перекрытием 50 %. " +
                        "Каналы движения (акселерометр, гироскоп, магнитометр) подаются " +
                        "в одномерную свёрточную сеть. Датчики приближения и освещённости " +
                        "вместе с оценкой ориентации образуют вектор из 16 признаков, " +
                        "который идёт вторым входом той же сети — так все пять датчиков " +
                        "влияют на определение активности. Тот же вектор использует " +
                        "отдельная сеть, определяющая, где находится телефон. " +
                        "Выходы сглаживаются по времени, после чего решение попадает " +
                        "в журнал: покадрово и в виде склеенных интервалов.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ModelRow(name: String, loaded: Boolean, assetName: String) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        StatusDot(loaded)
        Column(Modifier.padding(start = 8.dp)) {
            Text(name, style = MaterialTheme.typography.bodyMedium)
            Text(
                text = if (loaded) assetName else "$assetName — не загружена",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
