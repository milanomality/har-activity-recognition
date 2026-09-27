package com.example.har.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.har.ml.ActivityType
import com.example.har.ml.PhonePlacement
import com.example.har.ui.HarViewModel
import com.example.har.ui.theme.ActivityPalette

/**
 * Главный экран: что человек делает прямо сейчас и на основании чего.
 *
 * Экран намеренно показывает не только победивший класс, но и полное
 * распределение вероятностей и сырые показания датчиков — без этого
 * невозможно ни отладить модель, ни объяснить её поведение на защите.
 */
@Composable
fun LiveScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val state by vm.engineState.collectAsStateWithLifecycle()
    val result by vm.latest.collectAsStateWithLifecycle()
    val frame by vm.liveFrame.collectAsStateWithLifecycle()

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            val prediction = result?.prediction
            Card(
                Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (prediction != null) {
                        ActivityPalette.forActivity(prediction.activity.name).copy(alpha = 0.18f)
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    }
                ),
            ) {
                Column(
                    Modifier.fillMaxWidth().padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = prediction?.activity?.emoji ?: "⏸",
                        style = MaterialTheme.typography.displayMedium,
                    )
                    Text(
                        text = prediction?.activity?.title
                            ?: if (state.running) "Накопление окна…" else "Распознавание остановлено",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    if (prediction != null) {
                        Text(
                            text = "уверенность ${(prediction.activityConfidence * 100).toInt()} % · " +
                                prediction.source.title,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            text = "${prediction.placement.emoji} ${prediction.placement.title} " +
                                "(${(prediction.placementConfidence * 100).toInt()} %)",
                            style = MaterialTheme.typography.bodyLarge,
                        )
                    }
                }
            }
        }

        item {
            Button(
                onClick = vm::toggleRecognition,
                modifier = Modifier.fillMaxWidth(),
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

        state.error?.let { error ->
            item {
                Card(
                    Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    ),
                ) {
                    Text(
                        text = error,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }

        state.status?.warning?.let { warning ->
            item {
                Card(
                    Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer
                    ),
                ) {
                    Text(
                        text = warning,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        result?.let { r ->
            item {
                SectionCard("Распределение по активностям") {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        ActivityType.entries
                            .sortedByDescending { r.prediction.probabilities.getOrElse(it.id) { 0f } }
                            .forEach { type ->
                                ProbabilityBar(
                                    label = "${type.emoji} ${type.title}",
                                    probability = r.prediction.probabilities.getOrElse(type.id) { 0f },
                                    color = ActivityPalette.forActivity(type.name),
                                    highlighted = type == r.prediction.activity,
                                )
                            }
                    }
                }
            }

            item {
                SectionCard("Положение телефона") {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        PhonePlacement.entries.forEach { placement ->
                            ProbabilityBar(
                                label = "${placement.emoji} ${placement.title}",
                                probability = r.prediction.placementProbabilities
                                    .getOrElse(placement.id) { 0f },
                                color = MaterialTheme.colorScheme.secondary,
                                highlighted = placement == r.prediction.placement,
                            )
                        }
                    }
                }
            }

            item {
                SectionCard("Признаки окна (2.56 с)") {
                    val s = r.stats
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        StatRow("СКО ускорения", "${formatFloat(s.accMagStd)} м/с²")
                        StatRow("Частота главного пика", "${formatFloat(s.dominantFreqHz)} Гц")
                        // Мощность пика, а не СКЗ линейного ускорения: последнее
                        // в FeatureExtractor.stats получается вычитанием среднего
                        // модуля и потому тождественно равно accMagStd — строка
                        // дублировала бы предыдущую.
                        StatRow("Мощность главного пика", "${formatFloat(s.dominantPower, 3)} м/с²")
                        StatRow("Спектральная энтропия", formatFloat(s.spectralEntropy))
                        StatRow("Среднее вращение", "${formatFloat(s.gyroMagMean)} рад/с")
                        StatRow("Наклон телефона", "${formatFloat(s.tiltDeg, 0)}°")
                        StatRow("Стабильность ориентации", formatFloat(s.orientationStd, 4))
                    }
                }
            }
        }

        item {
            SectionCard("Показания датчиков") {
                val f = frame
                val availability = state.availability
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    SensorRow(
                        "Акселерометр",
                        availability?.accelerometer == true,
                        f?.let { "${formatFloat(it.ax)}  ${formatFloat(it.ay)}  ${formatFloat(it.az)}" },
                        "м/с²",
                    )
                    SensorRow(
                        "Гироскоп",
                        availability?.gyroscope == true,
                        f?.let { "${formatFloat(it.gx)}  ${formatFloat(it.gy)}  ${formatFloat(it.gz)}" },
                        "рад/с",
                    )
                    SensorRow(
                        "Магнитометр",
                        availability?.magnetometer == true,
                        f?.let { "${formatFloat(it.mx, 1)}  ${formatFloat(it.my, 1)}  ${formatFloat(it.mz, 1)}" },
                        "мкТл",
                    )
                    SensorRow(
                        "Приближение",
                        availability?.proximity == true,
                        f?.let { if (it.proximityNear) "перекрыт" else "открыт" },
                        "",
                    )
                    SensorRow(
                        "Освещённость",
                        availability?.light == true,
                        f?.let { if (it.lightLux < 0) "нет данных" else formatFloat(it.lightLux, 0) },
                        "лк",
                    )
                }

                val missing = availability?.missing.orEmpty()
                if (missing.isNotEmpty()) {
                    Text(
                        text = "Недоступны: ${missing.joinToString(", ")}. " +
                            "Соответствующие признаки подаются в модель как нулевые.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        item {
            SectionCard("Статистика сессии") {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatRow("Состояние", if (state.running) "работает" else "остановлено")
                    StatRow("Обработано кадров", state.framesProcessed.toString())
                    StatRow("Классифицировано окон", state.windowsProcessed.toString())
                    if (state.running && state.startedAtMs > 0) {
                        StatRow(
                            "Время работы",
                            formatDuration(System.currentTimeMillis() - state.startedAtMs),
                        )
                    }
                    // До первого запуска модели не загружены, и состояние
                    // классификатора неизвестно. Писать «эвристика» в этот
                    // момент — врать: модель может быть на месте.
                    StatRow(
                        "Классификатор",
                        when {
                            state.status == null -> "неизвестен до запуска"
                            state.status?.usingNeuralNet == true -> "нейросеть (TFLite)"
                            else -> "эвристика"
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun SensorRow(name: String, available: Boolean, value: String?, unit: String) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            StatusDot(available)
            Text(
                text = "  $name",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            text = when {
                !available -> "нет датчика"
                value == null -> "—"
                unit.isEmpty() -> value
                else -> "$value $unit"
            },
            style = MaterialTheme.typography.bodySmall,
            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
        )
    }
}
