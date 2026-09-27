package com.example.har.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.har.ml.ActivityType
import com.example.har.ml.PhonePlacement
import com.example.har.sensors.SensorHub
import com.example.har.ui.HarViewModel
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val fileTimeFormat = SimpleDateFormat("dd.MM HH:mm", Locale.getDefault())

/**
 * Режим сбора размеченных данных для обучения собственной модели.
 *
 * Пользователь заранее указывает, что он будет делать и где будет телефон,
 * затем нажимает «Начать» — и приложение пишет сырые показания всех датчиков
 * в CSV с этой меткой. Полученные файлы забирает `ml/train.py`.
 */
@Composable
fun CollectScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val state by vm.collection.collectAsStateWithLifecycle()
    val engineState by vm.engineState.collectAsStateWithLifecycle()

    // Счётчик отсчётов растёт в рекордере, а не в StateFlow: опрашиваем его
    // раз в секунду, чтобы не гонять рекомпозицию 50 раз в секунду.
    LaunchedEffect(state.recording) {
        while (state.recording) {
            vm.refreshCollectionProgress()
            delay(1000)
        }
    }

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Card(
                Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                ),
            ) {
                Text(
                    text = "Разметка выставляется до начала записи. Займите нужное положение, " +
                        "нажмите «Начать запись» и выполняйте выбранную активность без пауз " +
                        "хотя бы 2–3 минуты. Чем однороднее запись, тем чище обучающая выборка.",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }

        item {
            SectionCard("Что записываем") {
                Text("Активность", style = MaterialTheme.typography.labelLarge)
                ChipGroup {
                    ActivityType.entries.forEach { type ->
                        FilterChip(
                            selected = state.activity == type,
                            onClick = { vm.setCollectionActivity(type) },
                            enabled = !state.recording,
                            label = { Text("${type.emoji} ${type.title}") },
                            colors = FilterChipDefaults.filterChipColors(),
                        )
                    }
                }

                Text("Положение телефона", style = MaterialTheme.typography.labelLarge)
                ChipGroup {
                    PhonePlacement.entries.forEach { placement ->
                        FilterChip(
                            selected = state.placement == placement,
                            onClick = { vm.setCollectionPlacement(placement) },
                            enabled = !state.recording,
                            label = { Text("${placement.emoji} ${placement.title}") },
                        )
                    }
                }
            }
        }

        item {
            Card(
                Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (state.recording) {
                        MaterialTheme.colorScheme.errorContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    }
                ),
            ) {
                Column(
                    Modifier.fillMaxWidth().padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (state.recording) {
                        Text("● ЗАПИСЬ", style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold)
                        Text(
                            text = "${state.samples} отсчётов",
                            style = MaterialTheme.typography.headlineMedium,
                            fontFamily = FontFamily.Monospace,
                        )
                        Text(
                            text = "≈ ${state.samples / SensorHub.SAMPLE_RATE_HZ} с · " +
                                "${state.activity.title} · ${state.placement.title}",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    } else {
                        Text(
                            text = "Готово к записи: ${state.activity.title}, " +
                                state.placement.title,
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.SemiBold,
                        )
                        if (!engineState.running) {
                            Text(
                                text = "Датчики сейчас выключены — запись запустит их автоматически.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }

        item {
            Button(
                onClick = { if (state.recording) vm.stopCollection() else vm.startCollection() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (state.recording) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary
                ),
            ) {
                Text(if (state.recording) "Остановить запись" else "Начать запись")
            }
        }

        item {
            SectionCard(
                title = "Записанные файлы (${state.files.size})",
                trailing = {
                    if (state.files.isNotEmpty()) {
                        TextButton(onClick = vm::deleteDatasets) { Text("Удалить все") }
                    }
                },
            ) {
                if (state.files.isEmpty()) {
                    Text(
                        text = "Пока пусто. Файлы лежат во внутренней памяти приложения " +
                            "и выгружаются кнопкой «Поделиться» на каждой строке.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    StatRow("Суммарный объём", "${state.totalBytes / 1024} КБ")
                }
            }
        }

        items(state.files, key = { it.absolutePath }) { file ->
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        text = file.name,
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                    )
                    Text(
                        text = "${fileTimeFormat.format(Date(file.lastModified()))} · " +
                            "${file.length() / 1024} КБ",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                OutlinedButton(onClick = { vm.shareDataset(file) }) { Text("Поделиться") }
            }
        }
    }
}

/**
 * «Резиновая» группа чипов: варианты переносятся на следующую строку,
 * а не сжимаются — названия активностей длинные и обрезать их нельзя.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ChipGroup(content: @Composable () -> Unit) {
    FlowRow(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        content()
    }
}
