package com.example.har.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.har.data.LogRepository
import com.example.har.data.db.ActivityIntervalEntity
import com.example.har.ml.ActivityType
import com.example.har.ui.HarViewModel
import com.example.har.ui.theme.ActivityPalette
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
private val dayFormat = SimpleDateFormat("d MMMM yyyy", Locale("ru"))

/**
 * Журнал активности за выбранные сутки.
 *
 * Показывает агрегированные интервалы, а не отдельные окна: окно длится
 * 1.28 с, и список из тысяч строк был бы нечитаем. Покадровый уровень
 * журнала остаётся доступным через выгрузку CSV.
 */
@Composable
fun JournalScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val day by vm.selectedDay.collectAsStateWithLifecycle()
    val intervals by vm.intervals.collectAsStateWithLifecycle()
    val summary by vm.summary.collectAsStateWithLifecycle()

    val totalMs = summary.sumOf { it.totalMs }
    val isToday = day == LogRepository.startOfToday()

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = { vm.shiftDay(-1) }) {
                    Icon(Icons.AutoMirrored.Filled.KeyboardArrowLeft, "Предыдущий день")
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = dayFormat.format(Date(day)),
                        style = MaterialTheme.typography.titleMedium,
                    )
                    if (!isToday) {
                        TextButton(onClick = vm::goToToday) { Text("Сегодня") }
                    }
                }
                IconButton(
                    onClick = { vm.shiftDay(1) },
                    enabled = !isToday,
                ) {
                    Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, "Следующий день")
                }
            }
        }

        if (summary.isEmpty()) {
            item {
                Card(
                    Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    ),
                ) {
                    Text(
                        text = "За этот день записей нет.\n\n" +
                            "Запустите распознавание на вкладке «Сейчас» — журнал начнёт " +
                            "заполняться примерно через три секунды после старта.",
                        modifier = Modifier.padding(20.dp),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            return@LazyColumn
        }

        item {
            SectionCard("Сводка за день") {
                // Стековая полоса: доли активностей в общем времени наблюдения.
                Row(
                    Modifier
                        .fillMaxWidth()
                        .height(16.dp)
                        .clip(RoundedCornerShape(8.dp))
                ) {
                    summary.forEach { row ->
                        val share = if (totalMs > 0) row.totalMs.toFloat() / totalMs else 0f
                        if (share > 0f) {
                            Box(
                                Modifier
                                    .weight(share)
                                    .fillMaxHeight()
                                    .background(ActivityPalette.forActivity(row.activity))
                            )
                        }
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    summary.forEach { row ->
                        val type = ActivityType.fromName(row.activity)
                        val share = if (totalMs > 0) row.totalMs.toFloat() / totalMs else 0f
                        Row(
                            Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Box(
                                Modifier
                                    .size(12.dp)
                                    .clip(RoundedCornerShape(3.dp))
                                    .background(ActivityPalette.forActivity(row.activity))
                            )
                            Text(
                                text = "  ${type?.emoji ?: ""} ${type?.title ?: row.activity}",
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                text = "${formatDuration(row.totalMs)}  ${(share * 100).toInt()} %",
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                            )
                        }
                    }
                }

                StatRow("Всего под наблюдением", formatDuration(totalMs))
                StatRow("Интервалов", summary.sumOf { it.intervalCount }.toString())
            }
        }

        item {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(onClick = vm::exportIntervals, modifier = Modifier.weight(1f)) {
                    Text("Выгрузить CSV")
                }
                OutlinedButton(onClick = vm::exportWindows, modifier = Modifier.weight(1f)) {
                    Text("Покадрово")
                }
            }
        }

        item {
            Text(
                text = "Интервалы (${intervals.size})",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(top = 8.dp),
            )
        }

        items(intervals, key = { it.id }) { interval ->
            IntervalRow(interval)
        }
    }
}

@Composable
private fun IntervalRow(interval: ActivityIntervalEntity) {
    val type = ActivityType.fromName(interval.activity)
    val placement = LogRepository.dominantPlacement(interval.placementCounts)
    val color = ActivityPalette.forActivity(interval.activity)

    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.12f)),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier
                    .size(width = 4.dp, height = 40.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(color)
            )
            Column(Modifier.padding(start = 12.dp).weight(1f)) {
                Text(
                    text = "${type?.emoji ?: ""} ${type?.title ?: interval.activity}",
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "${timeFormat.format(Date(interval.startMs))} — " +
                        timeFormat.format(Date(interval.endMs)),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (placement != null) {
                    Text(
                        text = "${placement.emoji} ${placement.title}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = formatDuration(interval.durationMs),
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "${(interval.averageConfidence * 100).toInt()} %",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
