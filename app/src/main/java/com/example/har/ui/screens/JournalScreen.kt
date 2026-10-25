package com.example.har.ui.screens

import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.lazy.rememberLazyListState
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
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
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
import kotlinx.coroutines.launch

private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
private val dayFormat = SimpleDateFormat("d MMMM yyyy", Locale("ru"))

/**
 * Журнал активности за выбранные сутки.
 *
 * Показывает агрегированные интервалы, а не отдельные окна: окно длится
 * 1.28 с, и список из тысяч строк был бы нечитаем. Отдельное окно можно
 * найти ползунком хронологии, весь покадровый уровень — выгрузить в CSV.
 */
@Composable
fun JournalScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val day by vm.selectedDay.collectAsStateWithLifecycle()
    val intervals by vm.intervals.collectAsStateWithLifecycle()
    val summary by vm.summary.collectAsStateWithLifecycle()
    val cursorMs by vm.cursorMs.collectAsStateWithLifecycle()
    val cursorWindow by vm.cursorWindow.collectAsStateWithLifecycle()

    val totalMs = summary.sumOf { it.totalMs }
    val isToday = day == LogRepository.startOfToday()
    // Список идёт от новых к старым, хронология — слева направо.
    val timeline = remember(intervals) { intervals.sortedBy { it.startMs } }
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()

    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
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

        val cursor = cursorMs
        if (cursor != null && timeline.isNotEmpty()) {
            item(key = TIMELINE_KEY) {
                TimelineCard(
                    intervals = timeline,
                    cursorMs = cursor,
                    window = cursorWindow,
                    onCursorChange = vm::setCursor,
                )
            }
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
            val cursor = cursorMs
            IntervalRow(
                interval = interval,
                selected = cursor != null && cursor in interval.startMs..interval.endMs,
                onClick = {
                    // Нажатие на интервал ставит на него ползунок и поднимает к хронологии.
                    vm.setCursor(interval.startMs)
                    scope.launch { listState.animateScrollToItem(TIMELINE_INDEX) }
                },
            )
        }
    }
}

private const val TIMELINE_KEY = "timeline"

/** Позиция хронологии в списке: сразу под строкой выбора дня. */
private const val TIMELINE_INDEX = 1

@Composable
private fun IntervalRow(
    interval: ActivityIntervalEntity,
    selected: Boolean,
    onClick: () -> Unit,
) {
    val type = ActivityType.fromName(interval.activity)
    val placement = LogRepository.dominantPlacement(interval.placementCounts)
    val color = ActivityPalette.forActivity(interval.activity)

    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.12f)),
        border = if (selected) BorderStroke(2.dp, color) else null,
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
