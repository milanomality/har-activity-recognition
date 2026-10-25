package com.example.har.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.SkipNext
import androidx.compose.material.icons.filled.SkipPrevious
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.setProgress
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.har.data.db.ActivityIntervalEntity
import com.example.har.data.db.ActivityWindowEntity
import com.example.har.google.GoogleActivityType
import com.example.har.ml.ActivityType
import com.example.har.ml.InferenceSource
import com.example.har.ml.PhonePlacement
import com.example.har.ui.theme.ActivityPalette
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

private val clockFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
private val shortClockFormat = SimpleDateFormat("HH:mm", Locale.getDefault())

/** Масштаб хронологии: сколько времени помещается на ширину полосы. */
private enum class TimelineZoom(val title: String, val spanMs: Long?) {
    DAY("День", null),
    HOUR("1 ч", 60 * 60 * 1000L),
    TEN_MINUTES("10 мин", 10 * 60 * 1000L),
}

/**
 * Хронология дня с ползунком: интервалы журнала на оси времени и подробности
 * решения в выбранный момент.
 *
 * Ось идёт от первого интервала дня до последнего, а не охватывает сутки:
 * запись обычно длится несколько часов и на суточной шкале сжалась бы в узкую
 * полоску. Пропуски между интервалами остаются пустыми — журнал тогда не вёлся.
 * Окно 1.28 с на шкале целого дня тоньше пикселя, поэтому есть масштаб
 * и переход к соседнему интервалу.
 *
 * @param intervals интервалы дня по возрастанию времени, не пустой список
 */
@Composable
fun TimelineCard(
    intervals: List<ActivityIntervalEntity>,
    cursorMs: Long,
    window: ActivityWindowEntity?,
    onCursorChange: (Long) -> Unit,
    modifier: Modifier = Modifier,
) {
    val dayStart = intervals.first().startMs
    // Защита от нулевого размаха: день из одного окна.
    val dayEnd = maxOf(intervals.maxOf { it.endMs }, dayStart + 1_000)

    var zoom by rememberSaveable { mutableStateOf(TimelineZoom.DAY) }
    // Середина видимого отрезка при увеличении; при смене масштаба встаёт на ползунок.
    var anchor by rememberSaveable(zoom) { mutableLongStateOf(cursorMs) }

    val span = minOf(zoom.spanMs ?: Long.MAX_VALUE, dayEnd - dayStart)
    fun viewStartAround(center: Long) = (center - span / 2).coerceIn(dayStart, dayEnd - span)
    val anchoredStart = viewStartAround(anchor)
    val cursorVisible = cursorMs in anchoredStart..anchoredStart + span
    // Ползунок ушёл за край (кнопки, новые окна во время записи) — отрезок переезжает к нему.
    val viewStart = if (cursorVisible) anchoredStart else viewStartAround(cursorMs)
    val viewEnd = viewStart + span
    LaunchedEffect(cursorVisible) {
        if (!cursorVisible) anchor = cursorMs
    }

    val current = intervals.lastOrNull { it.startMs <= cursorMs }?.takeIf { cursorMs <= it.endMs }
    // Допуск в секунду: «назад» из самого начала интервала ведёт к предыдущему, а не к нему же.
    val previousStart = intervals.lastOrNull { it.startMs < cursorMs - 1_000 }?.startMs
    val nextStart = intervals.firstOrNull { it.startMs > cursorMs }?.startMs
    val axisFormat = if (span <= TimelineZoom.TEN_MINUTES.spanMs!!) clockFormat else shortClockFormat

    SectionCard("Хронология", modifier) {
        MomentHeader(cursorMs, current)

        Column {
            TimelineStrip(intervals, viewStart, viewEnd, cursorMs, onCursorChange)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(
                    text = axisFormat.format(Date(viewStart)),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = axisFormat.format(Date(viewEnd)),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            IconButton(
                onClick = { previousStart?.let(onCursorChange) },
                enabled = previousStart != null,
            ) {
                Icon(Icons.Filled.SkipPrevious, "Предыдущий интервал")
            }
            Row(
                Modifier.weight(1f),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
            ) {
                TimelineZoom.entries.forEach { entry ->
                    FilterChip(
                        selected = zoom == entry,
                        onClick = { zoom = entry },
                        label = { Text(entry.title) },
                        // Заливка основным цветом: в монохромной системной палитре
                        // стандартный фон выбранного чипа сливается с карточкой.
                        colors = FilterChipDefaults.filterChipColors(
                            selectedContainerColor = MaterialTheme.colorScheme.primary,
                            selectedLabelColor = MaterialTheme.colorScheme.onPrimary,
                        ),
                    )
                }
            }
            IconButton(
                onClick = { nextStart?.let(onCursorChange) },
                enabled = nextStart != null,
            ) {
                Icon(Icons.Filled.SkipNext, "Следующий интервал")
            }
        }

        if (window != null) MomentDetails(window)
    }
}

/** Время под ползунком и интервал, в который оно попало. */
@Composable
private fun MomentHeader(cursorMs: Long, interval: ActivityIntervalEntity?) {
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = clockFormat.format(Date(cursorMs)),
                style = MaterialTheme.typography.titleLarge,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.width(12.dp))
            if (interval != null) {
                val type = ActivityType.fromName(interval.activity)
                Box(
                    Modifier
                        .size(12.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(ActivityPalette.forActivity(interval.activity))
                )
                Text(
                    text = "  ${type?.emoji ?: ""} ${type?.title ?: interval.activity}",
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            } else {
                Text(
                    text = "журнал не вёлся",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        if (interval != null) {
            Text(
                text = "${clockFormat.format(Date(interval.startMs))} — " +
                    "${clockFormat.format(Date(interval.endMs))} · ${formatDuration(interval.durationMs)}",
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Полоса интервалов с ползунком. Нажатие ставит ползунок в точку, протяжка
 * по горизонтали ведёт его за пальцем; вертикальное движение остаётся списку.
 */
@Composable
private fun TimelineStrip(
    intervals: List<ActivityIntervalEntity>,
    viewStart: Long,
    viewEnd: Long,
    cursorMs: Long,
    onScrub: (Long) -> Unit,
) {
    val trackColor = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
    val handleColor = MaterialTheme.colorScheme.onSurface
    val ringColor = MaterialTheme.colorScheme.surface
    val currentOnScrub by rememberUpdatedState(onScrub)
    val currentStart by rememberUpdatedState(viewStart)
    val currentEnd by rememberUpdatedState(viewEnd)

    fun timeAt(x: Float, width: Int): Long {
        val fraction = (x / width).coerceIn(0f, 1f)
        return currentStart + ((currentEnd - currentStart) * fraction).toLong()
    }

    val fraction = ((cursorMs - viewStart).toFloat() / (viewEnd - viewStart)).coerceIn(0f, 1f)

    Canvas(
        Modifier
            .fillMaxWidth()
            .height(48.dp)
            .pointerInput(Unit) {
                detectTapGestures { currentOnScrub(timeAt(it.x, size.width)) }
            }
            .pointerInput(Unit) {
                detectHorizontalDragGestures(
                    onDragStart = { currentOnScrub(timeAt(it.x, size.width)) },
                ) { change, _ ->
                    change.consume()
                    currentOnScrub(timeAt(change.position.x, size.width))
                }
            }
            .semantics {
                contentDescription = "Хронология дня"
                stateDescription = clockFormat.format(Date(cursorMs))
                progressBarRangeInfo = ProgressBarRangeInfo(fraction, 0f..1f)
                setProgress { value ->
                    currentOnScrub(viewStart + ((viewEnd - viewStart) * value).toLong())
                    true
                }
            }
    ) {
        val span = (viewEnd - viewStart).toFloat()
        val stripTop = 8.dp.toPx()
        val stripHeight = 22.dp.toPx()
        val strip = RoundRect(0f, stripTop, size.width, stripTop + stripHeight, CornerRadius(6.dp.toPx()))

        clipPath(Path().apply { addRoundRect(strip) }) {
            drawRect(trackColor, Offset(0f, stripTop), Size(size.width, stripHeight))
            val minWidth = 1.dp.toPx()
            intervals.forEach { interval ->
                if (interval.endMs < viewStart || interval.startMs > viewEnd) return@forEach
                val x0 = (interval.startMs - viewStart) / span * size.width
                val x1 = (interval.endMs - viewStart) / span * size.width
                drawRect(
                    ActivityPalette.forActivity(interval.activity),
                    Offset(x0, stripTop),
                    Size(maxOf(x1 - x0, minWidth), stripHeight),
                )
            }
        }

        val x = fraction * size.width
        drawLine(handleColor, Offset(x, 2.dp.toPx()), Offset(x, 36.dp.toPx()), strokeWidth = 2.dp.toPx())
        val knob = Offset(x, 40.dp.toPx())
        drawCircle(handleColor, radius = 7.dp.toPx(), center = knob)
        drawCircle(ringColor, radius = 7.dp.toPx(), center = knob, style = Stroke(2.dp.toPx()))
    }
}

/** Решение в выбранный момент — то же окно, что записано в покадровый журнал. */
@Composable
private fun MomentDetails(window: ActivityWindowEntity) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        val source = InferenceSource.entries.firstOrNull { it.name == window.source }?.title ?: window.source
        StatRow("Уверенность", "${percent(window.confidence)} · $source")
        runnerUp(window)?.let { (type, probability) ->
            StatRow("Второй вариант", "${type.title} ${percent(probability)}")
        }
        PhonePlacement.fromName(window.placement)?.let {
            StatRow("Положение", "${it.emoji} ${it.title} ${percent(window.placementConfidence)}")
        }
        StatRow("Скорость", speedText(window))
        StatRow(
            "Темп",
            if (window.cadenceHz > 0f) "${(window.cadenceHz * 60).roundToInt()} шаг/мин" else "шагов нет",
        )
        if (window.googleActivity.isNotEmpty()) {
            val google = GoogleActivityType.fromName(window.googleActivity)
            val mark = when (window.googleAgrees) {
                1 -> " ✓"
                0 -> " ✗"
                else -> ""
            }
            StatRow("Google", "${google?.title ?: window.googleActivity} ${window.googleConfidence} %$mark")
        }
    }
}

private fun percent(value: Float): String = "${(value * 100).roundToInt()} %"

/**
 * Самый вероятный класс после выбранного — насколько решение было спорным.
 * Вероятности лежат в порядке [ActivityType]; записи старых версий с другим
 * набором классов пропускаются, иначе индексы указали бы не на те классы.
 */
private fun runnerUp(window: ActivityWindowEntity): Pair<ActivityType, Float>? {
    val probabilities = try {
        JSONArray(window.probabilities)
    } catch (_: Exception) {
        return null
    }
    if (probabilities.length() != ActivityType.COUNT) return null
    return ActivityType.entries
        .filter { it.name != window.activity }
        .map { it to probabilities.optDouble(it.id).toFloat() }
        .maxByOrNull { it.second }
}

private fun speedText(window: ActivityWindowEntity): String = when {
    window.speedReliable -> "${formatFloat(window.speedMs * 3.6f, 1)} км/ч"
    window.stepSpeedMs > 0f -> "${formatFloat(window.stepSpeedMs * 3.6f, 1)} км/ч по шагам"
    else -> "—"
}
