package com.example.har.ui.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.util.Locale
import java.util.concurrent.TimeUnit

/** Карточка с заголовком — единый контейнер для всех блоков приложения. */
@Composable
fun SectionCard(
    title: String,
    modifier: Modifier = Modifier,
    trailing: @Composable (() -> Unit)? = null,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                trailing?.invoke()
            }
            content()
        }
    }
}

/**
 * Горизонтальная полоса вероятности класса.
 *
 * Показывается для всех классов сразу, а не только для победителя:
 * распределение делает видимым, насколько решение было близким,
 * и это главный инструмент отладки классификатора на живом устройстве.
 */
@Composable
fun ProbabilityBar(
    label: String,
    probability: Float,
    color: Color,
    highlighted: Boolean,
    modifier: Modifier = Modifier,
) {
    val animated by animateFloatAsState(
        targetValue = probability.coerceIn(0f, 1f),
        label = "probability",
    )
    val barColor by animateColorAsState(
        targetValue = if (highlighted) color else color.copy(alpha = 0.35f),
        label = "barColor",
    )

    Column(modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = if (highlighted) FontWeight.Bold else FontWeight.Normal,
            )
            Text(
                text = "${(animated * 100).toInt()} %",
                style = MaterialTheme.typography.bodyMedium,
                fontFamily = FontFamily.Monospace,
                fontWeight = if (highlighted) FontWeight.Bold else FontWeight.Normal,
            )
        }
        Box(
            Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f))
        ) {
            Box(
                Modifier
                    .fillMaxWidth(animated)
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(barColor)
            )
        }
    }
}

/**
 * Строка «параметр — значение» с моноширинным значением для ровной колонки.
 *
 * Метка занимает свободное место, значение прижато вправо, между ними
 * гарантированный зазор: без него длинное значение вплотную упирается
 * в метку и они читаются как одно слово.
 */
@Composable
fun StatRow(label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace,
            textAlign = TextAlign.End,
        )
    }
}

/** Цветная точка-индикатор состояния (датчик есть / нет, сервис работает / нет). */
@Composable
fun StatusDot(active: Boolean, modifier: Modifier = Modifier) {
    Box(
        modifier
            .width(10.dp)
            .height(10.dp)
            .clip(RoundedCornerShape(5.dp))
            .background(
                if (active) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.outline.copy(alpha = 0.5f)
            )
    )
}

/** Длительность в человекочитаемом виде: «1 ч 05 мин» или «42 с». */
fun formatDuration(millis: Long): String {
    if (millis < 0) return "—"
    val hours = TimeUnit.MILLISECONDS.toHours(millis)
    val minutes = TimeUnit.MILLISECONDS.toMinutes(millis) % 60
    val seconds = TimeUnit.MILLISECONDS.toSeconds(millis) % 60
    return when {
        hours > 0 -> String.format(Locale.getDefault(), "%d ч %02d мин", hours, minutes)
        minutes > 0 -> String.format(Locale.getDefault(), "%d мин %02d с", minutes, seconds)
        else -> "$seconds с"
    }
}

fun formatFloat(value: Float, digits: Int = 2): String =
    if (value.isNaN() || value.isInfinite()) "—"
    else String.format(Locale.US, "%.${digits}f", value)
