package com.example.har.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.har.sensors.SensorFrame
import kotlin.math.abs
import kotlin.math.max

/** Одна линия на графике: подпись, цвет и как достать значение из кадра. */
class ChartSeries(
    val label: String,
    val color: Color,
    val value: (SensorFrame) -> Float,
)

/** Вертикальные отметки событий — например, засчитанных шагов. */
class ChartMarkers(
    val color: Color,
    /** true — на этом кадре событие произошло. */
    val at: (previous: SensorFrame, current: SensorFrame) -> Boolean,
)

/**
 * Бегущий график сигнала за последние секунды.
 *
 * Рисуется на Canvas без сторонних библиотек: точек немного (500 кадров),
 * а внешняя зависимость ради одного графика — лишний вес. Шкала по Y
 * подбирается по данным, но не уже [minSpan] — иначе шум лежащего телефона
 * растягивался бы на всю высоту и выглядел как движение.
 *
 * Значения NaN (например, курс, когда он не определён) разрывают линию.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SignalChart(
    frames: List<SensorFrame>,
    series: List<ChartSeries>,
    unit: String,
    modifier: Modifier = Modifier,
    minSpan: Float = 1f,
    fixedRange: ClosedFloatingPointRange<Float>? = null,
    markers: ChartMarkers? = null,
    height: Dp = 120.dp,
    digits: Int = 2,
) {
    val gridColor = MaterialTheme.colorScheme.outlineVariant
    val zeroColor = MaterialTheme.colorScheme.outline
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant

    // Диапазон по Y — по всем линиям сразу, чтобы оси были сравнимы.
    var lo = Float.POSITIVE_INFINITY
    var hi = Float.NEGATIVE_INFINITY
    if (fixedRange != null) {
        lo = fixedRange.start
        hi = fixedRange.endInclusive
    } else {
        for (f in frames) for (s in series) {
            val v = s.value(f)
            if (v.isNaN()) continue
            if (v < lo) lo = v
            if (v > hi) hi = v
        }
        if (lo > hi) { lo = -minSpan / 2; hi = minSpan / 2 }
        val span = hi - lo
        if (span < minSpan) {
            val mid = (hi + lo) / 2
            lo = mid - minSpan / 2
            hi = mid + minSpan / 2
        }
        val pad = (hi - lo) * 0.08f
        lo -= pad
        hi += pad
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Box(Modifier.fillMaxWidth().height(height)) {
            Canvas(Modifier.fillMaxWidth().height(height)) {
                val w = size.width
                val h = size.height
                fun y(v: Float) = h - (v - lo) / (hi - lo) * h

                // Сетка: верх, середина, низ.
                for (frac in listOf(0f, 0.5f, 1f)) {
                    drawLine(gridColor, Offset(0f, h * frac), Offset(w, h * frac), strokeWidth = 1f)
                }
                if (lo < 0f && hi > 0f) {
                    drawLine(
                        zeroColor, Offset(0f, y(0f)), Offset(w, y(0f)), strokeWidth = 1.5f,
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 8f)),
                    )
                }
                if (frames.size < 2) return@Canvas
                val step = w / (frames.size - 1)

                markers?.let { m ->
                    for (i in 1 until frames.size) {
                        if (m.at(frames[i - 1], frames[i])) {
                            val x = i * step
                            drawLine(m.color, Offset(x, 0f), Offset(x, h), strokeWidth = 2f)
                        }
                    }
                }

                for (s in series) {
                    val path = Path()
                    var penDown = false
                    for (i in frames.indices) {
                        val v = s.value(frames[i])
                        if (v.isNaN()) { penDown = false; continue }
                        val x = i * step
                        val yy = y(v.coerceIn(lo, hi))
                        if (penDown) path.lineTo(x, yy) else path.moveTo(x, yy)
                        penDown = true
                    }
                    drawPath(
                        path, s.color,
                        style = Stroke(width = 3f, cap = StrokeCap.Round, join = StrokeJoin.Round),
                    )
                }
            }
            Text(
                formatFloat(hi, digits),
                style = MaterialTheme.typography.labelSmall,
                color = labelColor,
                modifier = Modifier.align(Alignment.TopStart),
            )
            Text(
                formatFloat(lo, digits),
                style = MaterialTheme.typography.labelSmall,
                color = labelColor,
                modifier = Modifier.align(Alignment.BottomStart),
            )
        }

        // Легенда с текущим значением каждой линии.
        val last = frames.lastOrNull()
        FlowRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            for (s in series) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(8.dp).background(s.color, CircleShape))
                    val v = last?.let(s.value)
                    Text(
                        text = " ${s.label} " + when {
                            v == null || v.isNaN() -> "—"
                            abs(v) >= 1000f -> formatFloat(v, 0)
                            else -> formatFloat(v, digits)
                        } + " $unit",
                        style = MaterialTheme.typography.labelMedium,
                        color = labelColor,
                    )
                }
            }
        }
    }
}

/** Цвета осей X / Y / Z и модуля — одинаковые на всех графиках. */
object AxisColors {
    val x = Color(0xFFE5484D)
    val y = Color(0xFF30A46C)
    val z = Color(0xFF3E63DD)
    val magnitude = Color(0xFF8E4EC6)
    val extra = Color(0xFFF76B15)
}

/** Максимум модуля — для симметричной шкалы вокруг нуля. */
fun symmetricRange(frames: List<SensorFrame>, minAbs: Float, vararg values: (SensorFrame) -> Float): ClosedFloatingPointRange<Float> {
    var m = minAbs
    for (f in frames) for (v in values) {
        val x = v(f)
        if (!x.isNaN()) m = max(m, abs(x))
    }
    return -m * 1.08f..m * 1.08f
}
