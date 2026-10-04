package com.example.har.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.har.sensors.SensorFrame
import com.example.har.service.RecognitionEngine
import com.example.har.ui.HarViewModel

/**
 * Экран «Датчики»: бегущие графики всех используемых датчиков и величин,
 * которые из них вычисляются.
 *
 * Сверху — сырые показания (акселерометр, гироскоп, магнитометр, освещённость,
 * приближение), ниже — производные: движение в земных осях с отметками шагов,
 * ориентация и курс, скорость двумя способами. Так видно, из чего именно
 * получается каждое число на экране «Сейчас»: например, как при махе рукой
 * растут сырые оси, но почти не меняется вертикальное ускорение.
 */
@Composable
fun SensorsScreen(vm: HarViewModel, modifier: Modifier = Modifier) {
    val state by vm.engineState.collectAsStateWithLifecycle()
    val frames by vm.history.collectAsStateWithLifecycle()
    val availability = state.availability

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        if (!state.running || frames.isEmpty()) {
            item {
                SectionCard("Графики датчиков") {
                    Text(
                        text = if (!state.running) {
                            "Графики строятся во время распознавания. Запустите его на вкладке «Сейчас»."
                        } else {
                            "Ожидание первых показаний…"
                        },
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            return@LazyColumn
        }

        item {
            Text(
                "Последние ${RecognitionEngine.HISTORY_SECONDS} с, обновление 10 раз в секунду",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        // ---------------- Сырые показания ----------------

        item {
            ChartCard("Акселерометр", "С гравитацией: у лежащего телефона Z ≈ 9.8") {
                SignalChart(frames, xyz({ it.ax }, { it.ay }, { it.az }), "м/с²", minSpan = 2f)
            }
        }

        if (availability?.gyroscope != false) {
            item {
                ChartCard("Гироскоп", "Угловая скорость вокруг осей телефона") {
                    SignalChart(
                        frames, xyz({ it.gx }, { it.gy }, { it.gz }), "рад/с",
                        fixedRange = symmetricRange(frames, 0.5f, { it.gx }, { it.gy }, { it.gz }),
                    )
                }
            }
        }

        if (availability?.magnetometer != false) {
            item {
                ChartCard(
                    "Магнитометр",
                    "Модуль поля Земли 25–65 мкТл. Скачки модуля — металл или ток рядом",
                ) {
                    SignalChart(
                        frames,
                        xyz({ it.mx }, { it.my }, { it.mz }) +
                            ChartSeries("|B|", AxisColors.magnitude) { it.magNorm },
                        "мкТл", minSpan = 10f, digits = 1,
                    )
                }
            }
        }

        if (availability?.light == true || availability?.proximity == true) {
            item {
                ChartCard("Освещённость и приближение", "Освещённость в логарифмической шкале") {
                    if (availability.light) {
                        SignalChart(
                            frames,
                            listOf(ChartSeries("lg(1+лк)", AxisColors.extra) { f ->
                                if (f.lightLux < 0f) Float.NaN else kotlin.math.log10(1f + f.lightLux)
                            }),
                            "", fixedRange = 0f..4.7f, height = 80.dp,
                        )
                    }
                    if (availability.proximity) {
                        SignalChart(
                            frames,
                            listOf(ChartSeries("перекрыт", AxisColors.magnitude) { f ->
                                if (f.proximityNear) 1f else 0f
                            }),
                            "", fixedRange = -0.1f..1.1f, height = 50.dp, digits = 0,
                        )
                    }
                }
            }
        }

        // ---------------- Производные величины ----------------

        item {
            ChartCard(
                "Движение в земных осях",
                "Вертикаль — подпрыгивание тела при шаге, горизонталь — разгон, торможение, " +
                    "мах руки. Вертикальные черты — засчитанные шаги",
            ) {
                SignalChart(
                    frames,
                    listOf(
                        ChartSeries("верт.", AxisColors.z) { it.verticalAcc },
                        ChartSeries("гориз.", AxisColors.extra) { it.horizontalAcc },
                    ),
                    "м/с²",
                    fixedRange = symmetricRange(frames, 1f, { it.verticalAcc }, { it.horizontalAcc }),
                    markers = ChartMarkers(AxisColors.y.copy(alpha = 0.35f)) { prev, cur ->
                        cur.stepCount > prev.stepCount
                    },
                )
            }
        }

        item {
            ChartCard(
                "Ориентация",
                "Наклон: 0° — экраном вверх, 90° — вертикально. Поворот корпуса — " +
                    "угловая скорость вокруг вертикали Земли",
            ) {
                SignalChart(
                    frames,
                    listOf(ChartSeries("наклон", AxisColors.x) { it.insTiltDeg }),
                    "°", fixedRange = 0f..180f, height = 90.dp, digits = 0,
                )
                SignalChart(
                    frames,
                    listOf(ChartSeries("поворот", AxisColors.magnitude) { it.yawRate }),
                    "рад/с",
                    fixedRange = symmetricRange(frames, 0.5f, { it.yawRate }),
                    height = 90.dp,
                )
            }
        }

        if (availability?.magnetometer != false) {
            item {
                ChartCard(
                    "Компас",
                    "Курс оси телефона по магнитному полю. Разрыв линии — поле " +
                        "искажено или телефон смотрит вверх",
                ) {
                    SignalChart(
                        frames,
                        listOf(
                            ChartSeries("курс", AxisColors.z) { f ->
                                if (f.magTrusted) f.headingDeg else Float.NaN
                            },
                        ),
                        "°", fixedRange = 0f..360f, height = 90.dp, digits = 0,
                    )
                    SignalChart(
                        frames,
                        listOf(ChartSeries("наклонение", AxisColors.y) { f ->
                            if (f.magNorm > 0f) f.magInclinationDeg else Float.NaN
                        }),
                        "°", fixedRange = -90f..90f, height = 70.dp, digits = 0,
                    )
                }
            }
        }

        item {
            ChartCard(
                "Скорость",
                "Инерциальная навигация точна сразу после остановки; по шагам — " +
                    "пока человек идёт, без накопления ошибки",
            ) {
                SignalChart(
                    frames,
                    listOf(
                        ChartSeries("навигация", AxisColors.x) { it.speedMs * 3.6f },
                        ChartSeries("по шагам", AxisColors.y) { it.stepSpeedMs * 3.6f },
                    ),
                    "км/ч", minSpan = 5f, digits = 1,
                )
            }
        }
    }
}

@Composable
private fun ChartCard(title: String, caption: String, content: @Composable () -> Unit) {
    SectionCard(title) {
        Text(
            caption,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        content()
    }
}

private fun xyz(
    x: (SensorFrame) -> Float,
    y: (SensorFrame) -> Float,
    z: (SensorFrame) -> Float,
) = listOf(
    ChartSeries("X", AxisColors.x, x),
    ChartSeries("Y", AxisColors.y, y),
    ChartSeries("Z", AxisColors.z, z),
)
