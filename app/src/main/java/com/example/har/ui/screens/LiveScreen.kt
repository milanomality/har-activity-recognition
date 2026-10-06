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
import com.example.har.google.GoogleActivityTracker
import com.example.har.ml.ActivityType
import com.example.har.ml.MotionFusion
import com.example.har.ml.PlacementConditioning
import com.example.har.ml.PlacementFusion
import com.example.har.motion.InertialSpeedEstimator
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
                        StatRow(
                            "Тяжесть по осям X / Y",
                            "${formatFloat(s.gravityXRatio, 2)} / ${formatFloat(s.gravityYRatio, 2)}",
                        )
                        StatRow("Поза «у уха»", "${formatFloat(PlacementFusion.earPose(s) * 100f, 0)} %")
                        StatRow("Стабильность ориентации", formatFloat(s.orientationStd, 4))
                        StatRow(
                            "Скорость (инерц. навигация)",
                            when {
                                s.secondsSinceZupt < 0f -> "—"
                                s.speedReliable -> "${formatFloat(s.speedMs * 3.6f, 1)} км/ч"
                                // Без остановок ошибка интегрирования растёт без предела:
                                // показываем число, но честно помечаем, что оно не используется.
                                s.rotationSinceZupt > InertialSpeedEstimator.MAX_RELIABLE_ROTATION_RAD ->
                                    "${formatFloat(s.speedMs * 3.6f, 1)} км/ч · ненадёжно, телефон вращался"
                                else -> "${formatFloat(s.speedMs * 3.6f, 1)} км/ч · ненадёжно, " +
                                    "${formatFloat(s.secondsSinceZupt, 0)} с без остановки"
                            },
                        )
                    }
                }
            }

            item {
                SectionCard("Движение в земных осях") {
                    val s = r.stats
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (!s.motionComputed) {
                            StatRow("Навигация", "ещё не запущена")
                        } else {
                            StatRow("Вертикальное ускорение (СКЗ)", "${formatFloat(s.verticalAccRms)} м/с²")
                            StatRow("Горизонтальное ускорение (СКЗ)", "${formatFloat(s.horizontalAccRms)} м/с²")
                            StatRow("Доля вертикали", "${formatFloat(s.verticalShare * 100f, 0)} %")
                            StatRow("Рывок (СКЗ)", "${formatFloat(s.jerkRms, 1)} м/с³")
                            StatRow("Размах наклона", "${formatFloat(s.tiltSwingDeg, 0)}°")
                            StatRow("Поворот корпуса", "${formatFloat(s.yawRateMean)} рад/с")
                        }
                    }
                }
            }

            item {
                SectionCard("Шаги") {
                    val s = r.stats
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        StatRow("Шагов в окне", s.stepsInWindow.toString())
                        StatRow(
                            "Темп",
                            if (s.cadenceHz > 0f) "${formatFloat(s.cadenceHz * 60f, 0)} шаг/мин" else "—",
                        )
                        StatRow("Регулярность шага", formatFloat(s.stepRegularity))
                        StatRow("Размах за шаг", "${formatFloat(s.stepAmplitude)} м/с²")
                        StatRow(
                            "Скорость по шагам",
                            if (s.hasGait) "${formatFloat(s.stepSpeedMs * 3.6f, 1)} км/ч" else "—",
                        )
                    }
                }
            }

            item {
                SectionCard("Магнитометр") {
                    val s = r.stats
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (s.magDisturbedRatio < 0f) {
                            StatRow("Магнитометр", "нет данных")
                        } else {
                            StatRow("Модуль поля", "${formatFloat(s.magMagMean, 1)} мкТл")
                            StatRow("Колебания модуля (СКО)", "${formatFloat(s.magMagStd)} мкТл")
                            StatRow("Наклонение", "${formatFloat(s.magInclinationDeg, 0)}°")
                            StatRow("Поле искажено", "${formatFloat(s.magDisturbedRatio * 100f, 0)} % окна")
                            StatRow("Компас против гироскопа", "${formatFloat(s.magGyroMismatchDeg, 0)}°")
                        }
                    }
                }
            }

            item {
                SectionCard("Сверка с Google") {
                    val google by GoogleActivityTracker.latest.collectAsStateWithLifecycle()
                    val googleError by GoogleActivityTracker.error.collectAsStateWithLifecycle()
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        val g = r.google
                        val last = google
                        when {
                            googleError != null -> StatRow("Google", googleError ?: "")
                            g != null -> {
                                StatRow("Google думает", "${g.type.title} · ${g.confidence} %")
                                StatRow(
                                    "Совпадает с нашим",
                                    when (g.type.agreesWith(r.prediction.activity)) {
                                        true -> "да"
                                        false -> "нет"
                                        null -> "не с чем сравнить"
                                    },
                                )
                            }
                            last != null -> StatRow("Google", "ответ устарел (${last.type.title})")
                            else -> StatRow("Google", "ждём первый ответ")
                        }
                        if (state.googleCompared > 0) {
                            StatRow(
                                "Совпадений за сессию",
                                "${formatFloat(state.googleAgreed * 100f / state.googleCompared, 0)} % " +
                                    "из ${state.googleCompared} окон",
                            )
                        }
                        Text(
                            "Google Activity Recognition не знает положения телефона и не различает " +
                                "лестницу — она сравнивается с его «ходьбой». Ответы приходят раз в " +
                                "5–30 с и в наше решение не входят: это только эталон для сверки.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            item {
                SectionCard("Активность с учётом положения") {
                    val l = PlacementConditioning.likelihoods(r.prediction.placementProbabilities)
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        StatRow("Положение (этап 1)", "${r.prediction.placement.emoji} ${r.prediction.placement.title}")
                        ActivityType.entries.forEach { type ->
                            StatRow("${type.emoji} ${type.title}", "×${formatFloat(l[type.id], 2)}")
                        }
                        Text(
                            "Сначала определяется положение телефона, затем вероятность каждой " +
                                "активности умножается на её совместимость с этим положением: " +
                                "на столе телефон не несут, у уха не бегут, в руке не едут на велосипеде.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            item {
                SectionCard("Как учтены шаги и темп") {
                    val f = MotionFusion.factors(r.stats, r.prediction.placementProbabilities)
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        StatRow("Телефон в руке или у уха", "${formatFloat(f.handHeld * 100f, 0)} %")
                        StatRow("Шаговый ритм", "${formatFloat(f.gait * 100f, 0)} %")
                        StatRow("Движение рукой без шагов", "${formatFloat(f.handGesture * 100f, 0)} %")
                        StatRow("Темп мал для бега", "${formatFloat(f.tooSlowForRunning * 100f, 0)} %")
                        StatRow("Темп велик для ходьбы", "${formatFloat(f.tooFastForWalking * 100f, 0)} %")
                        StatRow("Неподвижность телефона", "${formatFloat(f.stillness * 100f, 0)} %")
                        StatRow("Сдвиг к покою (физ. запрет)", "${formatFloat(f.stillOverride * 100f, 0)} %")
                        Text(
                            "Движение рукой без шагов ослабляет ходьбу и бег; шаговый ритм ослабляет " +
                                "покой; темп разводит ходьбу и бег надёжнее амплитуды, " +
                                "которую в руке раздувает мах. Если телефон неподвижен и шагов нет, " +
                                "итог сдвигается к покою, как бы ни была уверена модель.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
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
