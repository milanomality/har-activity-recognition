package com.example.har.ml

import com.example.har.features.WindowStats

/**
 * Уточнение вероятностей активности по шагам и положению телефона.
 *
 * Работает так же, как [SpeedFusion]: вероятность каждого класса умножается
 * на множитель от [FLOOR] до 1, затем распределение нормируется. Решение
 * модели можно ослабить, но нельзя перечеркнуть.
 *
 * Главная задача — развести движение тела и движение телефона. Модель видит
 * амплитуду ускорения и вращения, но не знает, откуда она взялась: телефон
 * в руке, которым машут при разговоре, даёт те же числа, что и бег, а мах руки
 * при ходьбе раздувает амплитуду так, что ходьба похожа на бег. Здесь решение
 * опирается на то, что от руки не зависит:
 *  - **шаги** ищутся по вертикальному ускорению в земных осях, а их
 *    регулярность — по автокорреляции. Жест рукой не даёт повторяющегося
 *    вертикального рисунка; шаг даёт его в любом положении телефона;
 *  - **темп** шагов разделяет ходьбу и бег надёжнее амплитуды: мах руки
 *    меняет амплитуду, но не темп;
 *  - **положение телефона** решает, насколько верить амплитуде: в кармане
 *    она про тело, в руке — наполовину про руку.
 *
 * **Физический запрет.** Множители не ниже [FLOOR] не спасают, когда модель
 * уверенно ошибается вне своего опыта: телефон просто держат в руке почти
 * вертикально, а сеть, выучившая ориентацию телефона на велосипеде, отдаёт
 * велосипеду 0.999. Поэтому, если телефон почти неподвижен и шагов нет,
 * итоговое распределение смешивается с «покоем» с весом, равным степени
 * неподвижности: при полной неподвижности модель не может утверждать движение.
 * Тот же приём — для движения телефона рукой без шагов.
 */
object MotionFusion {

    const val FLOOR = 0.3f

    /** Ниже этого темпа «шаги» — не ходьба, а мах рукой или случайные толчки, Гц. */
    private const val MIN_GAIT_CADENCE_HZ = 1.2f

    /** Доля покоя, гарантированная при движении телефона рукой без шагов. */
    private const val GESTURE_OVERRIDE = 0.85f

    /** Промежуточные оценки — показываются на экране, чтобы решение было объяснимо. */
    data class Factors(
        /** Уверенность, что человек идёт или бежит: есть шаги и они регулярны, 0–1. */
        val gait: Float,
        /** Вероятность, что телефон в руке или у уха. */
        val handHeld: Float,
        /** Движение телефона рукой без шагов (жесты, перекладывание), 0–1. */
        val handGesture: Float,
        /** Темп слишком низок для бега, 0–1. */
        val tooSlowForRunning: Float,
        /** Темп слишком высок для ходьбы, 0–1. */
        val tooFastForWalking: Float,
        /** Телефон почти неподвижен: ни ускорения, ни вращения, 0–1. */
        val stillness: Float = 0f,
        /** С каким весом итог смешивается с «покоем» (физический запрет), 0–1. */
        val stillOverride: Float = 0f,
    )

    fun factors(s: WindowStats, placementProbabilities: FloatArray?): Factors {
        // Лежащий на столе телефон не качается. Если классификатор положения
        // говорит «на столе», а наклон за окно гуляет на градусы, телефон держат
        // в руке: почти вертикальный и неподвижный телефон в руке по свету
        // и приближению от стоящего на подставке не отличить.
        val onTableButSwinging = placementProbabilities?.let {
            it.getOrElse(PhonePlacement.ON_TABLE.id) { 0f } * gate(s.tiltSwingDeg - 2f, 4f)
        } ?: 0f
        val handHeld = (placementProbabilities?.let {
            it.getOrElse(PhonePlacement.IN_HAND.id) { 0f } + it.getOrElse(PhonePlacement.AT_EAR.id) { 0f }
        } ?: 0f).plus(onTableButSwinging).coerceAtMost(1f)

        // Неподвижность считается по модулям и от навигации не зависит:
        // 1 при СКО ускорения до 0.2 м/с² и вращении до 0.3 рад/с, 0 — от 0.6 и 0.7.
        val stillness = gate(0.6f - s.accMagStd, 0.4f) * gate(0.7f - s.gyroMagMean, 0.4f)

        if (!s.motionComputed) {
            return Factors(0f, handHeld, 0f, 0f, 0f, stillness, stillness)
        }

        // Шаговый ритм: хотя бы два шага в окне, повторяющийся рисунок вертикали
        // и человеческий темп. Мах рукой тоже даёт «шаги», но редкие — около
        // 0.8 в секунду, вдвое медленнее самой медленной ходьбы.
        val detectedGait = gate(s.stepsInWindow - 1f, 2f) *
            gate(s.stepRegularity - 0.25f, 0.3f) *
            gate(s.cadenceHz - MIN_GAIT_CADENCE_HZ, 0.3f)

        // Ритм, который задаёт рука: телефон в руке сильно вращается и качается,
        // а горизонталь не уступает вертикали. Ритмичный мах даёт «шаги» с темпом
        // ходьбы и регулярностью 0.9 — по одной вертикали его от ходьбы не отличить.
        // При настоящей ходьбе с телефоном в руке преобладает подпрыгивание тела.
        val handDriven = handHeld *
            gate(s.gyroMagMean - 0.8f, 0.6f) *
            gate(s.tiltSwingDeg - 25f, 20f) *
            (1f - gate(s.verticalShare - 0.4f, 0.2f))
        val gait = detectedGait * (1f - handDriven)

        // Телефон заметно движется (ускорение или вращение), а шагов нет.
        val moving = maxOf(gate(s.accMagStd - 0.4f, 0.8f), gate(s.gyroMagMean - 0.4f, 0.8f))
        // Жест рукой: телефон в руке, движется, вращается, но шагового ритма нет.
        val handGesture = handHeld * moving * (1f - gait)

        // Темп: ходьба 1.4–2.3 шага/с, бег от 2.4. На границе плавный переход.
        val tooSlowForRunning = if (s.cadenceHz > 0f) gate(2.4f - s.cadenceHz, 0.4f) else 0f
        val tooFastForWalking = if (s.cadenceHz > 0f) gate(s.cadenceHz - 2.4f, 0.4f) else 0f

        // Физический запрет: неподвижность без шагов или жест рукой без шагов.
        val stillOverride = maxOf(stillness * (1f - gait), GESTURE_OVERRIDE * handGesture)

        return Factors(
            gait, handHeld, handGesture, tooSlowForRunning, tooFastForWalking,
            stillness, stillOverride,
        )
    }

    /** Множители для каждого класса, в порядке [ActivityType]. */
    fun likelihoods(s: WindowStats, placementProbabilities: FloatArray?): FloatArray {
        val f = factors(s, placementProbabilities)
        val penalty = FloatArray(ActivityType.COUNT)
        val noGaitButMoving = if (s.motionComputed) {
            (1f - f.gait) * gate(s.accMagStd - 0.5f, 1.0f)
        } else {
            0f
        }

        // В кармане амплитуда — про тело, и правило темпа только уточняет модель.
        // В руке амплитуду раздувает рука, и темпу доверяем полностью.
        val cadenceWeight = 0.6f + 0.4f * f.handHeld

        penalty[ActivityType.STILL.id] = f.gait
        penalty[ActivityType.WALKING.id] = maxOf(noGaitButMoving, cadenceWeight * f.tooFastForWalking)
        penalty[ActivityType.RUNNING.id] = maxOf(noGaitButMoving, cadenceWeight * f.tooSlowForRunning)
        penalty[ActivityType.STAIRS_UP.id] = maxOf(noGaitButMoving, cadenceWeight * f.tooFastForWalking)
        penalty[ActivityType.STAIRS_DOWN.id] = maxOf(noGaitButMoving, cadenceWeight * f.tooFastForWalking)
        // На велосипеде руки на руле: телефон в руке с велосипедом несовместим.
        // Мах рукой при этом особенно похож на педалирование — периодическое
        // вращение около 1 Гц, как у бедра с телефоном в кармане.
        penalty[ActivityType.CYCLING.id] = maxOf(f.handGesture, f.handHeld)

        return FloatArray(ActivityType.COUNT) { 1f - (1f - FLOOR) * penalty[it].coerceIn(0f, 1f) }
    }

    fun apply(
        probabilities: FloatArray,
        stats: WindowStats,
        placementProbabilities: FloatArray?,
    ): FloatArray {
        val f = factors(stats, placementProbabilities)
        var out = probabilities
        if (stats.motionComputed) {
            val l = likelihoods(stats, placementProbabilities)
            val weighted = FloatArray(probabilities.size) { i -> probabilities[i] * l.getOrElse(i) { 1f } }
            val sum = weighted.sum()
            if (sum > 1e-9f) {
                for (i in weighted.indices) weighted[i] /= sum
                out = weighted
            }
        }
        // Смешивание, а не умножение: покой получает не меньше stillOverride,
        // сколь бы близкой к нулю ни была его вероятность у модели.
        val w = f.stillOverride.coerceIn(0f, 1f)
        if (w <= 0f) return out
        return FloatArray(out.size) { i ->
            (1f - w) * out[i] + if (i == ActivityType.STILL.id) w else 0f
        }
    }

    /** Мягкая ступенька: 0 при value <= 0, 1 при value >= width. */
    private fun gate(value: Float, width: Float): Float = (value / width).coerceIn(0f, 1f)
}
