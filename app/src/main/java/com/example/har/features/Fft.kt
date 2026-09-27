package com.example.har.features

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Быстрое преобразование Фурье (Кули — Тьюки, основание 2, на месте).
 *
 * Нужно для частотных признаков: темп шагов у ходьбы лежит около 1.5–2.5 Гц,
 * у бега — 2.5–4 Гц, а у поездки в транспорте выраженного пика нет вовсе.
 * Именно это различие делает частотные признаки самыми информативными в HAR.
 */
object Fft {

    /**
     * Спектр амплитуд сигнала [input].
     *
     * @param input сигнал; длина будет усечена/дополнена нулями до степени двойки
     * @return массив длиной n/2 — амплитуды бинов от 0 Гц до частоты Найквиста
     */
    fun magnitudeSpectrum(input: FloatArray): FloatArray {
        val n = largestPowerOfTwo(input.size)
        if (n < 2) return FloatArray(0)

        val re = DoubleArray(n)
        val im = DoubleArray(n)

        // Окно Ханна подавляет растекание спектра из-за разрыва на краях окна.
        for (i in 0 until n) {
            val w = 0.5 * (1.0 - cos(2.0 * PI * i / (n - 1)))
            re[i] = input[i].toDouble() * w
        }

        transform(re, im)

        val half = n / 2
        val out = FloatArray(half)
        for (i in 0 until half) {
            out[i] = sqrt(re[i] * re[i] + im[i] * im[i]).toFloat() / half
        }
        return out
    }

    /** Частота бина [index] при длине БПФ [n] и частоте дискретизации [sampleRate]. */
    fun binToHz(index: Int, n: Int, sampleRate: Int): Float =
        index.toFloat() * sampleRate / n

    private fun largestPowerOfTwo(size: Int): Int {
        var p = 1
        while (p * 2 <= size) p *= 2
        return p
    }

    /** Итеративное БПФ с перестановкой по обратным битам. */
    private fun transform(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }

        var len = 2
        while (len <= n) {
            val angle = -2.0 * PI / len
            val wRe = cos(angle)
            val wIm = sin(angle)
            var i = 0
            while (i < n) {
                var curRe = 1.0
                var curIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
                    val vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
                    re[i + k] = uRe + vRe
                    im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe
                    im[i + k + len / 2] = uIm - vIm
                    val nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += len
            }
            len = len shl 1
        }
    }
}
