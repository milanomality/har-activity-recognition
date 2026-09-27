package com.example.har.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Тонкая обёртка над интерпретатором TensorFlow Lite.
 *
 * Поддерживает две формы модели:
 *  * один вход — вектор признаков (классификатор положения телефона);
 *  * два входа — окно сигналов и вектор контекстных признаков
 *    (классификатор активности).
 *
 * Буферы входа и выхода выделяются один раз в конструкторе: инференс идёт
 * раз в 1.28 с непрерывно, и пересоздание буферов на каждом окне давало бы
 * лишнее давление на сборщик мусора в фоновом сервисе.
 */
class TfLiteModel private constructor(
    private val interpreter: Interpreter,
    private val bindings: List<Binding>,
    private val outputSize: Int,
) : Closeable {

    /**
     * Описание одного входа модели.
     *
     * Входы различаются по рангу тензора, а не по порядку: конвертер TFLite
     * не обязан сохранять порядок входов функциональной модели Keras,
     * и полагаться на него — верный способ однажды подать окно сигналов
     * туда, где ждут вектор признаков.
     */
    private class Binding(val tensorIndex: Int, val rank: Int, val elementCount: Int) {
        val buffer: ByteBuffer =
            ByteBuffer.allocateDirect(elementCount * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())

        fun fill(values: FloatArray) {
            require(values.size == elementCount) {
                "Вход ожидает $elementCount значений, передано ${values.size}"
            }
            buffer.rewind()
            for (v in values) buffer.putFloat(v)
            buffer.rewind()
        }
    }

    private val outputBuffer: Array<FloatArray> = arrayOf(FloatArray(outputSize))

    /** Вход-последовательность (ранг 3): окно сигналов. */
    private val sequenceBinding = bindings.firstOrNull { it.rank == 3 }

    /** Вход-вектор (ранг 2): агрегированные признаки. */
    private val vectorBinding = bindings.firstOrNull { it.rank == 2 }

    val outputClasses: Int get() = outputSize
    val inputCount: Int get() = bindings.size

    /** true, если модель принимает окно сигналов и вектор контекста. */
    val usesContext: Boolean get() = sequenceBinding != null && vectorBinding != null

    /** Число элементов во входе-векторе, или 0, если такого входа нет. */
    val vectorSize: Int get() = vectorBinding?.elementCount ?: 0

    /** Число элементов во входе-последовательности, или 0. */
    val sequenceSize: Int get() = sequenceBinding?.elementCount ?: 0

    /**
     * Прогон одновходовой модели.
     *
     * @return копия вектора вероятностей (softmax уже применён внутри модели)
     */
    @Synchronized
    fun predict(input: FloatArray): FloatArray {
        check(bindings.size == 1) {
            "Модель принимает ${bindings.size} входов, вызван одновходовой predict"
        }
        bindings[0].fill(input)
        interpreter.run(bindings[0].buffer, outputBuffer)
        return outputBuffer[0].copyOf()
    }

    /**
     * Прогон двухвходовой модели.
     *
     * @param sequence окно сигналов, развёрнутое в порядке «канал последним»
     * @param context вектор агрегированных признаков
     */
    @Synchronized
    fun predict(sequence: FloatArray, context: FloatArray): FloatArray {
        val seq = checkNotNull(sequenceBinding) { "У модели нет входа-последовательности" }
        val vec = checkNotNull(vectorBinding) { "У модели нет входа-вектора" }

        seq.fill(sequence)
        vec.fill(context)

        // Массив входов должен идти в порядке индексов тензоров модели,
        // а не в порядке наших аргументов.
        val ordered = bindings.sortedBy { it.tensorIndex }.map { it.buffer }.toTypedArray()
        interpreter.runForMultipleInputsOutputs(ordered, mapOf(0 to outputBuffer))
        return outputBuffer[0].copyOf()
    }

    override fun close() = interpreter.close()

    companion object {
        private const val TAG = "TfLiteModel"

        /**
         * Загружает модель из assets. Возвращает null, если файла нет —
         * это штатная ситуация до первого запуска `ml/train.py`,
         * в которой приложение переходит на эвристический классификатор.
         */
        fun fromAsset(context: Context, assetName: String): TfLiteModel? = try {
            val buffer = mapAsset(context, assetName)
            val options = Interpreter.Options().apply {
                // Два потока: больше не даёт выигрыша на таких маленьких сетях,
                // но заметно греет телефон при непрерывной работе.
                numThreads = 2
                setUseXNNPACK(true)
            }
            val interpreter = Interpreter(buffer, options)

            val bindings = (0 until interpreter.inputTensorCount).map { i ->
                val shape = interpreter.getInputTensor(i).shape()
                // Первая размерность — размер батча, он всегда 1.
                Binding(
                    tensorIndex = i,
                    rank = shape.size,
                    elementCount = shape.drop(1).fold(1) { acc, d -> acc * d },
                )
            }
            val outputSize = interpreter.getOutputTensor(0).shape().last()

            val shapes = (0 until interpreter.inputTensorCount).joinToString(", ") {
                interpreter.getInputTensor(it).shape().joinToString("x")
            }
            Log.i(TAG, "Загружена $assetName: входы [$shapes], классов $outputSize")
            TfLiteModel(interpreter, bindings, outputSize)
        } catch (e: FileNotFoundException) {
            // Отсутствие модели — штатное состояние, а не сбой: приложение
            // работает на эвристике, пока ml/train.py не запущен.
            // Стек вызовов тут только засоряет лог.
            Log.i(TAG, "Модель $assetName не найдена в assets, используется эвристика")
            null
        } catch (e: Exception) {
            Log.w(TAG, "Не удалось загрузить модель $assetName", e)
            null
        }

        /**
         * Модель отображается в память, а не читается в кучу: файл лежит
         * в APK несжатым (см. noCompress в app/build.gradle.kts), поэтому
         * интерпретатор работает прямо по странице файла.
         */
        private fun mapAsset(context: Context, assetName: String): MappedByteBuffer {
            context.assets.openFd(assetName).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    return stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength,
                    )
                }
            }
        }
    }
}
