package com.example.har

import com.example.har.ml.ActivityType
import com.example.har.ml.PhonePlacement
import com.example.har.ml.PlacementConditioning
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PlacementConditioningTest {

    private fun placement(p: PhonePlacement) = FloatArray(PhonePlacement.COUNT).also { it[p.id] = 1f }

    private fun uniformActivity() = FloatArray(ActivityType.COUNT) { 1f / ActivityType.COUNT }

    @Test
    fun `в кармане положение решение модели не меняет`() {
        val model = floatArrayOf(0.1f, 0.3f, 0.2f, 0.1f, 0.1f, 0.2f)
        val out = PlacementConditioning.apply(model, placement(PhonePlacement.POCKET))
        assertArrayEquals(model, out, 1e-5f)
    }

    @Test
    fun `на столе телефон не ходит`() {
        // Модель уверена в ходьбе, но положение — «на столе».
        val model = FloatArray(ActivityType.COUNT).also {
            it[ActivityType.WALKING.id] = 0.9f; it[ActivityType.STILL.id] = 0.1f
        }
        val out = PlacementConditioning.apply(model, placement(PhonePlacement.ON_TABLE))
        assertTrue("покой ${out[ActivityType.STILL.id]}", out[ActivityType.STILL.id] > 0.75f)
    }

    @Test
    fun `у уха не бегут и не едут на велосипеде`() {
        val out = PlacementConditioning.apply(uniformActivity(), placement(PhonePlacement.AT_EAR))
        assertTrue(out[ActivityType.RUNNING.id] < out[ActivityType.WALKING.id] / 4f)
        assertTrue(out[ActivityType.CYCLING.id] < out[ActivityType.WALKING.id] / 10f)
    }

    @Test
    fun `в руке велосипед ослаблен, ходьба нет`() {
        val l = PlacementConditioning.likelihoods(placement(PhonePlacement.IN_HAND))
        assertEquals(1f, l[ActivityType.WALKING.id], 1e-5f)
        assertTrue(l[ActivityType.CYCLING.id] < 0.2f)
    }

    @Test
    fun `спорное положение действует пропорционально`() {
        val mixed = FloatArray(PhonePlacement.COUNT).also {
            it[PhonePlacement.POCKET.id] = 0.5f; it[PhonePlacement.ON_TABLE.id] = 0.5f
        }
        val l = PlacementConditioning.likelihoods(mixed)
        val expected = 0.5f * 1f + 0.5f * PlacementConditioning.compatibility(PhonePlacement.ON_TABLE, ActivityType.WALKING)
        assertEquals(expected, l[ActivityType.WALKING.id], 1e-5f)
    }

    @Test
    fun `ни один класс не запрещается полностью и итог нормирован`() {
        PhonePlacement.entries.forEach { p ->
            val l = PlacementConditioning.likelihoods(placement(p))
            assertTrue(l.all { it > 0f })
            val out = PlacementConditioning.apply(uniformActivity(), placement(p))
            assertEquals(1f, out.sum(), 1e-4f)
        }
    }
}
