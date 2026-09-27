package com.example.har.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface ActivityWindowDao {

    @Insert
    suspend fun insert(window: ActivityWindowEntity): Long

    @Query("SELECT * FROM activity_windows ORDER BY startMs DESC LIMIT :limit")
    fun recent(limit: Int): Flow<List<ActivityWindowEntity>>

    @Query("SELECT * FROM activity_windows WHERE startMs >= :fromMs AND startMs < :toMs ORDER BY startMs ASC")
    suspend fun between(fromMs: Long, toMs: Long): List<ActivityWindowEntity>

    @Query("SELECT * FROM activity_windows ORDER BY startMs ASC")
    suspend fun all(): List<ActivityWindowEntity>

    @Query("SELECT COUNT(*) FROM activity_windows")
    fun countFlow(): Flow<Int>

    @Query("DELETE FROM activity_windows")
    suspend fun clear()

    /** Чистка журнала по возрасту — вызывается при старте сбора. */
    @Query("DELETE FROM activity_windows WHERE startMs < :beforeMs")
    suspend fun deleteOlderThan(beforeMs: Long): Int
}

@Dao
interface ActivityIntervalDao {

    @Insert
    suspend fun insert(interval: ActivityIntervalEntity): Long

    @Update
    suspend fun update(interval: ActivityIntervalEntity)

    /** Последний по времени интервал — кандидат на продление текущим окном. */
    @Query("SELECT * FROM activity_intervals ORDER BY endMs DESC LIMIT 1")
    suspend fun latest(): ActivityIntervalEntity?

    @Query("SELECT * FROM activity_intervals ORDER BY startMs DESC LIMIT :limit")
    fun recent(limit: Int): Flow<List<ActivityIntervalEntity>>

    @Query("SELECT * FROM activity_intervals WHERE endMs >= :fromMs AND startMs < :toMs ORDER BY startMs DESC")
    fun betweenFlow(fromMs: Long, toMs: Long): Flow<List<ActivityIntervalEntity>>

    @Query("SELECT * FROM activity_intervals ORDER BY startMs ASC")
    suspend fun all(): List<ActivityIntervalEntity>

    /**
     * Сводка по времени. Складываем длительности интервалов, а не считаем окна:
     * так пропуск нескольких окон (например, система придушила сервис)
     * не превращается в занижение суммарного времени.
     */
    @Query(
        """
        SELECT activity AS activity,
               SUM(endMs - startMs) AS totalMs,
               COUNT(*) AS intervalCount,
               CASE WHEN SUM(windowCount) > 0
                    THEN SUM(confidenceSum) / SUM(windowCount)
                    ELSE 0 END AS avgConfidence
        FROM activity_intervals
        WHERE endMs >= :fromMs AND startMs < :toMs
        GROUP BY activity
        ORDER BY totalMs DESC
        """
    )
    fun summaryBetween(fromMs: Long, toMs: Long): Flow<List<ActivitySummaryRow>>

    @Query("DELETE FROM activity_intervals")
    suspend fun clear()

    @Query("DELETE FROM activity_intervals WHERE endMs < :beforeMs")
    suspend fun deleteOlderThan(beforeMs: Long): Int
}
