package com.example.har.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [ActivityWindowEntity::class, ActivityIntervalEntity::class],
    version = 2,
    exportSchema = true,
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun windowDao(): ActivityWindowDao
    abstract fun intervalDao(): ActivityIntervalDao

    companion object {
        private const val DB_NAME = "har.db"

        @Volatile
        private var instance: AppDatabase? = null

        fun get(context: Context): AppDatabase =
            instance ?: synchronized(this) {
                instance ?: build(context.applicationContext).also { instance = it }
            }

        private fun build(context: Context): AppDatabase =
            Room.databaseBuilder(context, AppDatabase::class.java, DB_NAME)
                // Журнал пишется каждые 1.28 с из фонового сервиса. WAL позволяет
                // UI читать журнал, пока сервис в него пишет, без взаимных блокировок.
                .setJournalMode(RoomDatabase.JournalMode.WRITE_AHEAD_LOGGING)
                .addMigrations(MIGRATION_1_2)
                .build()

        /** Версия 2: скорость по инерциальной навигации в журнале окон. Старые записи получают 0. */
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN speed_ms REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN speed_reliable INTEGER NOT NULL DEFAULT 0")
            }
        }
    }
}
