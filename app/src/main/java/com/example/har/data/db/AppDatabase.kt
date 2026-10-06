package com.example.har.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [ActivityWindowEntity::class, ActivityIntervalEntity::class],
    version = 5,
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
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5)
                .build()

        /** Версия 2: скорость по инерциальной навигации в журнале окон. Старые записи получают 0. */
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN speed_ms REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN speed_reliable INTEGER NOT NULL DEFAULT 0")
            }
        }

        /**
         * Версия 3: промежуточные величины движения — земные оси, шаги, магнитное поле.
         * Старые записи получают нули: пересчитать их не из чего, сырые кадры не хранятся.
         */
        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN vertical_acc_rms REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN horizontal_acc_rms REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN jerk_rms REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN tilt_swing_deg REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN yaw_rate_mean REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN steps INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN cadence_hz REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN step_speed_ms REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN step_regularity REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN mag_inclination_deg REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN mag_disturbed_ratio REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN mag_gyro_mismatch_deg REAL NOT NULL DEFAULT 0")
            }
        }

        /**
         * Версия 4: всё, что нужно для разбора решения о положении, — сила тяжести
         * по осям телефона, поза «у уха», вероятности положения после сглаживания
         * и сырые выходы обеих моделей до поправок по правилам. По ним видно,
         * ошиблась модель или правила. Старые записи получают нули и пустые строки.
         */
        private val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                for (column in listOf("grav_x", "grav_y", "grav_z", "orientation_std", "ear_pose")) {
                    db.execSQL("ALTER TABLE activity_windows ADD COLUMN $column REAL NOT NULL DEFAULT 0")
                }
                for (column in listOf(
                    "placement_probabilities", "model_placement_probabilities", "model_activity_probabilities",
                )) {
                    db.execSQL("ALTER TABLE activity_windows ADD COLUMN $column TEXT NOT NULL DEFAULT ''")
                }
            }
        }

        /** Версия 5: решение Google Activity Recognition рядом с нашим — для сверки. */
        private val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN google_activity TEXT NOT NULL DEFAULT ''")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN google_confidence INTEGER NOT NULL DEFAULT -1")
                db.execSQL("ALTER TABLE activity_windows ADD COLUMN google_agrees INTEGER NOT NULL DEFAULT -1")
            }
        }
    }
}
