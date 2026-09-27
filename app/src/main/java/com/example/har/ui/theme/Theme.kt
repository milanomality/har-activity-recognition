package com.example.har.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val LightColors = lightColorScheme(
    primary = Color(0xFF2E5D9F),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD6E3FF),
    onPrimaryContainer = Color(0xFF001B3E),
    secondary = Color(0xFF00696E),
    secondaryContainer = Color(0xFF9DF0F6),
    onSecondaryContainer = Color(0xFF002022),
    tertiary = Color(0xFF7A5900),
    surfaceVariant = Color(0xFFE0E2EC),
    onSurfaceVariant = Color(0xFF43474E),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFFA8C8FF),
    onPrimary = Color(0xFF002F65),
    primaryContainer = Color(0xFF13448D),
    onPrimaryContainer = Color(0xFFD6E3FF),
    secondary = Color(0xFF81D4DA),
    secondaryContainer = Color(0xFF004F53),
    onSecondaryContainer = Color(0xFF9DF0F6),
    tertiary = Color(0xFFF0C048),
    surfaceVariant = Color(0xFF43474E),
    onSurfaceVariant = Color(0xFFC3C6CF),
)

/** Цвета классов активности — единые для карточки, диаграммы и списка журнала. */
object ActivityPalette {
    val still = Color(0xFF8E9AAF)
    val walking = Color(0xFF3F7D20)
    val running = Color(0xFFD64933)
    val stairsUp = Color(0xFF2E5D9F)
    val stairsDown = Color(0xFF7B8CDE)
    val vehicle = Color(0xFF9C6644)
    val cycling = Color(0xFFCA8A04)

    fun forActivity(name: String): Color = when (name) {
        "STILL" -> still
        "WALKING" -> walking
        "RUNNING" -> running
        "STAIRS_UP" -> stairsUp
        "STAIRS_DOWN" -> stairsDown
        "VEHICLE" -> vehicle
        "CYCLING" -> cycling
        else -> still
    }
}

@Composable
fun ActivityRecognizerTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    // На Android 12+ берём системную палитру: приложение выглядит «своим»
    // на устройстве пользователя без отдельной настройки.
    val colors = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    MaterialTheme(colorScheme = colors, content = content)
}
