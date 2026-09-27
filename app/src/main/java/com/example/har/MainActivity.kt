package com.example.har

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.DirectionsRun
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.automirrored.filled.ListAlt
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.har.data.CsvExporter
import com.example.har.ui.HarViewModel
import com.example.har.ui.screens.AboutScreen
import com.example.har.ui.screens.CollectScreen
import com.example.har.ui.screens.JournalScreen
import com.example.har.ui.screens.LiveScreen
import com.example.har.ui.theme.ActivityRecognizerTheme

/** Вкладки приложения. */
private enum class Tab(val title: String, val icon: ImageVector) {
    LIVE("Сейчас", Icons.AutoMirrored.Filled.DirectionsRun),
    JOURNAL("Журнал", Icons.AutoMirrored.Filled.ListAlt),
    COLLECT("Сбор данных", Icons.Default.FiberManualRecord),
    ABOUT("О модели", Icons.Default.Info),
}

class MainActivity : ComponentActivity() {

    /**
     * Разрешение на уведомления нужно не ради самих уведомлений: без него
     * на Android 13+ нельзя показать уведомление переднего сервиса, а без
     * него система не даст сервису работать при погашенном экране.
     */
    private val notificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* отказ не блокирует работу, просто сокращает время фоновой работы */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestNotificationPermissionIfNeeded()

        setContent {
            ActivityRecognizerTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    HarApp()
                }
            }
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.POST_NOTIFICATIONS,
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
}

@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
private fun HarApp(vm: HarViewModel = viewModel()) {
    var tab by remember { mutableStateOf(Tab.LIVE) }
    val snackbar = remember { SnackbarHostState() }
    val context = LocalContext.current
    val message by vm.message.collectAsStateWithLifecycle()
    val engineState by vm.engineState.collectAsStateWithLifecycle()

    // Сообщения о выгрузке предлагают сразу поделиться файлом: иначе
    // пользователю пришлось бы искать его во внутренней памяти вручную.
    LaunchedEffect(message) {
        val m = message ?: return@LaunchedEffect
        val result = snackbar.showSnackbar(
            message = m.text,
            actionLabel = if (m.shareFile != null) "Поделиться" else null,
            withDismissAction = true,
        )
        if (result == SnackbarResult.ActionPerformed && m.shareFile != null) {
            context.startActivity(CsvExporter.shareIntent(context, m.shareFile, m.shareTitle))
        }
        vm.consumeMessage()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(tab.title) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant,
                ),
                actions = {
                    if (engineState.running) {
                        Text(
                            text = "● сбор идёт  ",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                },
            )
        },
        bottomBar = {
            NavigationBar {
                Tab.entries.forEach { entry ->
                    NavigationBarItem(
                        selected = tab == entry,
                        onClick = { tab = entry },
                        icon = { Icon(entry.icon, contentDescription = entry.title) },
                        label = { Text(entry.title) },
                    )
                }
            }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Box(Modifier.padding(padding)) {
            when (tab) {
                Tab.LIVE -> LiveScreen(vm)
                Tab.JOURNAL -> JournalScreen(vm)
                Tab.COLLECT -> CollectScreen(vm)
                Tab.ABOUT -> AboutScreen(vm)
            }
        }
    }
}
