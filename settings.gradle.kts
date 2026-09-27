pluginManagement {
    repositories {
        // Фильтр по группам: в google() ищутся только андроидные артефакты,
        // остальное сразу уходит в mavenCentral. Это заметно ускоряет
        // разрешение зависимостей и убирает лишние сетевые запросы.
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "ActivityRecognizer"
include(":app")
