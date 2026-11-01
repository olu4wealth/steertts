plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.steertts"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.steertts"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // The model is read with AssetManager.open() into a direct ByteBuffer, so it does NOT need
    // noCompress. (That flag is only required for assets.openFd(), which this code avoids.)
    testOptions {
        unitTests.isReturnDefaultValues = true
        // Robolectric needs the merged resources/assets to run the real Service lifecycle.
        unitTests.isIncludeAndroidResources = true
    }
}

dependencies {
    // Pinned on purpose: "latest" is not a valid Gradle version literal.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
    testImplementation("junit:junit:4.13.2")
    // Runs android.speech.tts.TextToSpeechService lifecycle on the JVM, which is the only way to
    // check the engine's hooks without a device.
    testImplementation("org.robolectric:robolectric:4.14.1")
    // JVM tests link the desktop runtime, which ships linux-x64 natives inside the jar, so a real
    // graph can be executed without a device. The APK still packages the Android AAR.
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.31.1")
}

// The Android AAR carries no host natives and its ai.onnxruntime classes would shadow the
// desktop jar's, so drop it from every unit-test runtime classpath.
configurations
    .matching { it.name.startsWith("test") && it.name.endsWith("RuntimeClasspath") }
    .configureEach {
        exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime-android")
    }
