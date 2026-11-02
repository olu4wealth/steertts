#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Builds the SteerTTS Android APK (arm64) from this repository.

.DESCRIPTION
    Everything the build needs is discovered or passed in explicitly; nothing is hardcoded to one
    machine. The script:

      1. checks the model package is present (c/models/steer-tts-edge by default),
      2. writes android/local.properties (sdk.dir, cmake.dir) - it is machine-specific, not committed,
      3. runs Gradle with the JDK, NDK, CMake and signing key it was given,
      4. copies the APK to dist/.

    Run it from an ordinary PowerShell window (outside the DSH sandbox): the NDK build drives ninja,
    which spawns the C compiler, and a restricted sandbox blocks that.

.PARAMETER Configuration
    release (default) or debug. A debug build installs as ai.whissle.steertts.debug, side by side.

.EXAMPLE
    ./tools/build-apk.ps1
    ./tools/build-apk.ps1 -ModelDir D:\models\steer-tts-edge -Configuration debug
#>
[CmdletBinding()]
param(
    [ValidateSet('release', 'debug')] [string]$Configuration = 'release',

    # Model package (config.json, acoustic.safetensors, vocoder.safetensors, voices.*, native/front/).
    [string]$ModelDir,

    # Toolchain. Left empty they are discovered under ../toolchain (next to the repository, because
    # the JDK/NDK/CMake/key are hundreds of MB and have nothing to do with the source). They are
    # resolved in the body, not here: $PSScriptRoot is empty while param() defaults are evaluated
    # under Windows PowerShell 5.1 (which is what build-apk.cmd uses).
    [string]$Jdk,
    [string]$Ndk,
    [string]$Cmake,
    [string]$Sdk,

    # Signing. The key must stay the same for updates to an installed app to be accepted.
    # Passwords are never stored in this repository: they come from the environment
    # (STEERTTS_STORE_PASSWORD / STEERTTS_KEY_PASSWORD) or from a signing.properties file kept
    # next to the keystore, outside the repo.
    [string]$Keystore,
    [string]$KeyAlias,
    [string]$StorePassword,
    [string]$KeyPassword,
    [string]$SigningFile
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$android = Join-Path $repo 'android'
$toolchain = Join-Path (Split-Path $repo -Parent) 'toolchain'

function Resolve-OrThrow([string]$Path, [string]$What) {
    if (-not (Test-Path $Path)) { throw "$What not found at '$Path'. Pass it explicitly, e.g. -$What <dir>." }
    return (Resolve-Path $Path).Path
}

# ---------------------------------------------------------------- toolchain defaults
if (-not $Jdk) {
    $candidate = Get-ChildItem (Join-Path $toolchain 'jdk') -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin/java.exe') } | Select-Object -First 1
    if ($candidate) { $Jdk = $candidate.FullName }
}
if (-not $Ndk) { $Ndk = Join-Path $toolchain 'ndk27' }
if (-not $Cmake) { $Cmake = Join-Path $toolchain 'cmake' }
if (-not $Keystore) { $Keystore = Join-Path $toolchain 'steertts-release.keystore' }

# ---------------------------------------------------------------- signing secrets
# The keystore lives outside the repository and so does its password: read it from the environment
# first (a CI-friendly one-liner), else from a properties file beside the key. Nothing secret is
# ever written into a tracked file, so this repository can be public.
if (-not $SigningFile) { $SigningFile = Join-Path $toolchain 'signing.properties' }

function Get-SigningValue([string]$Name) {
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if ($fromEnv) { return $fromEnv }
    if (Test-Path $SigningFile) {
        foreach ($line in Get-Content $SigningFile) {
            $trimmed = $line.Trim()
            if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith(';')) { continue }
            $eq = $trimmed.IndexOf('=')
            if ($eq -gt 0 -and $trimmed.Substring(0, $eq).Trim() -eq $Name) {
                return $trimmed.Substring($eq + 1).Trim()
            }
        }
    }
    return $null
}

if (-not $StorePassword) { $StorePassword = Get-SigningValue 'STEERTTS_STORE_PASSWORD' }
if (-not $KeyPassword) { $KeyPassword = Get-SigningValue 'STEERTTS_KEY_PASSWORD' }
if (-not $KeyAlias) { $KeyAlias = Get-SigningValue 'STEERTTS_KEY_ALIAS' }

# ---------------------------------------------------------------- inputs
if (-not $ModelDir) { $ModelDir = Join-Path $repo 'c/models/steer-tts-edge' }
# Make it absolute: Gradle resolves a relative -PsteerttsModel against the *project* directory, not
# this shell's, so a relative path here would silently point somewhere else.
if (Test-Path $ModelDir) { $ModelDir = (Resolve-Path $ModelDir).Path }
if (-not (Test-Path (Join-Path $ModelDir 'config.json'))) {
    throw @"
No model package at '$ModelDir'.

The weights are licence-gated (CC BY-NC 4.0). Accept the licence at
https://huggingface.co/WhissleAI/steer-tts-edge , create a READ token at
https://huggingface.co/settings/tokens , then run:

    `$env:HF_TOKEN = 'hf_...'
    python tools/fetch_model.py

Or pass -ModelDir <dir> pointing at an existing package.
"@
}

if (-not $Sdk) { $Sdk = Join-Path $env:LOCALAPPDATA 'Android/Sdk' }
$Sdk = Resolve-OrThrow $Sdk 'Sdk'
$Jdk = Resolve-OrThrow (Join-Path $Jdk 'bin/java.exe') 'Jdk' | Split-Path | Split-Path
$Ndk = Resolve-OrThrow $Ndk 'Ndk'
$Cmake = Resolve-OrThrow (Join-Path $Cmake 'bin/cmake.exe') 'Cmake' | Split-Path | Split-Path

# ---------------------------------------------------------------- environment
$env:JAVA_HOME = $Jdk
$env:ANDROID_HOME = $Sdk
$env:ANDROID_SDK_ROOT = $Sdk
# AGP writes scratch files through java.io.tmpdir; keep them with the toolchain so a locked-down
# %TEMP% cannot break the build and the repository stays source-only.
$tmp = Join-Path $toolchain 'tmp'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TMP = $tmp; $env:TEMP = $tmp
# Gradle's download cache is ~500 MB and must not land inside the repository: keep it with the
# toolchain, next to the JDK/NDK, unless the caller already points somewhere else.
if (-not $env:GRADLE_USER_HOME) {
    $env:GRADLE_USER_HOME = Join-Path $toolchain 'gradle-home'
}

# local.properties is machine-specific (gitignored); regenerate it every run.
@"
sdk.dir=$($Sdk.Replace('\', '/'))
cmake.dir=$($Cmake.Replace('\', '/'))
"@ | Set-Content -Encoding ASCII (Join-Path $android 'local.properties')

# ---------------------------------------------------------------- build
$task = if ($Configuration -eq 'release') { 'assembleRelease' } else { 'assembleDebug' }
$gradleArgs = @(
    '--no-daemon', '--console=plain', $task,
    "-PsteerttsModel=$ModelDir",
    "-PsteerttsNdk=$Ndk",
    "-Dorg.gradle.jvmargs=-Xmx2g -Dfile.encoding=UTF-8 -Djava.io.tmpdir=$($tmp.Replace('\', '/'))"
)
if ($Configuration -eq 'release') {
    $Keystore = Resolve-OrThrow $Keystore 'Keystore'
    if (-not $StorePassword -or -not $KeyPassword) {
        throw @"
The signing password is not set, and it is deliberately not stored in this repository.

Export it for this shell:

    `$env:STEERTTS_STORE_PASSWORD = '...'
    `$env:STEERTTS_KEY_PASSWORD   = '...'

or put it in $SigningFile (outside the repository):

    STEERTTS_STORE_PASSWORD=...
    STEERTTS_KEY_PASSWORD=...
    STEERTTS_KEY_ALIAS=...

Or build -Configuration debug, which needs no key.
"@
    }
    if (-not $KeyAlias) { throw "Key alias unknown: pass -KeyAlias, or set STEERTTS_KEY_ALIAS." }

    # Handed to Gradle as project properties through the environment rather than on the command
    # line, so the passwords never appear in a process listing or a build log.
    $env:ORG_GRADLE_PROJECT_STEERTTS_STORE_FILE = $Keystore.Replace('\', '/')
    $env:ORG_GRADLE_PROJECT_STEERTTS_STORE_PASSWORD = $StorePassword
    $env:ORG_GRADLE_PROJECT_STEERTTS_KEY_ALIAS = $KeyAlias
    $env:ORG_GRADLE_PROJECT_STEERTTS_KEY_PASSWORD = $KeyPassword
}

Write-Host "Building $Configuration APK" -ForegroundColor Cyan
Write-Host "  model  $ModelDir"
Write-Host "  jdk    $Jdk"
Write-Host "  ndk    $Ndk"
Write-Host "  cmake  $Cmake"
Write-Host "  sdk    $Sdk"

Push-Location $android
try {
    # Gradle writes SDK/AGP warnings to stderr. With $ErrorActionPreference = 'Stop' PowerShell
    # turns the first of those into a terminating NativeCommandError, so relax it just for the
    # call and decide on the exit code instead.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & (Join-Path $android 'gradlew.bat') @gradleArgs
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "Gradle failed with exit code $code" }
} finally {
    Pop-Location
}

# ---------------------------------------------------------------- deliver
$built = Get-ChildItem (Join-Path $android "app/build/outputs/apk/$Configuration") -Filter '*.apk' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $built) { throw "No APK under app/build/outputs/apk/$Configuration" }

$dist = Join-Path $repo 'dist'
New-Item -ItemType Directory -Force -Path $dist | Out-Null
$name = "SteerTTS-$Configuration-$((Get-Date).ToString('yyyyMMdd')).apk"
$target = Join-Path $dist $name
Copy-Item $built.FullName $target -Force

$mb = [math]::Round((Get-Item $target).Length / 1MB, 1)
Write-Host ""
Write-Host "APK: $target ($mb MB)" -ForegroundColor Green
Write-Host "Install: copy it to the phone and open it (allow install from this source), then"
Write-Host "         Settings > Accessibility > Text-to-speech output > Preferred engine > SteerTTS."
