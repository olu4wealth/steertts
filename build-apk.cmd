@echo off
REM Windows launcher for tools\build-apk.ps1.
REM PowerShell's default execution policy (RemoteSigned) refuses unsigned local scripts, so the
REM policy is bypassed for this one process only. Any arguments are passed straight through:
REM
REM     build-apk.cmd
REM     build-apk.cmd -Configuration debug
REM     build-apk.cmd -ModelDir D:\models\steer-tts-edge
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build-apk.ps1" %*
exit /b %ERRORLEVEL%
