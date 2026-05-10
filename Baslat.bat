@echo off
title nASR Alt Yazi Otomasyonu - Portable Surum
color 0A

echo ========================================================
echo [SISTEM] nASR Tasinabilir (Portable) Surum Baslatiliyor...
echo ========================================================
echo.

:: FFMPEG'i bulabilmesi icin icinde bulundugumuz klasoru gecici olarak Windows'a tanitiyoruz
set PATH=%~dp0;%PATH%

:: Modellerin "C:\Users\Kullanici" yerine klasor icindeki "models" klasorune inmesi icin sihirli komut!
set HF_HOME=%~dp0models
set TORCH_HOME=%~dp0models

:: Kendi icimizdeki tasinabilir Python klasorunu isaret ediyoruz (pythonw ile siyah ekran acilmaz)
set PYTHON_EXE=%~dp0python\pythonw.exe

:: Python var mi diye kontrol edelim
if not exist "%PYTHON_EXE%" (
    echo [HATA] Tasinabilir Python klasoru ^(python^\^) bulunamadi!
    echo Lutfen programi eksiksiz indirdiginizden ve cikarttiginizdan emin olun.
    echo.
    pause
    exit /b
)

:: nASR.py dosyasini arka planda baslatiyoruz, bat dosyasi kapanir
start "" "%PYTHON_EXE%" nASR.py