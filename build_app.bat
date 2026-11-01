@echo off
REM build_app.bat — Build distributable application
REM Run from project root: build_app.bat

echo ============================================================
echo   Building Pupil ^& Limbus Detector
echo ============================================================
echo.

REM ── Check Python ──
python --version 2>NUL
if errorlevel 1 (
    echo ERROR: Python not found. Activate your virtual environment first.
    echo   .venv\Scripts\activate
    exit /b 1
)

REM ── Check ONNX models ──
if not exist models\onnx\segmentation_quantized.onnx (
    if not exist models\onnx\segmentation.onnx (
        echo ERROR: No ONNX models found in models\onnx\
        echo   Run first: python scripts\convert_to_onnx.py
        exit /b 1
    )
)
echo [OK] ONNX models found

REM ── Install build dependencies ──
echo.
echo [1/4] Installing build dependencies...
pip install pyinstaller>=6.0 >NUL 2>&1

REM ── Verify onnxruntime is installed ──
python -c "import onnxruntime" 2>NUL
if errorlevel 1 (
    echo Installing onnxruntime...
    pip install onnxruntime
)
echo [OK] Dependencies ready

REM ── Clean previous builds ──
echo.
echo [2/4] Cleaning previous builds...
if exist build\dist rd /s /q build\dist
if exist build\build_temp rd /s /q build\build_temp

REM ── Build ──
echo.
echo [3/4] Building application with PyInstaller...
echo       This may take 2-5 minutes...
echo.

pyinstaller ^
    --distpath build\dist ^
    --workpath build\build_temp ^
    --clean ^
    --noconfirm ^
    build\pupil_detector.spec

if errorlevel 1 (
    echo.
    echo ============================================================
    echo   BUILD FAILED
    echo   Check the output above for errors.
    echo ============================================================
    exit /b 1
)

REM ── Verify ──
echo.
echo [4/4] Verifying build...

set APP_DIR=build\dist\PupilLimbusDetector

if not exist "%APP_DIR%\PupilLimbusDetector.exe" (
    echo ERROR: Executable not found!
    exit /b 1
)

if not exist "%APP_DIR%\models\onnx\segmentation_quantized.onnx" (
    if not exist "%APP_DIR%\models\onnx\segmentation.onnx" (
        echo WARNING: ONNX model not found in build output!
        echo          The app may not work. Check the spec file.
    )
)

echo.
echo ============================================================
echo   BUILD SUCCESSFUL
echo ============================================================
echo.
echo   Output directory:
echo     %APP_DIR%\
echo.
echo   To run:
echo     %APP_DIR%\PupilLimbusDetector.exe
echo.
echo   To distribute:
echo     Zip the entire %APP_DIR% folder
echo     and send to your client.
echo.

REM ── Show size ──
echo   Contents:
dir /s /b "%APP_DIR%\*.exe" "%APP_DIR%\*.onnx" 2>NUL
echo.

REM ── Calculate total size ──
for /f "tokens=3" %%a in ('dir "%APP_DIR%" /s /-c ^| findstr "File(s)"') do (
    set /a SIZE_MB=%%a / 1048576
)
echo   Total size: approximately %SIZE_MB% MB
echo.
echo ============================================================