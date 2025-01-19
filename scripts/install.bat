@echo off
setlocal enabledelayedexpansion

:: Check for NVIDIA GPU
wmic path win32_VideoController get name | findstr /i "NVIDIA" > nul
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected
    pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.5.1 torchvision==0.20.1
    pip install -r requirements.txt
    goto :end
)

:: Check for AMD GPU
wmic path win32_VideoController get name | findstr /i "AMD" > nul
if %errorlevel% equ 0 (
    echo AMD GPU detected
    pip install --index-url https://download.pytorch.org/whl/rocm5.6 torch==2.5.1 torchvision==0.20.1
    pip install -r requirements.txt
    goto :end
)

:: No supported GPU found
echo No NVIDIA or AMD GPU detected. Installing CPU-only version.
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt

:end
echo Installation completed.
pause