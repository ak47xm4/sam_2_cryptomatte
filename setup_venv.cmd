@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo 建立或更新 .venv
if not exist "%~dp0.venv\Scripts\python.exe" (
  py -3 -m venv "%~dp0.venv"
  if errorlevel 1 (
    echo 建立 .venv 失敗。需要 Python 3.12。
    pause
    exit /b 1
  )
)

set "PYTHON=%~dp0.venv\Scripts\python.exe"
"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r "%~dp0requirements-sam3.txt"
if errorlevel 1 (
  echo 安裝 Python 套件失敗。
  pause
  exit /b 1
)

echo.
echo 基礎套件已裝進 .venv。encode 可以跑。
echo track 還需要 PyTorch CUDA 和 facebookresearch/sam3，請在這個 .venv 裡安裝：
echo   .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo   git clone https://github.com/facebookresearch/sam3.git
echo   .venv\Scripts\python.exe -m pip install -e sam3
echo.
pause
