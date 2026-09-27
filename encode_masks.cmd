@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  不跑 SAM，把現成 mask 轉成 Cryptomatte
REM  MASKS 裡面每張圖是一個物件，檔名就是名稱。
REM  若 MASKS 底下是 1001\ 1002\ 這種子資料夾，OUTPUT 請填資料夾。
REM ============================================================
set "MASKS=H:\your_masks"
set "OUTPUT=H:\your_masks_cryptomatte\frame.exr"
REM ============================================================

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo 找不到 .venv。請先執行 setup_venv.cmd
  pause
  exit /b 1
)
if not exist "%MASKS%" (
  echo 找不到 mask 資料夾: %MASKS%
  echo 請先編輯這個 cmd 裡的 MASKS。
  pause
  exit /b 1
)

"%PYTHON%" "%~dp0py\sam3_to_cryptomatte.py" encode --masks "%MASKS%" --output "%OUTPUT%"

echo.
echo 結束代碼 %ERRORLEVEL%
pause
exit /b %ERRORLEVEL%
