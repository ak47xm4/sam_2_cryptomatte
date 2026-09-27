@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  SAM 3.1 tracking matte -> Cryptomatte
REM  改下面這幾行即可。PROMPTS 有內容時會蓋過 PRESET。
REM ============================================================
@REM set "INPUT=H:\your_shot.mp4"
set "INPUT=F:\video_4_test\PF0003-01_Clip-1_REC709_2K_1.mov"
set "OUTPUT=H:\260927_sam3_test"
@REM set "PROMPTS=person,wall,plant"
set "PRESET=fast"
set "CHECKPOINT=H:\comfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"

REM 可選：幀範圍、分數、每個概念最多幾個實例
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "EXTRA=--conf 0.4 --min-area 128 --max-objects 16"
REM 例如只輸出第 0 到第 50 幀： set "EXTRA=--start 0 --end 50"
REM 整段 everything：把 PROMPTS 改成空的，PRESET 改成 everything
REM ============================================================

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo 找不到 .venv。請先執行 setup_venv.cmd
  pause
  exit /b 1
)
if not exist "%INPUT%" (
  echo 找不到輸入: %INPUT%
  echo 請先編輯這個 cmd 裡的 INPUT。
  pause
  exit /b 1
)
if not exist "%CHECKPOINT%" (
  echo 找不到 checkpoint: %CHECKPOINT%
  pause
  exit /b 1
)

if "%PROMPTS%"=="" (
  "%PYTHON%" "%~dp0py\sam3_to_cryptomatte.py" track --input "%INPUT%" --output "%OUTPUT%" --preset "%PRESET%" --checkpoint "%CHECKPOINT%" %EXTRA%
) else (
  "%PYTHON%" "%~dp0py\sam3_to_cryptomatte.py" track --input "%INPUT%" --output "%OUTPUT%" --prompts "%PROMPTS%" --checkpoint "%CHECKPOINT%" %EXTRA%
)

echo.
echo 結束代碼 %ERRORLEVEL%
echo EXR 在 %OUTPUT%\exr
pause
exit /b %ERRORLEVEL%
