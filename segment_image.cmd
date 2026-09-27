@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  單張圖 SAM 3 偵測 -> Cryptomatte
REM  不跑影片 tracker。每張圖各自編號，幀與幀之間的 ID 不會接上。
REM  INPUT 可以是一張 jpg/png/tif/exr，或一個影格資料夾。
REM ============================================================
set "INPUT=F:\video_4_test\PF0003-06_Clip-6_REC709_2K_jpg\00000.jpg"
set "OUTPUT=H:\260927_sam3_test\still.exr"
@REM set "PROMPTS=person,wall,plant"
set "PROMPTS=person,man,woman,child,face,hand,body,car,truck,bus,bicycle,motorcycle,bike,vehicle,animal,cat,dog,bird,horse,cow,sheep,plant,tree,flower,grass,food,fruit,accessory,bag,hat,backpack,umbrella,shoe,clothing,jacket,shirt,pants,dress,coat,skirt,scarf,furniture,chair,sofa,bed,table,desk,window,door,stairs,building,house,wall,floor,ceiling,road,bridge,sign,light,lamp,electronics,phone,computer,screen,television,laptop,mouse,keyboard,book,pen,tool,bottle,cup,plate,mirror,clock,appliance,refrigerator,microwave,oven,sink,toilet,bathtub,sports,ball,bat,glove,goal,net,instrument,guitar,drum,piano,violin,street,sidewalk,crosswalk,pavement,curb,traffic light,traffic sign,streetlight,streetlamp,pole,fire hydrant,mailbox,parking meter,fence,railing,billboard,advertisement,graffiti,intersection,overpass,underpass,alley,park,playground,square,plaza,statue,monument,fountain,bench,bus stop,train,tram,railroad,track,railway,bridge,satellite dish,chimney,roof,sky,cloud,sun,moon,river,canal,pond,lake,stream,waterfall,ocean,sea,harbor,beach,sand,dock,boat,ship,island,mountain,hill,valley,rock,cliff,forest,woods,bush,shrub,leaf,path,trail,countryside,meadow,field,farmland,vineyard,orchard,plantation,desert,plain,plateau,canyon,cave,land,terrain,slope,view,panorama,scenery,background,horizon,distance,cityscape,skyline,landmark,urban,suburban,rural,settlement,village,town,city,tower,skyscraper,office,market,store,shop,restaurant,cafe,street vendor,bike lane,pedestrian,walkway"
set "PRESET=fast"
set "CHECKPOINT=H:\comfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors"
set "EXTRA=--conf 0.2 --min-area 128 --max-objects 256 --ranks 24"
REM ============================================================

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo 找不到 .venv。請先執行 setup_venv.cmd
  pause
  exit /b 1
)
if not exist "%INPUT%" (
  echo 找不到輸入: %INPUT%
  echo 請先把單張圖的路徑填進這個 cmd。
  pause
  exit /b 1
)
if not exist "%CHECKPOINT%" (
  echo 找不到 checkpoint: %CHECKPOINT%
  pause
  exit /b 1
)

if "%PROMPTS%"=="" (
  "%PYTHON%" "%~dp0py\sam3_to_cryptomatte.py" segment --input "%INPUT%" --output "%OUTPUT%" --preset "%PRESET%" --checkpoint "%CHECKPOINT%" %EXTRA%
) else (
  "%PYTHON%" "%~dp0py\sam3_to_cryptomatte.py" segment --input "%INPUT%" --output "%OUTPUT%" --prompts "%PROMPTS%" --checkpoint "%CHECKPOINT%" %EXTRA%
)

echo.
echo Error code: %ERRORLEVEL%
@REM pause
exit /b %ERRORLEVEL%
