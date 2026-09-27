# SAM 3 tracking matte → Cryptomatte

用 [SAM 3](https://github.com/facebookresearch/sam3) 找出畫面裡每個文字概念的全部分割實例，沿時間追蹤 matte，再寫成 Cryptomatte EXR。ID 的演算法和 EXR metadata 沿用這個 repo 2024 年已能進合成軟體的寫法：MurmurHash3 32-bit、`uint32_to_float32`、`ViewLayer.CryptoObject00/01/02`。

SAM 3 不是第一代 SAM 的網格自動分割。它吃的是短語，例如 `person` 或 `red chair`，然後把符合的實例全部找出來，並在影片裡保持同一個 object id。這個 CLI 的 `everything` preset 就是依序追蹤人、衣服、車、動物、道具和場景，用來逼近「分割畫面裡的東西」。

2024 的單幀腳本還在，沒有改行為：

- `py/sam_cv_test_A_v0021.py`
- `py/mmh3_for_cryptomatte.py`
- `py/write_multi_RGBA_exr.py`

## 環境

專案裡的 `.venv` 用 Python 3.12。雙擊 `setup_venv.cmd` 會建立它並裝 Cryptomatte 需要的套件。`track_sam3.cmd` 和 `encode_masks.cmd` 開頭的變數就是範本，改路徑後再雙擊。

ComfyUI 的 `h:\comfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors` 可以用。它是 SAM 3.1 multiplex 的 fp16 權重，tensor 名稱是 `detector.*` 和 `tracker.*`，跟官方 `sam3.1_multiplex.pt` 同一套。官方程式只會 `torch.load` `.pt`，CLI 在遇到這個 safetensors 時會改走 `build_sam3_multiplex_video_predictor`。Tokenizer 不在這顆檔案裡，所以 `.venv` 仍要安裝 [facebookresearch/sam3](https://github.com/facebookresearch/sam3)。FlashAttention 3 在 Windows 上常常沒有，載入時固定關掉。

## 安裝

需要 NVIDIA GPU、CUDA、Python 3.12。

```bash
setup_venv.cmd
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
.venv\Scripts\python.exe -m pip install -e sam3
```

到 Hugging Face 的 [facebook/sam3](https://huggingface.co/facebook/sam3) 申請 checkpoint，通過後：

```bash
hf auth login
```

## 追蹤並輸出

```bash
py -3 py/sam3_to_cryptomatte.py track --input D:\shot.mp4 --output D:\cm
```

常用寫法：

```bash
py -3 py/sam3_to_cryptomatte.py track --input D:\frames --output D:\cm --preset fast
py -3 py/sam3_to_cryptomatte.py track --input D:\shot.mp4 --output D:\cm --prompts person,car,chair
py -3 py/sam3_to_cryptomatte.py track --input D:\shot.mp4 --output D:\cm --prompts-file concepts.txt
py -3 py/sam3_to_cryptomatte.py track --input D:\shot.mp4 --output D:\cm --dry-run
```

`--preset` 可以是 `everything`（預設）、`fast`、`people`、`vehicles`、`animals`、`props`、`environment`。`--prompts` 或 `--prompts-file` 會蓋過 preset。`concepts.txt` 一行一個概念，`#` 開頭是註解。

輸入可以是：

- 影片：`.mp4` `.mov` `.avi` `.mkv` `.webm`
- 單張圖：`.jpg` `.png` `.tif` `.exr` 等
- 影格資料夾。檔名若是 `1001.jpg` 這種整數，會直接交給 SAM 3。`shot.1001.exr` 這類名字會先轉成暫存 JPEG，EXR 仍用原來的檔名。

每個概念都會把選取的幀範圍再追蹤一次。`everything` 大約三十幾個概念，長鏡頭會很久。先用 `--preset fast` 或幾個 `--prompts` 看品質，再加概念。

物件進到畫面後才出現時，同一個概念在後續幀仍會收下新的 id。跨概念若和已接受的 mask IoU 超過 `--dup-iou`（預設 0.75），會當成同一個東西丟掉，避免 `person` 和 `human` 各寫一層。`face` 包在 `person` 裡面時 IoU 通常不高，兩層都會留下。

## 輸出

```text
D:\cm\
  exr\shot.1001.exr
  preview\shot.1001.png
  cryptomatte_manifest.json
```

影片的幀號預設從 1001 起，可用 `--start-number` 和 `--pad` 改。影格資料夾沿用原本的檔名，不加起始幀號。

EXR 是 float32，不是 half。Cryptomatte 的 ID 是 float 的原始位元，壓成 half 會壞。每個 layer 的 RGBA 是兩組 id / coverage。預設 6 個 rank，也就是：

- `ViewLayer.CryptoObject00`
- `ViewLayer.CryptoObject01`
- `ViewLayer.CryptoObject02`

重疊時，分數高的 matte 留在 rank 0，下一個物件寫進下一個 rank。這和 2024 `v0021` 不同：舊腳本在重疊處會把新的 ID 寫進已經佔用的 rank。Hash 和 metadata 名稱沒有改，合成軟體還是認同一組 channel。舊腳本仍可拿來對單幀 PNG。

Preview PNG 用穩定顏色把每個 id 畫出來，並標上 `person_0003` 這類名字，用來檢查追蹤有沒有跳號。加 `--preview-video` 會再寫 `preview.mp4`。

`--keep-cache` 會留下 `_sam3_cache`。之後若只想改 rank 或 Cryptomatte 名稱、不想重跑 GPU：

```bash
py -3 py/sam3_to_cryptomatte.py reencode --cache D:\cm\_sam3_cache --output D:\cm_v2 --ranks 8
```

## 只把現成 mask 轉成 Cryptomatte

一幀一個資料夾，每張圖一個物件，檔名就是 Cryptomatte 名稱。這條路徑不載入 SAM 3。

```bash
py -3 py/sam3_to_cryptomatte.py encode --masks D:\masks --output D:\cm\frame.exr
```

若 `D:\masks\1001\*.png`、`D:\masks\1002\*.png` 這種一層子資料夾是一幀：

```bash
py -3 py/sam3_to_cryptomatte.py encode --masks D:\masks --output D:\cm
```

同一個檔名在各幀會得到同一個 ID，所以可以拿來交已經追好的 matte。

確認 EXR 讀寫沒有改到 ID 位元：

```bash
py -3 py/cryptomatte_from_masks.py
```

## 在 Nuke / Blender 裡看

Cryptomatte 名稱是 `ViewLayer.CryptoObject`。Nuke 的 Cryptomatte 節點選這個 layer，manifest 裡的名字就是 `person_0003`、`car_0000` 這種追蹤 id。同一個名字在整段 EXR 裡的 hash 固定，所以 picker 不會逐幀換物件。

若重疊超過 6 個 rank，多出來的像素會被捨棄，終端機會提示。把 `--ranks` 調成更大的偶數即可。
