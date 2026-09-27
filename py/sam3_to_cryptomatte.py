"""SAM 3 tracking mattes to Cryptomatte EXR.

Examples
--------
py -3 py/sam3_to_cryptomatte.py track --input D:\\shot.mp4 --output D:\\cm
py -3 py/sam3_to_cryptomatte.py track --input D:\\frames --output D:\\cm --preset fast
py -3 py/sam3_to_cryptomatte.py track --input D:\\shot.mp4 --output D:\\cm --prompts person,car,chair
py -3 py/sam3_to_cryptomatte.py encode --masks D:\\masks --output D:\\cm\\frame.exr
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sam3_image import segment_stills  # noqa: E402
from sam3_prompts import PRESETS  # noqa: E402
from sam3_track import encode_masks, reencode, track_and_export  # noqa: E402


def main(argv=None):
    _configure_stdio()
    parser = argparse.ArgumentParser(
        prog="sam3_to_cryptomatte",
        description=("用 SAM 3 依文字概念分割畫面中的實例、沿時間追蹤 matte，"
                     "再寫成和 2024 腳本相同格式的 Cryptomatte EXR。"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("preset: " + ", ".join(sorted(PRESETS)) + "\n"
                "SAM 3 會把「一個概念的所有實例」找出來並追蹤。"
                "預設 everything 會依序跑人、衣服、車、動物、道具、場景，用來逼近分割畫面裡的東西。\n"
                "每個概念都是一次完整追蹤，概念越多越久。"),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--crypto-name",
        default="ViewLayer.CryptoObject",
        help=
        "Cryptomatte metadata 名稱。預設和 2024 腳本一樣，channel 會是 ViewLayer.CryptoObject00。",
    )
    common.add_argument("--ranks",
                        type=int,
                        default=6,
                        help="ID rank 數量，必須是偶數。兩個 rank 是一層 EXR。")
    common.add_argument("--shuffle-ranks",
                        action="store_true",
                        help="重疊且 coverage 相同時，隨機對調 rank。")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--no-preview",
                        action="store_true",
                        help="不寫彩色 preview PNG。")
    common.add_argument("--preview-video",
                        action="store_true",
                        help="額外寫 preview.mp4。")
    common.add_argument("--fps",
                        type=float,
                        default=0,
                        help="preview 影片的 fps。0 表示沿用輸入。")

    sub = parser.add_subparsers(dest="command", required=True)

    track = sub.add_parser("track",
                           parents=[common],
                           help="SAM 3 分割並追蹤，輸出 Cryptomatte 序列。")
    track.add_argument("--input", required=True, help="影片、單張圖，或影格資料夾。")
    track.add_argument("--output", required=True, help="輸出資料夾。")
    track.add_argument("--preset",
                       default="everything",
                       choices=sorted(PRESETS))
    track.add_argument("--prompts",
                       default="",
                       help="逗號分隔的概念，會蓋過 --preset。例如 person,car,tree")
    track.add_argument("--prompts-file",
                       default="",
                       help="文字檔，一行一個概念，# 開頭是註解。")
    track.add_argument("--prompt-frame",
                       type=int,
                       default=0,
                       help="在哪一幀下文字提示。")
    track.add_argument(
        "--direction",
        default="auto",
        choices=["auto", "forward", "backward", "both"],
        help="auto：輸出範圍在偵測幀之後就往前追；範圍包住偵測幀前後就雙向追。",
    )
    track.add_argument("--start",
                       type=int,
                       default=None,
                       help="輸出的第一幀（含），0 為輸入的第一幀。")
    track.add_argument("--end", type=int, default=None, help="輸出的最後一幀（含）。")
    track.add_argument("--start-number",
                       type=int,
                       default=1001,
                       help="影片輸出的幀號。影格資料夾會沿用原檔名。")
    track.add_argument("--pad", type=int, default=4, help="影片輸出幀號的位數。")
    track.add_argument("--conf", type=float, default=0.4, help="低於這個分數的實例不要。")
    track.add_argument("--min-area",
                       type=int,
                       default=128,
                       help="小於這個像素數的 mask 不要。")
    track.add_argument("--dup-iou",
                       type=float,
                       default=0.75,
                       help="和已接受的物件 IoU 高於這個值就視為重複。")
    track.add_argument("--max-objects",
                       type=int,
                       default=48,
                       help="每個概念最多追蹤幾個實例。")
    track.add_argument("--checkpoint",
                       default="",
                       help="本地 SAM 3 checkpoint。空白則由 sam3 自己下載。")
    track.add_argument("--gpus", default="", help="GPU id，例如 0 或 0,1。")
    track.add_argument("--compile",
                       action="store_true",
                       help="打開 torch.compile。Windows 上常常比較不穩。")
    track.add_argument("--async-load",
                       action="store_true",
                       help="SAM 3 非同步讀幀。")
    track.add_argument("--keep-cache",
                       action="store_true",
                       help="留下 mask 暫存，之後可用 reencode。")
    track.add_argument("--dry-run",
                       action="store_true",
                       help="只印輸入、概念和追蹤範圍，不載入模型。")
    track.set_defaults(func=track_and_export)

    still = sub.add_parser(
        "segment",
        parents=[common],
        help="單張圖用 SAM 3 偵測器分割，不追蹤，寫出 Cryptomatte。",
    )
    still.add_argument("--input", required=True, help="一張圖，或一個靜態影格資料夾。")
    still.add_argument("--output", required=True, help="單一 .exr，或多張圖的輸出資料夾。")
    still.add_argument("--preset", default="fast", choices=sorted(PRESETS))
    still.add_argument("--prompts", default="", help="逗號分隔的概念，會蓋過 --preset。")
    still.add_argument("--prompts-file", default="", help="文字檔，一行一個概念。")
    still.add_argument("--conf", type=float, default=0.5, help="低於這個分數的實例不要。")
    still.add_argument("--min-area",
                       type=int,
                       default=128,
                       help="小於這個像素數的 mask 不要。")
    still.add_argument("--dup-iou",
                       type=float,
                       default=0.75,
                       help="跨概念 IoU 高於這個值就視為重複。")
    still.add_argument("--max-objects",
                       type=int,
                       default=16,
                       help="每個概念最多幾個實例。")
    still.add_argument(
        "--edge-close",
        type=int,
        default=3,
        help="把碎掉的邊緣閉合，單位是像素。0 表示不做。",
    )
    still.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="不要填滿 mask 內部的破洞。",
    )
    still.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不要打亂重疊處的 rank。預設每個重疊物件都會隨機輪到 rank 0，前後都能點到。",
    )
    still.add_argument(
        "--checkpoint",
        default="",
        help="本地 SAM 3 checkpoint。偵測器會只用其中的 detector 權重。",
    )
    still.set_defaults(func=segment_stills, ranks=24)

    encode = sub.add_parser(
        "encode",
        parents=[common],
        help="不跑 SAM，把現成的 mask 圖轉成 Cryptomatte。資料夾內每張圖是一個物件。",
    )
    encode.add_argument("--masks",
                        required=True,
                        help="一幀的 mask 資料夾，或每個子資料夾是一幀。")
    encode.add_argument("--output", required=True, help="單一 .exr，或序列的輸出資料夾。")
    encode.set_defaults(func=encode_masks)

    again = sub.add_parser("reencode",
                           parents=[common],
                           help="用 --keep-cache 留下的暫存重寫 EXR。")
    again.add_argument("--cache",
                       required=True,
                       help="track 輸出裡的 _sam3_cache 資料夾。")
    again.add_argument("--output", required=True, help="新的輸出資料夾。")
    again.set_defaults(func=reencode)

    args = parser.parse_args(argv)
    _validate_common(args)
    args.func(args)


def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _validate_common(args):
    if args.ranks < 2 or args.ranks % 2 != 0 or args.ranks > 32:
        raise SystemExit(
            "--ranks 必須是 2 到 32 的偶數。segment 預設 24，也就是 12 層 CryptoObject00–11。")
    name = args.crypto_name.strip()
    if not name or name[-1].isdigit():
        raise SystemExit("--crypto-name 不能是空的，也不能以數字結尾。")
    args.crypto_name = name
    if args.command in ("track", "segment"):
        if args.conf < 0:
            raise SystemExit("--conf 不能是負數。")
        if args.min_area < 0 or args.max_objects < 1:
            raise SystemExit("--min-area 不能是負數，--max-objects 至少要 1。")
        if not 0 <= args.dup_iou <= 1:
            raise SystemExit("--dup-iou 必須在 0 到 1。")
    if args.command == "track" and args.pad < 1:
        raise SystemExit("--pad 至少要 1。")


if __name__ == "__main__":
    main()
