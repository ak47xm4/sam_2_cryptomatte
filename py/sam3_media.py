"""Resolve a video, an image, or a frame folder into something SAM 3 can open.

SAM 3 reads a video file, a single image, or a folder whose filenames are
``<integer>.jpg/png/...``. VFX names such as ``shot.1001.exr`` are converted
to a temporary JPEG sequence so frame order stays numeric, while the EXR
output keeps the original frame stem.
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np

SAM_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
STILL_EXTS = SAM_IMAGE_EXTS | {".tif", ".exr"}


class Media:
    def __init__(self, kind, resource_path, width, height, fps, frame_count, frames, video_stem, video_path):
        self.kind = kind
        self.resource_path = resource_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.frames = frames
        self.video_stem = video_stem
        self.video_path = video_path

    def output_stem(self, frame_index, start_number, pad):
        if self.kind == "video":
            return f"{self.video_stem}.{start_number + frame_index:0{pad}d}"
        if frame_index < 0 or frame_index >= len(self.frames):
            raise IndexError(f"frame {frame_index} 超出輸入範圍")
        return self.frames[frame_index]["output_stem"]

    def source_path(self, frame_index):
        if self.kind == "video" or not self.frames:
            return None
        if frame_index < 0 or frame_index >= len(self.frames):
            return None
        return self.frames[frame_index].get("source_path")


class VideoReader:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        self.index = -1

    def read(self, index):
        if not self.capture.isOpened():
            return None
        if index != self.index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.index = index
        return frame

    def close(self):
        self.capture.release()


def prepare_media(input_path, cache_dir):
    path = Path(input_path)
    if not path.exists():
        raise SystemExit(f"找不到輸入: {path}")

    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return _video_media(path)
    if path.is_file():
        return _single_image_media(path, cache_dir)
    if path.is_dir():
        return _folder_media(path, cache_dir)
    raise SystemExit(f"不支援的輸入: {path}")


def read_bgr(path):
    path = Path(path)
    if path.suffix.lower() == ".exr":
        return _exr_to_bgr(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"讀不到影像: {path}")
    return image


def _video_media(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV 開不了影片: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if fps <= 1.0:
        fps = 24.0
    return Media(
        kind="video",
        resource_path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
        frames=[],
        video_stem=path.stem,
        video_path=str(path),
    )


def _single_image_media(path, cache_dir):
    if path.suffix.lower() not in STILL_EXTS:
        raise SystemExit(
            f"不支援的影像格式 {path.suffix}。影片請用 {sorted(VIDEO_EXTS)}。"
        )
    image = read_bgr(path)
    height, width = image.shape[:2]
    if path.suffix.lower() in SAM_IMAGE_EXTS:
        resource = str(path)
    else:
        resource = str(_write_jpeg(image, Path(cache_dir) / "jpg_frames" / "00000.jpg"))
    return Media(
        kind="image",
        resource_path=resource,
        width=width,
        height=height,
        fps=24.0,
        frame_count=1,
        frames=[{"index": 0, "output_stem": path.stem, "source_path": str(path)}],
        video_stem=path.stem,
        video_path=None,
    )


def _folder_media(folder, cache_dir):
    names = [
        name for name in os.listdir(folder)
        if Path(name).suffix.lower() in (STILL_EXTS | {".tif"})
    ]
    if not names:
        raise SystemExit(f"資料夾裡沒有影像: {folder}")
    names = _order_frames(names)
    paths = [folder / name for name in names]
    direct = all(path.suffix.lower() in SAM_IMAGE_EXTS and path.stem.isdigit() for path in paths)
    if direct:
        first = read_bgr(paths[0])
        height, width = first.shape[:2]
        frames = [
            {"index": index, "output_stem": path.stem, "source_path": str(path)}
            for index, path in enumerate(paths)
        ]
        return Media(
            kind="sequence",
            resource_path=str(folder),
            width=width,
            height=height,
            fps=24.0,
            frame_count=len(frames),
            frames=frames,
            video_stem=folder.name,
            video_path=None,
        )

    jpeg_dir = Path(cache_dir) / "jpg_frames"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    width = height = None
    for index, source in enumerate(paths):
        image = read_bgr(source)
        if width is None:
            height, width = image.shape[:2]
        _write_jpeg(image, jpeg_dir / f"{index:05d}.jpg")
        frames.append({
            "index": index,
            "output_stem": source.stem,
            "source_path": str(source),
        })
    print(
        f"輸入檔名不是 SAM 3 能直接排序的 <整數>.jpg/png，已轉成暫存 JPEG: {jpeg_dir}",
        flush=True,
    )
    return Media(
        kind="sequence",
        resource_path=str(jpeg_dir),
        width=width,
        height=height,
        fps=24.0,
        frame_count=len(frames),
        frames=frames,
        video_stem=folder.name,
        video_path=None,
    )


def _order_frames(names):
    """Match SAM 3 when every stem is an integer; otherwise sort by the trailing number."""
    try:
        return sorted(names, key=lambda name: int(Path(name).stem))
    except ValueError:
        return sorted(names, key=_natural_key)


def _natural_key(name):
    stem = Path(name).stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return (0, int(match.group(1)), stem.lower())
    return (1, 0, stem.lower())


def _write_jpeg(image_bgr, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"寫不出 JPEG: {dest}")
    return dest


def _exr_to_bgr(path):
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise SystemExit(f"讀 EXR 需要 OpenEXR。詳細錯誤: {exc}") from exc

    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    channels = set(header["channels"].keys())
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    def read(name):
        raw = exr.channel(name, pixel_type)
        return np.frombuffer(raw, dtype=np.float32).reshape(height, width)

    if {"R", "G", "B"} <= channels:
        rgb = np.stack([read("R"), read("G"), read("B")], axis=-1)
    elif {"r", "g", "b"} <= channels:
        rgb = np.stack([read("r"), read("g"), read("b")], axis=-1)
    else:
        exr.close()
        raise SystemExit(f"EXR 沒有 RGB channel: {path}")
    exr.close()

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    peak = float(np.percentile(rgb, 99.5))
    if peak > 1.0:
        rgb = np.clip(rgb / max(peak, 1e-6), 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / 2.2)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    bgr = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    return np.ascontiguousarray(bgr)
