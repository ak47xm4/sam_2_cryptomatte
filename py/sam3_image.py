"""Segment one still with the SAM 3 image detector and write Cryptomatte.

This path does not load the video tracker. Each picture is independent, so
object IDs are not stable from frame to frame.
"""

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cryptomatte_from_masks import (
    build_manifest,
    pack_ranks,
    ranks_to_layers,
    render_preview,
    require_openexr,
    save_cryptomatte_exr,
    write_manifest_sidecar,
)
from sam3_media import STILL_EXTS, read_bgr
from sam3_prompts import resolve_prompts
from sam3_track import _safetensors_torch_load, _sam3_bpe_path

_IMAGE_EXTS = STILL_EXTS


def segment_stills(args):
    require_openexr()
    prompts = resolve_prompts(args.preset, args.prompts, args.prompts_file)
    sources = _list_stills(args.input)
    output = Path(args.output)
    single_exr = len(sources) == 1 and output.suffix.lower() == ".exr"
    if single_exr:
        exr_paths = [output]
        preview_paths = [
            None if args.no_preview else output.with_suffix(".preview.png")
        ]
        sidecar = output.with_suffix(".manifest.json")
    else:
        if output.suffix:
            raise SystemExit("多張圖的 --output 必須是資料夾。")
        exr_dir = output / "exr"
        preview_dir = output / "preview"
        exr_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_preview:
            preview_dir.mkdir(parents=True, exist_ok=True)
        exr_paths = [exr_dir / f"{path.stem}.exr" for path in sources]
        preview_paths = [
            None if args.no_preview else preview_dir / f"{path.stem}.png"
            for path in sources
        ]
        sidecar = output / "cryptomatte_manifest.json"

    print(f"單張偵測，不追蹤。概念 ({len(prompts)}): {', '.join(prompts)}", flush=True)
    processor = _load_image_processor(args)
    written = []
    layer_names = [
        f"{args.crypto_name}{index:02d}" for index in range(args.ranks // 2)
    ]
    meta_key = "cryptomatte/" + __layer_hash(args.crypto_name)
    all_names = []

    for index, source in enumerate(sources):
        print(f"[{index + 1}/{len(sources)}] {source.name}", flush=True)
        bgr = read_bgr(source)
        items = _segment_bgr(processor, bgr, prompts, args)
        height, width = bgr.shape[:2]
        names = [name for name, _score, _coverage in items]
        all_names.extend(names)
        manifest = build_manifest(sorted(set(all_names)))
        coverages = [(name, coverage) for name, _score, coverage in items]
        id_ch, cov_ch, dropped = pack_ranks(
            coverages,
            height,
            width,
            args.ranks,
            shuffle=not args.no_shuffle,
            seed=args.seed + index,
        )
        layers, layer_names = ranks_to_layers(id_ch, cov_ch, args.crypto_name)
        meta_key = save_cryptomatte_exr(exr_paths[index], layers, layer_names,
                                        manifest, args.crypto_name)
        written.append(exr_paths[index].name)
        if preview_paths[index] is not None:
            preview_paths[index].parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(preview_paths[index]),
                render_preview(bgr, coverages, width, height),
            )
        print(f"    {len(names)} 個物件 -> {exr_paths[index]}", flush=True)
        if dropped:
            print(f"    {dropped} 個像素超過 rank 上限", flush=True)

    write_manifest_sidecar(
        sidecar,
        args.crypto_name,
        args.ranks,
        build_manifest(sorted(set(all_names))),
        layer_names,
        meta_key,
        written,
    )
    print(f"寫出 {len(written)} 張 EXR。manifest: {sidecar}", flush=True)


def _segment_bgr(processor, bgr, prompts, args):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state = processor.set_image(Image.fromarray(rgb))
    accepted = []
    references = []
    for prompt_index, prompt in enumerate(prompts):
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt, state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or scores is None or len(scores) == 0:
            print(f"    {prompt!r}: 0", flush=True)
            continue
        masks_np = masks.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy().reshape(-1)
        order = np.argsort(-scores_np)
        kept = 0
        token = _token(prompt, prompt_index)
        for row in order:
            if kept >= args.max_objects:
                break
            score = float(scores_np[row])
            if score < args.conf:
                continue
            mask = _continuous_edge(
                np.squeeze(masks_np[row]) > 0.5,
                close_radius=args.edge_close,
                fill_holes=not args.no_fill_holes,
            )
            area = int(mask.sum())
            if area < args.min_area:
                continue
            small = _downsample_bool(mask)
            if any(_iou(small, ref) >= args.dup_iou for ref in references):
                continue
            name = f"{token}_{kept:04d}_shape"
            accepted.append((name, score, mask.astype(np.float32)))
            references.append(small)
            kept += 1
        print(f"    {prompt!r}: {kept}", flush=True)
    accepted.sort(key=lambda item: (-item[1], item[0]))
    return accepted


def _load_image_processor(args):
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise SystemExit(
            "無法載入 sam3 影像模型。請確認 .venv 已安裝 facebookresearch/sam3。\n"
            f"詳細錯誤: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("SAM 3 需要可用的 NVIDIA CUDA GPU。")
    checkpoint = args.checkpoint or None
    print("載入 SAM 3 影像偵測器（不含影片 tracker）...", flush=True)
    kwargs = {
        "bpe_path": _sam3_bpe_path(),
        "device": "cuda",
        "eval_mode": True,
        "enable_segmentation": True,
        "enable_inst_interactivity": False,
        "compile": False,
        "load_from_HF": checkpoint is None,
    }
    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint
    with _safetensors_torch_load(checkpoint):
        model = build_sam3_image_model(**kwargs)
    print(
        f"偵測器已載入。GPU {torch.cuda.memory_allocated() // 1024**2} MiB",
        flush=True,
    )
    return Sam3Processor(model, device="cuda", confidence_threshold=args.conf)


def _list_stills(input_path):
    path = Path(input_path)
    if not path.exists():
        raise SystemExit(f"找不到輸入: {path}")
    if path.is_file():
        if path.suffix.lower() not in _IMAGE_EXTS:
            raise SystemExit("影片追蹤已停用。請改給單張圖或影格資料夾"
                             f"（{' '.join(sorted(_IMAGE_EXTS))}）。")
        return [path]
    files = [
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in _IMAGE_EXTS
    ]
    if not files:
        raise SystemExit(f"資料夾裡沒有靜態影像: {path}")
    return sorted(files, key=lambda item: _frame_key(item.name))


def _frame_key(name):
    stem = Path(name).stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return (0, int(match.group(1)), stem.lower())
    return (1, 0, stem.lower())


def _continuous_edge(mask, close_radius, fill_holes):
    """Close pinholes and stair-steps in a hard SAM mask.

    SAM's mask is a low-resolution logit thresholded at 0.5, so the contour
    breaks into specks. Closing reconnects those gaps. This does not recover
    hair; that needs a trimap matting model such as ViTMatte.
    """
    if close_radius <= 0 and not fill_holes:
        return mask
    image = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    if close_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_radius * 2 + 1, close_radius * 2 + 1),
        )
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        image = _fill_holes(image)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=0.8)
    return blurred >= 127


def _fill_holes(image):
    padded = cv2.copyMakeBorder(image,
                                1,
                                1,
                                1,
                                1,
                                cv2.BORDER_CONSTANT,
                                value=0)
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def _downsample_bool(mask, max_side=480):
    height, width = mask.shape
    scale = max_side / float(max(height, width))
    if scale >= 1.0:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (max(1, int(round(width * scale))), max(1, int(round(
            height * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _iou(left, right):
    if left.shape != right.shape:
        return 0.0
    intersection = np.logical_and(left, right).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _token(prompt, index):
    raw = re.sub(r"\s+", "_", prompt.strip().lower())
    raw = re.sub(r"[^0-9a-z_\-]+", "", raw).strip("_")
    return (raw or f"concept{index:02d}")[:48]


def __layer_hash(crypto_name):
    from mmh3_for_cryptomatte import layer_hash
    return layer_hash(crypto_name)
