"""Run SAM 3 text concepts across a shot and cache tracked mattes.

``add_prompt`` resets the tracker, and one session holds one text concept.
Each concept is therefore detected, propagated, and saved on its own. Object
names are ``{concept}_{obj_id}`` so the same Cryptomatte ID follows that
instance for the whole shot.
"""

import contextlib
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cryptomatte_from_masks import (
    build_manifest,
    load_coverage,
    render_preview,
    require_openexr,
    ranks_to_layers,
    pack_ranks,
    save_cryptomatte_exr,
    write_manifest_sidecar,
)
from sam3_media import VideoReader, prepare_media, read_bgr
from sam3_prompts import resolve_prompts


def track_and_export(args):
    require_openexr()
    output_dir = Path(args.output)
    if output_dir.suffix:
        raise SystemExit("track 的 --output 必須是資料夾，不是單一檔案。")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_sam3_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    prompts = resolve_prompts(args.preset, args.prompts, args.prompts_file)
    media = prepare_media(args.input, cache_dir)
    direction, start_frame, max_track, write_start, write_end = plan_propagation(
        media, args)

    print(f"輸入: {args.input}", flush=True)
    print(
        f"類型: {media.kind}  {media.width}x{media.height}  "
        f"幀數: {media.frame_count or '未知'}  fps: {media.fps:.3f}",
        flush=True,
    )
    print(f"概念 ({len(prompts)}): {', '.join(prompts)}", flush=True)
    print(
        f"追蹤方向: {direction}  起始幀: {start_frame}  "
        f"max_frame_num_to_track: {max_track}  "
        f"輸出幀: {write_start}..{write_end if write_end is not None else '最後'}",
        flush=True,
    )
    if len(prompts) > 12:
        print("概念很多，每個概念都會把整段再追蹤一次。", flush=True)
    if args.dry_run:
        print("dry-run：沒有載入 SAM 3。", flush=True)
        _delete_tree(cache_dir)
        return

    predictor, session_id = _open_session(media.resource_path, args)
    references = []
    objects_by_frame = defaultdict(list)
    frames_meta = {}
    try:
        for prompt_index, prompt in enumerate(prompts):
            _track_one_prompt(
                predictor=predictor,
                session_id=session_id,
                prompt=prompt,
                prompt_index=prompt_index,
                prompt_count=len(prompts),
                args=args,
                media=media,
                direction=direction,
                start_frame=start_frame,
                max_track=max_track,
                write_start=write_start,
                write_end=write_end,
                references=references,
                objects_by_frame=objects_by_frame,
                frames_meta=frames_meta,
                cache_dir=cache_dir,
            )
            _release_cuda_cache()
    finally:
        _close_session(predictor, session_id)

    run = {
        "media_kind": media.kind,
        "resource_path": media.resource_path,
        "video_path": media.video_path,
        "width": media.width,
        "height": media.height,
        "fps": media.fps,
        "frames": [frames_meta[index] for index in sorted(frames_meta)],
        "objects": {
            str(index): objects_by_frame[index]
            for index in sorted(objects_by_frame)
        },
    }
    run_path = cache_dir / "run.json"
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    export_run(
        run,
        output_dir=output_dir,
        cache_dir=cache_dir,
        crypto_name=args.crypto_name,
        ranks=args.ranks,
        shuffle=args.shuffle_ranks,
        seed=args.seed,
        preview=not args.no_preview,
        preview_video=args.preview_video,
        fps=args.fps or media.fps or 24.0,
    )
    if not args.keep_cache:
        _delete_tree(cache_dir)
        print("已刪除暫存 mask。若要事後改 rank 再輸出，請加 --keep-cache。", flush=True)


def reencode(args):
    cache_dir = Path(args.cache)
    run_path = cache_dir / "run.json"
    if not run_path.is_file():
        raise SystemExit(f"找不到 {run_path}。track 時需要 --keep-cache。")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_run(
        run,
        output_dir=output_dir,
        cache_dir=cache_dir,
        crypto_name=args.crypto_name,
        ranks=args.ranks,
        shuffle=args.shuffle_ranks,
        seed=args.seed,
        preview=not args.no_preview,
        preview_video=args.preview_video,
        fps=args.fps or float(run.get("fps") or 24.0),
    )


def export_run(run, output_dir, cache_dir, crypto_name, ranks, shuffle, seed,
               preview, preview_video, fps):
    frames = run.get("frames") or []
    objects = run.get("objects") or {}
    if not frames:
        raise SystemExit("沒有任何追蹤幀。請確認影片、概念，或把 --conf 調低。")

    names = sorted(
        {entry["name"]
         for entries in objects.values()
         for entry in entries})
    if not names:
        print("沒有任何物件通過門檻。可調低 --conf 或 --min-area，或改用更貼近畫面的 --prompts。",
              flush=True)
    manifest = build_manifest(names)
    width = int(run["width"])
    height = int(run["height"])
    exr_dir = Path(output_dir) / "exr"
    preview_dir = Path(output_dir) / "preview"
    exr_dir.mkdir(parents=True, exist_ok=True)
    if preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    preview_video_path = Path(output_dir) / "preview.mp4"
    reader = VideoReader(
        run["video_path"]) if preview and run.get("video_path") else None
    written = []
    dropped_total = 0
    layer_names = [f"{crypto_name}{index:02d}" for index in range(ranks // 2)]
    meta_key = "cryptomatte/" + __layer_hash(crypto_name)

    try:
        for frame in frames:
            index = int(frame["index"])
            entries = objects.get(str(index), [])
            items = _load_cached_items(entries, cache_dir)
            items.sort(key=lambda item: (-item[1], item[0]))
            coverages = [(name, coverage) for name, _score, coverage in items]
            id_ch, cov_ch, dropped = pack_ranks(coverages,
                                                height,
                                                width,
                                                ranks,
                                                shuffle=shuffle,
                                                seed=seed + index)
            dropped_total += dropped
            layers, layer_names = ranks_to_layers(id_ch, cov_ch, crypto_name)
            stem = frame["output_stem"]
            exr_path = exr_dir / f"{stem}.exr"
            meta_key = save_cryptomatte_exr(exr_path, layers, layer_names,
                                            manifest, crypto_name)
            written.append(exr_path.name)
            if preview:
                source = _source_bgr(reader, frame, run)
                preview_image = render_preview(source, coverages, width,
                                               height)
                preview_path = preview_dir / f"{stem}.png"
                cv2.imwrite(str(preview_path), preview_image)
                if preview_video:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(preview_video_path),
                                                       fourcc, float(fps),
                                                       (width, height))
                    if video_writer is not None and video_writer.isOpened():
                        video_writer.write(preview_image)
            if dropped:
                print(f"  {stem}: {dropped} 個像素超過 {ranks} 個 rank，已捨棄",
                      flush=True)
    finally:
        if reader is not None:
            reader.close()
        if video_writer is not None:
            video_writer.release()

    sidecar = Path(output_dir) / "cryptomatte_manifest.json"
    write_manifest_sidecar(sidecar, crypto_name, ranks, manifest, layer_names,
                           meta_key, written)
    print(f"寫出 {len(written)} 張 EXR，{len(names)} 個物件。", flush=True)
    print(f"EXR: {exr_dir}", flush=True)
    print(f"manifest: {sidecar}", flush=True)
    if preview:
        print(f"preview: {preview_dir}", flush=True)
    if preview_video and preview_video_path.is_file():
        print(f"preview 影片: {preview_video_path}", flush=True)
    if dropped_total:
        print(f"共有 {dropped_total} 個像素塞不進 rank。可加大 --ranks（必須是偶數）。",
              flush=True)


def encode_masks(args):
    """Turn a folder of mask images into Cryptomatte without running SAM 3."""
    require_openexr()
    from cryptomatte_from_masks import list_mask_files

    root = Path(args.masks)
    if not root.is_dir():
        raise SystemExit(f"找不到 mask 資料夾: {root}")
    images = list_mask_files(root)
    subdirs = [
        path for path in root.iterdir()
        if path.is_dir() and list_mask_files(path)
    ]
    if images and subdirs:
        raise SystemExit("masks 資料夾同時有圖片和子資料夾，請只留一種。")
    if not images and not subdirs:
        raise SystemExit(f"找不到 mask 圖片: {root}")

    output = Path(args.output)
    if subdirs:
        if output.suffix:
            raise SystemExit("序列 mask 的 --output 必須是資料夾。")
        subdirs = sorted(subdirs, key=lambda path: _subdir_key(path.name))
        _encode_sequence(subdirs, output, args)
        return
    if output.suffix.lower() == ".exr":
        _encode_one_folder(root,
                           output,
                           args,
                           preview_path=_preview_path_for_exr(output, args))
        return
    output.mkdir(parents=True, exist_ok=True)
    _encode_one_folder(
        root,
        output / "exr" / f"{root.name}.exr",
        args,
        preview_path=None if args.no_preview else output / "preview" /
        f"{root.name}.png",
        sidecar=output / "cryptomatte_manifest.json",
    )


def plan_propagation(media, args):
    count = media.frame_count
    prompt = args.prompt_frame
    write_start = 0 if args.start is None else args.start
    if write_start < 0 or prompt < 0:
        raise SystemExit("--start 和 --prompt-frame 不能是負數。")
    if count > 0 and prompt >= count:
        raise SystemExit(f"--prompt-frame {prompt} 超出幀數 {count}。")
    if count > 0:
        last = count - 1
        write_end = last if args.end is None else min(args.end, last)
    else:
        write_end = args.end
    if write_end is not None and write_end < write_start:
        raise SystemExit("--end 比 --start 還早。")
    if prompt < write_start or (write_end is not None and prompt > write_end):
        raise SystemExit("偵測幀 --prompt-frame 必須落在 --start 和 --end 裡面。")

    direction = args.direction
    if direction == "auto":
        # The prompt frame is required to sit inside the output range, so auto
        # only chooses forward, or both when frames before the prompt are kept.
        direction = "both" if write_start < prompt else "forward"

    if direction == "forward":
        if write_end is None:
            max_track = None
        else:
            max_track = write_end - prompt
        return direction, prompt, max_track, write_start, write_end
    if direction == "backward":
        return direction, prompt, prompt - write_start, write_start, write_end

    back_len = prompt - write_start
    if write_end is not None:
        forward_len = write_end - prompt
        max_track = max(forward_len, back_len)
    elif count > 0:
        max_track = max((count - 1) - prompt, back_len)
    else:
        max_track = None
    return "both", prompt, max_track, write_start, write_end


def _track_one_prompt(
    predictor,
    session_id,
    prompt,
    prompt_index,
    prompt_count,
    args,
    media,
    direction,
    start_frame,
    max_track,
    write_start,
    write_end,
    references,
    objects_by_frame,
    frames_meta,
    cache_dir,
):
    print(f"[{prompt_index + 1}/{prompt_count}] 偵測 {prompt!r}", flush=True)
    response = predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": args.prompt_frame,
        "text": prompt,
    })
    token = _token(prompt, prompt_index)
    accepted = {}
    rejected = set()
    detected, _ = _select_instances(
        parse_instances(response.get("outputs")),
        token=token,
        references=references,
        args=args,
        accepted=accepted,
        rejected=rejected,
        allow_new=True,
    )
    for obj_id in sorted(rejected):
        _remove_object(predictor, session_id, obj_id)
    print(f"    這一幀接受 {len(accepted)} 個，排除 {len(rejected)} 個", flush=True)

    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": direction,
        "start_frame_index": start_frame,
    }
    if max_track is not None:
        request["max_frame_num_to_track"] = int(max_track)

    seen = set()
    saved_frames = 0
    # Backward propagation does not yield the prompted frame, so keep that matte
    # from the detection result when the frame is inside the output range.
    if direction == "backward" and _frame_in_range(args.prompt_frame,
                                                   write_start, write_end):
        seen.add(args.prompt_frame)
        _remember_frame(frames_meta, media, args.prompt_frame, args)
        if detected:
            _save_prompt_frame(cache_dir, args.prompt_frame, prompt_index,
                               detected, objects_by_frame)
            saved_frames += 1
    for response in predictor.handle_stream_request(request):
        frame_index = int(response["frame_index"])
        if frame_index in seen:
            continue
        seen.add(frame_index)
        if not _frame_in_range(frame_index, write_start, write_end):
            continue
        instances = parse_instances(response.get("outputs"))
        named, newly_rejected = _select_instances(
            instances,
            token=token,
            references=references,
            args=args,
            accepted=accepted,
            rejected=rejected,
            allow_new=True,
        )
        for obj_id in newly_rejected:
            _remove_object(predictor, session_id, obj_id)
        _remember_frame(frames_meta, media, frame_index, args)
        if named:
            _save_prompt_frame(cache_dir, frame_index, prompt_index, named,
                               objects_by_frame)
            saved_frames += 1
        if frame_index % 25 == 0:
            print(f"    frame {frame_index}: {len(named)} masks", flush=True)
    print(f"    {prompt!r} 寫入 {saved_frames} 幀，累計物件 {len(accepted)}",
          flush=True)


def _select_instances(instances, token, references, args, accepted, rejected,
                      allow_new):
    """Return names for this frame and object ids newly rejected.

    ``accepted`` and ``rejected`` are updated in place. An id that was already
    accepted keeps its name for the rest of the shot.
    """
    named = []
    newly_rejected = []
    ordered = sorted(instances, key=lambda item: item[1], reverse=True)
    for obj_id, score, mask in ordered:
        if obj_id in rejected:
            continue
        if obj_id in accepted:
            if int(mask.sum()) > 0:
                named.append((accepted[obj_id], score, mask))
            continue
        if not allow_new:
            continue
        area = int(mask.sum())
        if score < args.conf or area < args.min_area or len(
                accepted) >= args.max_objects:
            rejected.add(obj_id)
            newly_rejected.append(obj_id)
            continue
        small = _downsample_bool(mask)
        if any(_iou(small, ref) >= args.dup_iou for ref in references):
            rejected.add(obj_id)
            newly_rejected.append(obj_id)
            continue
        name = f"{token}_{obj_id:04d}"
        accepted[obj_id] = name
        references.append(small)
        named.append((name, score, mask))
    return named, newly_rejected


def _save_prompt_frame(cache_dir, frame_index, prompt_index, named,
                       objects_by_frame):
    masks = np.stack([mask.astype(np.uint8) for _name, _score, mask in named])
    names = np.asarray([name for name, _score, _mask in named], dtype="U128")
    scores = np.asarray([score for _name, score, _mask in named],
                        dtype=np.float32)
    relative = Path("masks") / f"{frame_index:06d}_p{prompt_index:03d}.npz"
    path = Path(cache_dir) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=names, scores=scores, masks=masks)
    for row, (name, score, _mask) in enumerate(named):
        objects_by_frame[frame_index].append({
            "name": name,
            "score": float(score),
            "file": relative.as_posix(),
            "row": row,
        })


def _frame_in_range(frame_index, write_start, write_end):
    if frame_index < write_start:
        return False
    if write_end is not None and frame_index > write_end:
        return False
    return True


def _remember_frame(frames_meta, media, frame_index, args):
    if frame_index in frames_meta:
        return
    source = media.source_path(frame_index)
    frames_meta[frame_index] = {
        "index": frame_index,
        "output_stem": media.output_stem(frame_index, args.start_number,
                                         args.pad),
        "source_path": source,
    }


def _load_cached_items(entries, cache_dir):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["file"]].append(entry)
    items = []
    for relative, group in grouped.items():
        path = Path(cache_dir) / relative
        data = np.load(path)
        masks = np.array(data["masks"])
        data.close()
        for entry in group:
            items.append((
                entry["name"],
                float(entry["score"]),
                masks[int(entry["row"])].astype(np.float32),
            ))
    return items


def _source_bgr(reader, frame, run):
    source_path = frame.get("source_path")
    if source_path:
        try:
            return read_bgr(source_path)
        except Exception:
            return None
    if reader is None:
        return None
    return reader.read(int(frame["index"]))


def parse_instances(outputs):
    """Normalize SAM 3 video outputs to ``(obj_id, score, bool HxW)``."""
    if not outputs:
        return []
    if isinstance(outputs, dict) and _first_key(
            outputs, ("out_binary_masks", "masks", "binary_masks")):
        masks = _as_numpy(outputs[_first_key(
            outputs, ("out_binary_masks", "masks", "binary_masks"))])
        ids = _as_numpy(outputs[_first_key(
            outputs, ("out_obj_ids", "obj_ids", "object_ids"))]).reshape(-1)
        score_key = _first_key(outputs, ("out_probs", "probs", "scores"))
        if score_key is None:
            scores = np.ones(len(ids), dtype=np.float32)
        else:
            scores = _as_numpy(outputs[score_key]).reshape(-1)
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim != 3:
            raise RuntimeError(f"無法解讀 SAM 3 mask 形狀: {masks.shape}")
        instances = []
        count = min(len(ids), masks.shape[0], len(scores))
        for index in range(count):
            instances.append((int(ids[index]), float(scores[index]),
                              _to_bool(masks[index])))
        return instances

    if isinstance(outputs, dict):
        instances = []
        for key, value in outputs.items():
            if not isinstance(value, dict) or "mask" not in value:
                continue
            mask = _to_bool(np.squeeze(_as_numpy(value["mask"])))
            score = float(value.get("score", value.get("prob", 1.0)))
            instances.append((int(key), score, mask))
        if instances:
            return instances
    keys = list(outputs.keys()) if isinstance(outputs,
                                              dict) else type(outputs).__name__
    raise RuntimeError(f"無法解讀 SAM 3 輸出。keys={keys}")


def _open_session(resource_path, args):
    try:
        import torch
        import sam3.model_builder as sam3_builder
    except ImportError as exc:
        raise SystemExit(
            "無法載入 sam3。請先安裝 https://github.com/facebookresearch/sam3 ，"
            "並在 Hugging Face 取得 checkpoint 權限後執行 hf auth login。\n"
            f"詳細錯誤: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("SAM 3 需要可用的 NVIDIA CUDA GPU。")

    checkpoint = args.checkpoint or None
    if args.gpus:
        first_gpu = int(args.gpus.split(",")[0])
        torch.cuda.set_device(first_gpu)

    if _is_multiplex_checkpoint(checkpoint):
        builder = getattr(sam3_builder, "build_sam3_multiplex_video_predictor",
                          None)
        if builder is None:
            raise SystemExit("這顆 checkpoint 是 SAM 3.1 multiplex。"
                             "請把 facebookresearch/sam3 更新到 2026-03 之後再安裝。")
        print(f"載入 SAM 3.1 multiplex: {checkpoint}", flush=True)
        model_objects = min(int(args.max_objects), 16)
        if model_objects < int(args.max_objects):
            print(
                f"顯存不夠同時追 {args.max_objects} 個，追蹤器上限改成 {model_objects}。",
                flush=True,
            )
        with _safetensors_torch_load(checkpoint):
            with contextlib.redirect_stdout(io.StringIO()) as load_log:
                predictor = builder(
                    checkpoint_path=checkpoint,
                    bpe_path=_sam3_bpe_path(),
                    max_num_objects=model_objects,
                    use_fa3=False,
                    compile=bool(args.compile),
                    async_loading_frames=True,
                )
        _summarize_checkpoint_log(load_log.getvalue())
        _store_weights_fp16(predictor.model)
    else:
        kwargs = {}
        if checkpoint:
            kwargs["checkpoint_path"] = checkpoint
        if args.gpus:
            kwargs["gpus_to_use"] = [
                int(part) for part in args.gpus.split(",") if part.strip()
            ]
        if args.compile:
            kwargs["compile"] = True
        if args.async_load:
            kwargs["async_loading_frames"] = True
        if checkpoint:
            print(f"載入 SAM 3: {checkpoint}", flush=True)
        else:
            print("載入 SAM 3（未指定 checkpoint，會從 Hugging Face 下載）...", flush=True)
        predictor = sam3_builder.build_sam3_video_predictor(**kwargs)
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(resource_path),
        # 250 frames of 2K will not fit beside the model on a 10 GB GPU.
        "offload_video_to_cpu": True,
    })
    print(f"session: {response['session_id']}", flush=True)
    return predictor, response["session_id"]


def _store_weights_fp16(module):
    """Keep the checkpoint math in float16 without touching RoPE buffers."""
    import torch
    count = 0
    for param in module.parameters():
        if param.is_floating_point() and param.dtype != torch.float16:
            param.data = param.data.to(dtype=torch.float16)
            count += param.numel()
    gib = count * 2 / (1024**3)
    print(f"權重改存 float16，大約 {gib:.1f} GB。", flush=True)


def _release_cuda_cache():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summarize_checkpoint_log(text):
    """The multiplex builder prints every key. Keep one short line instead."""
    missing = None
    unexpected = None
    for line in text.splitlines():
        if line.startswith("Missing keys ("):
            missing = line.split(":", 1)[0]
        elif line.startswith("Unexpected keys ("):
            unexpected = line.split(":", 1)[0]
    if missing and "freqs_cis" in text and not unexpected:
        print(f"權重已載入。{missing} 是 RoPE 位置緩衝，checkpoint 本來就沒有。", flush=True)
    elif missing or unexpected:
        print(f"權重已載入。{missing or ''} {unexpected or ''}".strip(), flush=True)
    else:
        print("權重已載入。", flush=True)


def _sam3_bpe_path():
    """Editable installs break pkg_resources, so point at the vocab file directly."""
    import sam3
    roots = []
    if getattr(sam3, "__file__", None):
        roots.append(Path(sam3.__file__).resolve().parent)
    roots.extend(Path(entry) for entry in getattr(sam3, "__path__", []))
    name = "bpe_simple_vocab_16e6.txt.gz"
    for root in roots:
        for candidate in (root / "assets" / name,
                          root / "sam3" / "assets" / name):
            if candidate.is_file():
                return str(candidate)
    raise SystemExit("找不到 SAM 3 的 BPE 詞表 bpe_simple_vocab_16e6.txt.gz")


def _is_multiplex_checkpoint(path):
    if not path:
        return False
    name = Path(path).name.lower()
    return "multiplex" in name or "sam3.1" in name or "sam3p1" in name


def _safetensors_torch_load(path):
    """Let sam3's torch.load() read a ComfyUI .safetensors checkpoint.

    The official builder unpickles a .pt whose tensors are named
    ``detector.*`` and ``tracker.*``. ComfyUI stores the same tensors in
    safetensors. Returning ``{"model": state_dict}`` matches the unwrap in
    ``build_sam3_multiplex_video_predictor``.
    """
    from contextlib import contextmanager

    @contextmanager
    def _patch():
        if not path or not str(path).lower().endswith(".safetensors"):
            yield
            return
        import torch
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise SystemExit(
                "讀 safetensors checkpoint 需要 safetensors。請在 .venv 執行 "
                "python -m pip install safetensors") from exc

        resolved = str(Path(path).resolve())
        target = resolved.lower()
        original = torch.load
        cached = {}

        def wrapped(file, *args, **kwargs):
            candidate = file if isinstance(file, (str, Path)) else getattr(
                file, "name", None)
            if candidate is not None and str(
                    Path(candidate).resolve()).lower() == target:
                if "state" not in cached:
                    print("從 safetensors 讀取權重...", flush=True)
                    cached["state"] = {
                        "model": load_file(resolved, device="cpu")
                    }
                return cached["state"]
            return original(file, *args, **kwargs)

        torch.load = wrapped
        try:
            yield
        finally:
            torch.load = original

    return _patch()


def _close_session(predictor, session_id):
    if predictor is None:
        return
    try:
        if session_id:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id
            })
    except Exception as exc:
        print(f"關閉 session 時發生錯誤: {exc}", flush=True)
    shutdown = getattr(predictor, "shutdown", None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception as exc:
            print(f"關閉 predictor 時發生錯誤: {exc}", flush=True)


def _remove_object(predictor, session_id, obj_id):
    try:
        predictor.handle_request({
            "type": "remove_object",
            "session_id": session_id,
            "obj_id": int(obj_id),
        })
    except Exception:
        return


def _encode_one_folder(folder, exr_path, args, preview_path, sidecar=None):
    from cryptomatte_from_masks import list_mask_files

    files = list_mask_files(folder)
    items = []
    for path in files:
        coverage = load_coverage(path)
        if float(coverage.sum()) <= 0:
            continue
        items.append((path.stem, float(coverage.sum()), coverage))
    if not items:
        raise SystemExit(f"沒有有效的 mask: {folder}")
    items.sort(key=lambda item: (-item[1], item[0]))
    height, width = items[0][2].shape
    names = [name for name, _score, _coverage in items]
    manifest = build_manifest(names)
    coverages = [(name, coverage) for name, _score, coverage in items]
    id_ch, cov_ch, dropped = pack_ranks(coverages,
                                        height,
                                        width,
                                        args.ranks,
                                        shuffle=args.shuffle_ranks,
                                        seed=args.seed)
    layers, layer_names = ranks_to_layers(id_ch, cov_ch, args.crypto_name)
    meta_key = save_cryptomatte_exr(exr_path, layers, layer_names, manifest,
                                    args.crypto_name)
    if sidecar is None:
        sidecar = Path(exr_path).with_suffix(".manifest.json")
    write_manifest_sidecar(sidecar, args.crypto_name, args.ranks, manifest,
                           layer_names, meta_key, [Path(exr_path).name])
    if preview_path is not None:
        Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path),
                    render_preview(None, coverages, width, height))
    print(f"寫出 {exr_path}（{len(names)} 個物件）", flush=True)
    if dropped:
        print(f"  {dropped} 個像素超過 rank 上限", flush=True)


def _encode_sequence(subdirs, output_dir, args):
    from cryptomatte_from_masks import list_mask_files

    names = set()
    folders = []
    for folder in subdirs:
        files = list_mask_files(folder)
        if not files:
            continue
        folders.append((folder, files))
        for path in files:
            names.add(path.stem)
    manifest = build_manifest(sorted(names))
    output_dir = Path(output_dir)
    exr_dir = output_dir / "exr"
    preview_dir = output_dir / "preview"
    exr_dir.mkdir(parents=True, exist_ok=True)
    width = height = None
    written = []
    layer_names = []
    meta_key = "cryptomatte/" + __layer_hash(args.crypto_name)
    video_writer = None
    for folder, files in folders:
        items = []
        for path in files:
            coverage = load_coverage(path)
            if coverage.shape[0] == 0:
                continue
            items.append((path.stem, float(coverage.sum()), coverage))
        if not items:
            continue
        items.sort(key=lambda item: (-item[1], item[0]))
        if height is None:
            height, width = items[0][2].shape
        coverages = [(name, coverage) for name, _score, coverage in items]
        id_ch, cov_ch, _dropped = pack_ranks(coverages,
                                             height,
                                             width,
                                             args.ranks,
                                             shuffle=args.shuffle_ranks,
                                             seed=args.seed)
        layers, layer_names = ranks_to_layers(id_ch, cov_ch, args.crypto_name)
        exr_path = exr_dir / f"{folder.name}.exr"
        meta_key = save_cryptomatte_exr(exr_path, layers, layer_names,
                                        manifest, args.crypto_name)
        written.append(exr_path.name)
        if not args.no_preview:
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview = render_preview(None, coverages, width, height)
            cv2.imwrite(str(preview_dir / f"{folder.name}.png"), preview)
            if args.preview_video:
                if video_writer is None:
                    fps = args.fps or 24.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_dir / "preview.mp4"), fourcc, float(fps),
                        (width, height))
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(preview)
    if video_writer is not None:
        video_writer.release()
    write_manifest_sidecar(
        output_dir / "cryptomatte_manifest.json",
        args.crypto_name,
        args.ranks,
        manifest,
        layer_names,
        meta_key,
        written,
    )
    print(f"寫出 {len(written)} 張 EXR 到 {exr_dir}", flush=True)


def _preview_path_for_exr(exr_path, args):
    if args.no_preview:
        return None
    return Path(exr_path).with_suffix(".preview.png")


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


def _iou(a, b):
    if a.shape != b.shape:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _token(prompt, index):
    raw = re.sub(r"\s+", "_", prompt.strip().lower())
    raw = re.sub(r"[^0-9a-z_\-]+", "", raw).strip("_")
    return (raw or f"concept{index:02d}")[:48]


def _subdir_key(name):
    match = re.search(r"(\d+)$", name)
    if match:
        return (0, int(match.group(1)), name.lower())
    return (1, 0, name.lower())


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_bool(mask):
    mask = np.squeeze(mask)
    if mask.dtype == np.bool_:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask > 0


def _first_key(mapping, keys):
    for key in keys:
        if key in mapping:
            return key
    return None


def __layer_hash(crypto_name):
    from mmh3_for_cryptomatte import layer_hash
    return layer_hash(crypto_name)


def _delete_tree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)
