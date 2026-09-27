"""Pack per-object mattes into a Cryptomatte EXR.

The ID hash and the EXR metadata match the 2024 scripts in this repo
(``mmh3_for_cryptomatte.py`` and ``write_multi_RGBA_exr.py``):

- MurmurHash3 32-bit, stored as raw float32 bits
- metadata name ``ViewLayer.CryptoObject`` by default
- layers ``{name}00``, ``{name}01``, ... with RGBA = id, coverage, id, coverage

Rank 0 keeps the highest-score matte. Where a later matte overlaps pixels
that are already taken, it is written into the next free rank instead of
replacing the earlier ID.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from mmh3_for_cryptomatte import hash_object_name, layer_hash

try:
    import Imath
    import OpenEXR
except ImportError as exc:  # pragma: no cover - import guard for a clear CLI error
    Imath = None
    OpenEXR = None
    _OPENEXR_IMPORT_ERROR = exc
else:
    _OPENEXR_IMPORT_ERROR = None

MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def require_openexr():
    if OpenEXR is None:
        raise SystemExit("無法載入 OpenEXR / Imath，不能寫 Cryptomatte EXR。\n"
                         f"詳細錯誤: {_OPENEXR_IMPORT_ERROR}")


def build_manifest(names):
    manifest = {}
    for name in names:
        manifest.update(hash_object_name(name)["fk"])
    return manifest


def id_float32(name):
    """Float32 value whose bits are the Cryptomatte ID of ``name``."""
    return np.float32(hash_object_name(name)["fff"])


def pack_ranks(items, height, width, ranks, shuffle=False, seed=0):
    """Place ``(name, coverage)`` mattes into rank buffers.

    ``items`` must already be sorted so the matte that should win rank 0
    comes first. ``coverage`` is float32 ``HxW`` in ``0..1``, or a boolean mask.
    """
    if ranks < 2 or ranks % 2 != 0:
        raise ValueError("ranks must be an even integer >= 2")

    id_ch = np.zeros((ranks, height, width), dtype=np.float32)
    cov_ch = np.zeros((ranks, height, width), dtype=np.float32)
    dropped = 0

    for name, coverage in items:
        coverage = _as_coverage(coverage, height, width)
        if not np.any(coverage > 0):
            continue
        value = id_float32(name)
        remaining = coverage
        placed = False
        for rank in range(ranks):
            slot = (remaining > 0) & (cov_ch[rank] <= 0)
            if not np.any(slot):
                continue
            id_ch[rank][slot] = value
            cov_ch[rank][slot] = remaining[slot]
            remaining = remaining.copy()
            remaining[slot] = 0
            placed = True
            if not np.any(remaining > 0):
                break
        if placed and np.any(remaining > 0):
            dropped += int(np.count_nonzero(remaining > 0))

    if shuffle:
        _shuffle_tied_ranks(id_ch, cov_ch, seed)
    return id_ch, cov_ch, dropped


def ranks_to_layers(id_ch, cov_ch, crypto_name):
    ranks = id_ch.shape[0]
    layers = []
    layer_names = []
    for layer_index in range(ranks // 2):
        rgba = np.zeros((id_ch.shape[1], id_ch.shape[2], 4), dtype=np.float32)
        rgba[:, :, 0] = id_ch[layer_index * 2]
        rgba[:, :, 1] = cov_ch[layer_index * 2]
        rgba[:, :, 2] = id_ch[layer_index * 2 + 1]
        rgba[:, :, 3] = cov_ch[layer_index * 2 + 1]
        layers.append(rgba)
        layer_names.append(f"{crypto_name}{layer_index:02d}")
    return layers, layer_names


def save_cryptomatte_exr(output_path, layers, layer_names, manifest,
                         crypto_name):
    """Write a multi-layer float32 EXR with Cryptomatte metadata."""
    require_openexr()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width, _ = layers[0].shape
    header = OpenEXR.Header(width, height)
    try:
        header["compression"] = Imath.Compression(
            Imath.Compression.ZIP_COMPRESSION)
    except Exception:
        pass

    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header["channels"] = {}
    for layer_name in layer_names:
        for channel in "RGBA":
            header["channels"][f"{layer_name}.{channel}"] = float_chan

    # Hash key is the metadata name, without the 00/01/02 rank suffix.
    # This is the same key the 2024 writer used for ViewLayer.CryptoObject.
    meta_id = layer_hash(crypto_name)
    prefix = "cryptomatte/" + meta_id
    manifest_json = json.dumps(manifest,
                               ensure_ascii=True,
                               separators=(",", ":"),
                               sort_keys=True)
    _set_header_string(header, prefix + "/conversion", "uint32_to_float32")
    _set_header_string(header, prefix + "/manifest", manifest_json)
    _set_header_string(header, prefix + "/name", crypto_name)
    _set_header_string(header, prefix + "/hash", "MurmurHash3_32")

    pixel_data = {}
    for image, layer_name in zip(layers, layer_names):
        image = np.ascontiguousarray(image, dtype=np.float32)
        for channel_index, channel in enumerate("RGBA"):
            pixel_data[
                f"{layer_name}.{channel}"] = image[:, :,
                                                   channel_index].tobytes()

    exr_file = OpenEXR.OutputFile(str(output_path), header)
    exr_file.writePixels(pixel_data)
    exr_file.close()
    return "cryptomatte/" + meta_id


def write_manifest_sidecar(path, crypto_name, ranks, manifest, layer_names,
                           meta_key, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cryptomatte_name": crypto_name,
        "hash_method": "MurmurHash3_32",
        "conversion": "uint32_to_float32",
        "metadata_key": meta_key,
        "ranks": ranks,
        "layer_names": layer_names,
        "manifest": manifest,
        "frames": frames,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) +
        "\n",
        encoding="utf-8")


def render_preview(base_bgr, items, width, height):
    """Color each matte with a stable color derived from its Cryptomatte ID."""
    if base_bgr is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = base_bgr
        if canvas.shape[0] != height or canvas.shape[1] != width:
            canvas = cv2.resize(canvas, (width, height),
                                interpolation=cv2.INTER_AREA)
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        canvas = canvas.copy()

    area_limit = max(64, int(0.002 * width * height))
    for name, coverage in items:
        coverage = _as_coverage(coverage, height, width)
        mask = coverage > 0.05
        if not np.any(mask):
            continue
        color = np.array(_preview_bgr(name), dtype=np.float32)
        region = canvas[mask].astype(np.float32)
        canvas[mask] = np.clip(region * 0.35 + color * 0.65, 0,
                               255).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, _preview_bgr(name), 1,
                         cv2.LINE_AA)
        area = int(np.count_nonzero(mask))
        if area >= area_limit:
            ys, xs = np.nonzero(mask)
            cx = int(xs.mean())
            cy = int(ys.mean())
            cv2.putText(
                canvas,
                name,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def load_coverage(path):
    """Load a mask image as a float32 coverage map in 0..1."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"讀不到 mask: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if int(alpha.min()) < _opaque_max(alpha.dtype) and int(
                alpha.max()) > 0:
            return _unit_float(alpha)
        image = image[:, :, :3]
    if image.ndim == 3:
        image = image.max(axis=2)
    return _unit_float(image)


def list_mask_files(folder):
    folder = Path(folder)
    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def _as_coverage(coverage, height, width):
    coverage = np.asarray(coverage)
    if coverage.shape != (height, width):
        raise ValueError(f"mask 尺寸是 {coverage.shape}，這一幀是 {(height, width)}")
    if coverage.dtype == np.bool_:
        return coverage.astype(np.float32)
    coverage = coverage.astype(np.float32)
    peak = float(np.max(coverage)) if coverage.size else 0.0
    if peak > 1.0:
        coverage = coverage / 255.0
    return np.clip(coverage, 0.0, 1.0)


def _unit_float(channel):
    if np.issubdtype(channel.dtype, np.floating):
        values = channel.astype(np.float32)
        return np.clip(values, 0.0, 1.0)
    if channel.dtype == np.uint16:
        return channel.astype(np.float32) / 65535.0
    return channel.astype(np.float32) / 255.0


def _opaque_max(dtype):
    if dtype == np.uint16:
        return 65535
    return 255


def push_secondary_ids_back(id_ch, cov_ch, seed, start_rank=2):
    """Keep rank 0. Park every other ID from ``start_rank`` onward.

    Ranks 0–1 are layer 00, 2–3 are layer 01, 4–5 are layer 02.
    ``start_rank=4`` puts overlaps on the third layer so the first two
    layers only carry the primary object.
    """
    ranks = id_ch.shape[0]
    if ranks <= start_rank:
        return
    rng = np.random.default_rng(seed)
    secondary_id = id_ch[1:].copy()
    secondary_cov = cov_ch[1:].copy()
    keys = rng.random(secondary_id.shape, dtype=np.float32)
    keys = np.where(secondary_cov > 0, keys, np.float32(np.inf))
    order = np.argsort(keys, axis=0)
    sorted_id = np.take_along_axis(secondary_id, order, axis=0)
    sorted_cov = np.take_along_axis(secondary_cov, order, axis=0)
    id_ch[1:] = 0
    cov_ch[1:] = 0
    count = min(sorted_id.shape[0], ranks - start_rank)
    id_ch[start_rank:start_rank + count] = sorted_id[:count]
    cov_ch[start_rank:start_rank + count] = sorted_cov[:count]


def _shuffle_tied_ranks(id_ch, cov_ch, seed):
    """Randomize rank order where coverages tie.

    Hard masks all store coverage 1, so the first object would always sit in
    rank 0 and a VFX picker would only hit that one. Empty ranks stay empty,
    and a pixel with a single object stays on rank 0.
    """
    ranks = id_ch.shape[0]
    rng = np.random.default_rng(seed)
    occupied = cov_ch > 0
    keys = rng.random((ranks, ) + id_ch.shape[1:], dtype=np.float32)
    keys = np.where(occupied, keys, np.float32(np.inf))
    order = np.argsort(keys, axis=0)
    id_ch[:] = np.take_along_axis(id_ch, order, axis=0)
    cov_ch[:] = np.take_along_axis(cov_ch, order, axis=0)


def _preview_bgr(name):
    hex_id = hash_object_name(name)["hash_hex"]
    value = int(hex_id, 16)
    red = (value >> 16) & 255
    green = (value >> 8) & 255
    blue = value & 255
    if red + green + blue < 96:
        red, green, blue = 255 - red, 255 - green, 255 - blue
    return int(blue), int(green), int(red)


def _set_header_string(header, key, text):
    # The 2024 writer stored metadata as UTF-8 bytes. Keep that, and fall
    # back to a Python str if this OpenEXR build rejects bytes.
    try:
        header[key] = text.encode("utf-8")
    except Exception:
        header[key] = text


def self_check():
    """Write a tiny EXR and read the rank-0 ID back."""
    require_openexr()
    height, width = 4, 4
    coverage_a = np.ones((height, width), dtype=np.float32)
    coverage_b = np.zeros((height, width), dtype=np.float32)
    coverage_b[:, :2] = 1.0
    items = [("alpha", coverage_a), ("beta", coverage_b)]
    manifest = build_manifest(["alpha", "beta"])
    id_ch, cov_ch, dropped = pack_ranks(items, height, width, ranks=4)
    if dropped != 0:
        raise AssertionError(dropped)
    if not np.all(id_ch[0] == id_float32("alpha")):
        raise AssertionError("rank 0 should be alpha")
    if not np.all(id_ch[1][:, :2] == id_float32("beta")):
        raise AssertionError("overlap should fall through to rank 1")
    if np.any(id_ch[1][:, 2:] != 0):
        raise AssertionError("non-overlap should stay empty on rank 1")

    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[:, :2] = 1.0
    right[:, 2:] = 1.0
    split_ids, split_cov, split_dropped = pack_ranks([("alpha", left),
                                                      ("beta", right)],
                                                     height,
                                                     width,
                                                     ranks=4)
    if split_dropped != 0:
        raise AssertionError(split_dropped)
    if not np.all(split_ids[0][:, :2] == id_float32("alpha")):
        raise AssertionError("non-overlapping mattes should share rank 0")
    if not np.all(split_ids[0][:, 2:] == id_float32("beta")):
        raise AssertionError("the other matte should also sit on rank 0")
    if np.any(split_cov[1] != 0):
        raise AssertionError(
            "rank 1 should stay empty when mattes do not overlap")

    layers, layer_names = ranks_to_layers(id_ch, cov_ch,
                                          "ViewLayer.CryptoObject")
    output = Path(__file__).resolve().parent / "_self_check_cryptomatte.exr"
    meta_key = save_cryptomatte_exr(output, layers, layer_names, manifest,
                                    "ViewLayer.CryptoObject")
    exr = OpenEXR.InputFile(str(output))
    header = exr.header()
    header_keys = {str(key) for key in header.keys()}
    if not any(meta_key in key for key in header_keys):
        raise AssertionError(f"missing {meta_key} in {header_keys}")
    raw = exr.channel("ViewLayer.CryptoObject00.R",
                      Imath.PixelType(Imath.PixelType.FLOAT))
    exr.close()
    restored = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    if not np.all(restored == id_float32("alpha")):
        raise AssertionError("EXR round-trip changed the Cryptomatte ID bits")
    output.unlink(missing_ok=True)
    print("cryptomatte self-check ok")


if __name__ == "__main__":
    self_check()
