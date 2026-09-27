"""Text-concept presets for SAM 3 "segment the scene" tracking.

SAM 3 finds every instance of one short noun phrase, then tracks those
instances. It does not have SAM 1's class-agnostic grid. A preset is just
an ordered list of phrases. Each phrase is one full tracking pass.
"""

from pathlib import Path

PRESETS = {
    "everything": [
        "person",
        "face",
        "hair",
        "clothing",
        "hand",
        "shoe",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "animal",
        "dog",
        "cat",
        "bird",
        "chair",
        "table",
        "sofa",
        "bag",
        "bottle",
        "cup",
        "phone",
        "laptop",
        "book",
        "plant",
        "tree",
        "flower",
        "building",
        "door",
        "window",
        "sign",
        "sky",
        "ground",
        "road",
        "water",
    ],
    "fast": [
        "person",
        "clothing",
        "car",
        "animal",
        "chair",
        "plant",
        "building",
        "sky",
        "ground",
    ],
    "people": [
        "person",
        "face",
        "hair",
        "clothing",
        "hand",
        "shoe",
    ],
    "vehicles": [
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
    ],
    "animals": [
        "animal",
        "dog",
        "cat",
        "bird",
    ],
    "props": [
        "chair",
        "table",
        "sofa",
        "bag",
        "bottle",
        "cup",
        "phone",
        "laptop",
        "book",
    ],
    "environment": [
        "plant",
        "tree",
        "flower",
        "building",
        "door",
        "window",
        "sign",
        "sky",
        "ground",
        "road",
        "water",
    ],
}


def read_prompt_file(path):
    prompts = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def unique_prompts(prompts):
    seen = set()
    ordered = []
    for prompt in prompts:
        cleaned = " ".join(prompt.split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def resolve_prompts(preset, prompts_csv, prompts_file):
    if prompts_csv:
        raw = [part for part in prompts_csv.split(",")]
    elif prompts_file:
        raw = read_prompt_file(prompts_file)
    else:
        try:
            raw = PRESETS[preset]
        except KeyError as exc:
            known = ", ".join(sorted(PRESETS))
            raise SystemExit(f"未知的 preset {preset!r}。可用: {known}") from exc
    prompts = unique_prompts(raw)
    if not prompts:
        raise SystemExit("沒有任何文字概念。請用 --prompts、--prompts-file 或 --preset。")
    return prompts
