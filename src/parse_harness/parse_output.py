from __future__ import annotations

import re
from typing import Iterable


BLOCK_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.MULTILINE | re.DOTALL,
)


def model_to_source_bbox(
    bbox: Iterable[float],
    *,
    source_size: tuple[int, int],
    model_size: tuple[int, int] = (1664, 2048),
) -> tuple[int, int, int, int]:
    """Undo Parse's shrink-to-fit plus centered-padding image transform."""
    source_width, source_height = source_size
    model_width, model_height = model_size
    scale = min(model_width / max(1, source_width), model_height / max(1, source_height))
    if scale < 1.0:
        resized_width = int(source_width * scale)
        resized_height = int(source_height * scale)
        scale_x = resized_width / max(1, source_width)
        scale_y = resized_height / max(1, source_height)
    else:
        resized_width, resized_height = source_width, source_height
        scale_x = scale_y = 1.0
    offset_x = max(0, (model_width - resized_width) // 2)
    offset_y = max(0, (model_height - resized_height) // 2)
    x1, y1, x2, y2 = (float(value) for value in bbox)
    mapped_x = sorted(((x1 - offset_x) / scale_x, (x2 - offset_x) / scale_x))
    mapped_y = sorted(((y1 - offset_y) / scale_y, (y2 - offset_y) / scale_y))
    return (
        round(max(0.0, min(source_width, mapped_x[0]))),
        round(max(0.0, min(source_height, mapped_y[0]))),
        round(max(0.0, min(source_width, mapped_x[1]))),
        round(max(0.0, min(source_height, mapped_y[1]))),
    )


def extract_blocks(
    raw_text: str,
    *,
    source_size: tuple[int, int],
    model_size: tuple[int, int] = (1664, 2048),
) -> list[dict]:
    """Decode Parse blocks without changing their generated order."""
    model_width, model_height = model_size
    blocks: list[dict] = []
    for match in BLOCK_RE.finditer(raw_text):
        x1, y1, text, x2, y2, category = match.groups()
        normalized_bbox = [float(x1), float(y1), float(x2), float(y2)]
        model_bbox = [
            round(normalized_bbox[0] * model_width),
            round(normalized_bbox[1] * model_height),
            round(normalized_bbox[2] * model_width),
            round(normalized_bbox[3] * model_height),
        ]
        blocks.append(
            {
                "category": category,
                "text": text.replace("<tbc>", "").strip(),
                "normalized_bbox": normalized_bbox,
                "model_bbox": model_bbox,
                "source_bbox": list(
                    model_to_source_bbox(
                        model_bbox,
                        source_size=source_size,
                        model_size=model_size,
                    )
                ),
            }
        )
    return blocks
