from __future__ import annotations

import re
from pathlib import Path


FIGURE_CATEGORIES = {"picture", "chart", "figure"}


def _safe_category(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")
    return cleaned or "figure"


def _write_crop(block: dict, page: dict, output_root: Path, figure_index: int) -> tuple[Path, str] | None:
    from PIL import Image

    image_path = page.get("image_path")
    bbox = block.get("source_bbox")
    if not image_path or not bbox or len(bbox) != 4:
        return None
    with Image.open(image_path) as image:
        x1, y1, x2, y2 = (round(float(value)) for value in bbox)
        x1, x2 = sorted((max(0, min(image.width, x1)), max(0, min(image.width, x2))))
        y1, y2 = sorted((max(0, min(image.height, y1)), max(0, min(image.height, y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        category = _safe_category(str(block.get("category", "figure")))
        document_id = str(page["document_id"])
        name = f"page_{int(page['page_number']):04d}_{category}_{figure_index:02d}.png"
        crop_path = output_root / "assets" / document_id / name
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x1, y1, x2, y2)).save(crop_path, format="PNG")
    relative_link = f"../../assets/{document_id}/{name}"
    return crop_path, relative_link


def _raw_page_markdown(page: dict) -> str:
    return "\n\n".join(
        str(block.get("text", "")).strip()
        for block in page.get("blocks", [])
        if str(block.get("text", "")).strip()
    )


def _zoom_page_markdown(page: dict, output_root: Path, figure_mode: str) -> str:
    parts: list[str] = []
    figure_index = 0
    for block in page.get("blocks", []):
        text = str(block.get("text", "")).strip()
        category = str(block.get("category", ""))
        if category.lower() not in FIGURE_CATEGORIES:
            if text:
                parts.append(text)
            continue
        figure_index += 1
        crop_written = False
        if figure_mode in {"crop", "both"}:
            crop = _write_crop(block, page, output_root, figure_index)
            if crop is not None:
                _, relative_link = crop
                parts.append(f"![{category}]({relative_link})")
                crop_written = True
        if figure_mode in {"description", "both"} and text:
            parts.append(text)
        elif figure_mode == "crop" and not crop_written and text:
            parts.append(text)
    return "\n\n".join(parts)


def write_document_outputs(
    page_records: list[dict],
    output_root: Path,
    *,
    output_mode: str,
    figure_mode: str,
) -> dict[str, Path]:
    if output_mode not in {"raw", "zoom", "both"}:
        raise ValueError(f"Unsupported output mode: {output_mode}")
    if figure_mode not in {"crop", "description", "both"}:
        raise ValueError(f"Unsupported figure mode: {figure_mode}")
    if not page_records:
        return {}
    document_ids = {str(page["document_id"]) for page in page_records}
    if len(document_ids) != 1:
        raise ValueError("write_document_outputs accepts one document at a time")
    document_id = document_ids.pop()
    pages = sorted(page_records, key=lambda page: int(page["page_number"]))
    paths: dict[str, Path] = {}
    if output_mode in {"raw", "both"}:
        path = output_root / "markdown" / "raw" / f"{document_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n\n".join(filter(None, (_raw_page_markdown(page) for page in pages)))
        path.write_text(content.rstrip() + "\n" if content else "", encoding="utf-8")
        paths["raw"] = path
    if output_mode in {"zoom", "both"}:
        path = output_root / "markdown" / "zoom" / f"{document_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n\n".join(
            filter(None, (_zoom_page_markdown(page, output_root, figure_mode) for page in pages))
        )
        path.write_text(content.rstrip() + "\n" if content else "", encoding="utf-8")
        paths["zoom"] = path
    return paths
