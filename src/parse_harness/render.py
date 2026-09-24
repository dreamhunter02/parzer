from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageItem:
    page_id: str
    document_id: str
    source_pdf: str
    page_number: int
    image_path: str
    image_width: int
    image_height: int
    render_scale: float


def discover_pdfs(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    return sorted(
        (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )


def render_scale(
    width_points: float,
    height_points: float,
    *,
    max_width: int = 1664,
    max_height: int = 2048,
) -> float:
    """Scale a PDF page to the largest Parse 2.0-supported RGB canvas."""
    return min(max_width / width_points, max_height / height_points)


def _safe_stem(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return value or "document"


def _document_ids(paths: list[Path]) -> dict[Path, str]:
    counts: dict[str, int] = {}
    result: dict[Path, str] = {}
    for path in paths:
        stem = _safe_stem(path)
        counts[stem] = counts.get(stem, 0) + 1
        result[path] = stem
    duplicates = {stem for stem, count in counts.items() if count > 1}
    for path, stem in list(result.items()):
        if stem in duplicates:
            suffix = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
            result[path] = f"{stem}-{suffix}"
    return result


def render_pdfs(input_path: Path, output_dir: Path, *, resume: bool = False) -> list[PageItem]:
    import pypdfium2 as pdfium
    from PIL import Image

    pdf_paths = discover_pdfs(input_path)
    document_ids = _document_ids(pdf_paths)
    pages_root = output_dir / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)
    items: list[PageItem] = []
    for pdf_path in pdf_paths:
        document_id = document_ids[pdf_path]
        document_dir = pages_root / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                width_points, height_points = page.get_size()
                scale = render_scale(width_points, height_points)
                image_path = document_dir / f"page_{page_index + 1:04d}.png"
                if resume and image_path.exists():
                    with Image.open(image_path) as existing:
                        image_width, image_height = existing.size
                else:
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil().convert("RGB")
                    image.save(image_path, format="PNG")
                    image_width, image_height = image.size
                items.append(
                    PageItem(
                        page_id=f"{document_id}:page_{page_index + 1:04d}",
                        document_id=document_id,
                        source_pdf=str(pdf_path),
                        page_number=page_index + 1,
                        image_path=str(image_path),
                        image_width=image_width,
                        image_height=image_height,
                        render_scale=scale,
                    )
                )
        finally:
            pdf.close()
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    return items
