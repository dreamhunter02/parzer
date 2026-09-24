from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from .assembly import write_document_outputs
from .inference import OpenAIParseClient, run_inference
from .render import render_pdfs


def run_extract(
    *,
    input_path: Path,
    output_root: Path,
    base_url: str,
    model: str,
    output_mode: str = "both",
    figure_mode: str = "crop",
    concurrency: int = 8,
    timeout: float = 1800.0,
    max_tokens: int = 9000,
    retries: int = 2,
    resume: bool = False,
    request_text: Callable[[Path], str] | None = None,
) -> dict:
    output_root = output_root.expanduser().resolve()
    pages = render_pdfs(input_path, output_root, resume=resume)
    client = request_text or OpenAIParseClient(
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    records = run_inference(
        pages,
        output_root,
        client,
        concurrency=concurrency,
        retries=retries,
        resume=resume,
        retry_delay=1.0,
    )
    by_document: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_document[str(record["document_id"])].append(record)
    for document_records in by_document.values():
        write_document_outputs(
            document_records,
            output_root,
            output_mode=output_mode,
            figure_mode=figure_mode,
        )
    return {
        "documents": len({page.document_id for page in pages}),
        "pages_total": len(pages),
        "pages_succeeded": len(records),
        "pages_failed": len(pages) - len(records),
        "output": str(output_root),
    }
