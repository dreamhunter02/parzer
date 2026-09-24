from __future__ import annotations

import base64
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .parse_output import extract_blocks
from .render import PageItem


DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"


def generation_options() -> dict:
    return {"repetition_penalty": 1.1, "top_k": 1, "skip_special_tokens": False}


def build_message_content(*, image_url: str, prompt: str) -> list[dict]:
    image = {"type": "image_url", "image_url": {"url": image_url}}
    return [{"type": "text", "text": prompt}, image]


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAIParseClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 9000,
        timeout: float = 1800.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens

    def __call__(self, image_path: Path) -> str:
        request: dict = dict(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": build_message_content(
                        image_url=_image_data_url(image_path),
                        prompt=self.prompt,
                    ),
                }
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
            extra_body=generation_options(),
        )
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        return message.content or ""


def infer_page(page: PageItem, request_text: Callable[[Path], str]) -> dict:
    response = request_text(Path(page.image_path))
    record = asdict(page)
    record["raw_generation"] = response
    record["blocks"] = extract_blocks(
        response,
        source_size=(page.image_width, page.image_height),
    )
    return record


def _record_path(output_root: Path, page: PageItem) -> Path:
    return output_root / "raw_responses" / page.document_id / f"page_{page.page_number:04d}.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_inference(
    pages: list[PageItem],
    output_root: Path,
    request_text: Callable[[Path], str],
    *,
    concurrency: int,
    retries: int,
    resume: bool,
    retry_delay: float = 0.0,
) -> list[dict]:
    records: list[dict] = []
    pending: list[PageItem] = []
    for page in pages:
        path = _record_path(output_root, page)
        if resume and path.exists():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append(page)

    def run_one(page: PageItem) -> dict:
        for attempt in range(retries + 1):
            try:
                return infer_page(page, request_text)
            except Exception:
                if attempt >= retries:
                    raise
                if retry_delay:
                    time.sleep(retry_delay * (2**attempt))
        raise RuntimeError("unreachable")

    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(run_one, page): page for page in pending}
        for future in as_completed(futures):
            page = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "page_id": page.page_id,
                        "document_id": page.document_id,
                        "page_number": page.page_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            _write_json(_record_path(output_root, page), record)
            records.append(record)

    errors_path = output_root / "errors.jsonl"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(
        "".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors),
        encoding="utf-8",
    )
    return sorted(records, key=lambda record: (record["document_id"], int(record["page_number"])))
