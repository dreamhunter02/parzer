from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_extract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct Nemotron Parse 2.0 PDF evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="Render PDFs, run Parse, and write Markdown")
    extract.add_argument("--input", required=True, type=Path, help="PDF file or directory of PDFs")
    extract.add_argument("--output", required=True, type=Path, help="Output directory")
    extract.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL")
    extract.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-Parse-2.0",
        help="Parse 2.0 model identifier served by the vLLM endpoint",
    )
    extract.add_argument("--output-mode", choices=["raw", "zoom", "both"], default="both")
    extract.add_argument("--figure-mode", choices=["crop", "description", "both"], default="crop")
    extract.add_argument("--concurrency", type=int, default=8)
    extract.add_argument("--timeout", type=float, default=1800.0)
    extract.add_argument("--max-tokens", type=int, default=9000)
    extract.add_argument("--retries", type=int, default=2)
    extract.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        summary = run_extract(
            input_path=args.input,
            output_root=args.output,
            base_url=args.base_url,
            model=args.model,
            output_mode=args.output_mode,
            figure_mode=args.figure_mode,
            concurrency=args.concurrency,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            retries=args.retries,
            resume=args.resume,
        )
        print(json.dumps(summary, indent=2))
