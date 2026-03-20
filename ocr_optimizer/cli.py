"""CLI interface for OCR Optimizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig, load_config


def _add_ocr_args(parser: argparse.ArgumentParser):
    """Add common OCR pipeline arguments."""
    parser.add_argument("-c", "--config", help="Path to config YAML file")
    parser.add_argument("-m", "--model", help="Override Ollama model name")
    parser.add_argument("--upscale", type=int, help="Upscale factor (default: 2)")
    parser.add_argument("--contrast", type=float, help="Contrast factor (default: 1.3)")
    parser.add_argument("--columns", type=int, help="Number of column crops (0=no cropping)")
    parser.add_argument("--passes", type=int, help="OCR passes per crop region")
    parser.add_argument("--overlap", type=int, help="Crop overlap in pixels")


def _apply_ocr_overrides(args, config: PipelineConfig):
    """Apply CLI overrides to config."""
    if args.model:
        config.ollama.model = args.model
    if args.upscale is not None:
        config.preprocess.upscale_factor = args.upscale
    if args.contrast is not None:
        config.preprocess.contrast = args.contrast
    if args.columns is not None:
        if args.columns == 0:
            config.crop.strategy = "none"
        else:
            config.crop.strategy = "columns"
            config.crop.num_columns = args.columns
    if args.passes is not None:
        config.passes_per_crop = args.passes
    if args.overlap is not None:
        config.crop.overlap_px = args.overlap


def cmd_ocr(args):
    """Run OCR on a single image."""
    from .pipeline import run_pipeline

    config = load_config(args.config)
    _apply_ocr_overrides(args, config)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    verbose = args.verbose and not args.quiet
    result = run_pipeline(image_path, config, verbose=verbose)

    if args.json:
        output = {
            "text": result.text,
            "confidence": round(result.confidence, 4),
            "total_elapsed": round(result.total_elapsed, 2),
            "num_passes": result.num_passes,
            "passes": [
                {
                    "region": r.region_name,
                    "chars": len(r.text),
                    "elapsed": round(r.elapsed, 2),
                    "tokens": r.eval_count,
                    "tokens_per_sec": round(r.tokens_per_sec, 1),
                }
                for r in result.pass_results
            ],
            "lines": [
                {
                    "text": ml.text,
                    "confidence": round(ml.confidence, 4),
                    "source": ml.source,
                }
                for ml in result.merged.lines
            ],
        }
        text = json.dumps(output, ensure_ascii=False, indent=2)
    else:
        text = result.text
        if not args.quiet:
            print(f"--- OCR Optimizer ---", file=sys.stderr)
            print(f"Passes: {result.num_passes} | "
                  f"Time: {result.total_elapsed:.1f}s | "
                  f"Confidence: {result.confidence:.1%} | "
                  f"Chars: {len(result.text)}", file=sys.stderr)
            for r in result.pass_results:
                print(f"  {r.region_name}: {r.elapsed:.1f}s, "
                      f"{r.eval_count} tok, {len(r.text)} chars", file=sys.stderr)
            print(f"---", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        if not args.quiet:
            print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(text)


def cmd_train(args):
    """Train neural-memory brain from OCR'd images."""
    from .trainer import train_brain

    config = load_config(args.config)
    _apply_ocr_overrides(args, config)

    source = Path(args.source)
    if not source.exists():
        print(f"Error: path not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    result = train_brain(
        source=source,
        brain_db=args.brain_db,
        domain_tag=args.tag,
        ocr_config=config,
        recursive=not args.no_recursive,
        verbose=True,
        min_confidence=args.min_confidence,
        min_chars=args.min_chars,
    )

    # Print final report
    print(f"\nTraining Report:")
    print(f"  Images processed: {result.total_images}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Total chars: {result.total_chars:,}")
    print(f"  Fibers created: {result.encoded_fibers}")
    print(f"  Time: {result.total_elapsed:.1f}s")

    if result.errors:
        print(f"\nErrors:")
        for path, err in result.errors:
            print(f"  {path}: {err}")

    if args.json:
        report = {
            "total_images": result.total_images,
            "successful": result.successful,
            "failed": result.failed,
            "total_chars": result.total_chars,
            "encoded_fibers": result.encoded_fibers,
            "total_elapsed": round(result.total_elapsed, 2),
            "errors": [{"file": p, "error": e} for p, e in result.errors],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="ocr-optimize",
        description="Multi-pass OCR optimizer with neural-memory training",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- ocr subcommand ---
    ocr_parser = subparsers.add_parser("ocr", help="Run OCR on a single image")
    ocr_parser.add_argument("image", help="Path to image file")
    _add_ocr_args(ocr_parser)
    ocr_parser.add_argument("-o", "--output", help="Write result to file")
    ocr_parser.add_argument("--json", action="store_true", help="Output JSON with metadata")
    ocr_parser.add_argument("-v", "--verbose", action="store_true", help="Show progress")
    ocr_parser.add_argument("-q", "--quiet", action="store_true", help="Only output OCR text")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="OCR images and train neural-memory brain")
    train_parser.add_argument("source", help="Image file or directory of images")
    _add_ocr_args(train_parser)
    train_parser.add_argument("--brain-db", help="Path to brain .db file (default: ~/.neuralmemory/brains/default.db)")
    train_parser.add_argument("--tag", default="ocr-document", help="Domain tag for trained memories (default: ocr-document)")
    train_parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum OCR confidence to accept (default: 0.3)")
    train_parser.add_argument("--min-chars", type=int, default=20, help="Minimum text length to accept (default: 20)")
    train_parser.add_argument("--no-recursive", action="store_true", help="Don't scan subdirectories")
    train_parser.add_argument("--json", action="store_true", help="Output JSON report")

    args = parser.parse_args()

    # Backward compat: if no subcommand, treat first positional as image for OCR
    if args.command is None:
        # Re-parse as legacy single-image mode
        legacy = argparse.ArgumentParser(prog="ocr-optimize")
        legacy.add_argument("image", help="Path to image file")
        _add_ocr_args(legacy)
        legacy.add_argument("-o", "--output", help="Write result to file")
        legacy.add_argument("--json", action="store_true")
        legacy.add_argument("-v", "--verbose", action="store_true")
        legacy.add_argument("-q", "--quiet", action="store_true")
        args = legacy.parse_args()
        cmd_ocr(args)
    elif args.command == "ocr":
        cmd_ocr(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
