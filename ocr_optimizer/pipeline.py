"""OCR pipeline orchestrator."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import PipelineConfig, load_config
from .cropper import crop_regions
from .merge import MergedResult, merge_results
from .ocr_client import OCRResult, OllamaOCR
from .preprocessing import preprocess_image


@dataclass
class PipelineResult:
    merged: MergedResult
    pass_results: list[OCRResult] = field(default_factory=list)
    total_elapsed: float = 0.0
    num_passes: int = 0

    @property
    def text(self) -> str:
        return self.merged.text

    @property
    def confidence(self) -> float:
        return self.merged.confidence


def run_pipeline(
    image: str | Path | Image.Image,
    config: PipelineConfig | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Run the full OCR optimization pipeline.

    Args:
        image: Path to image file or PIL Image object.
        config: Pipeline configuration. Uses defaults if None.
        verbose: Print progress to stderr.

    Returns:
        PipelineResult with merged text, confidence, and per-pass details.
    """
    if config is None:
        config = load_config()

    def log(msg: str):
        if verbose:
            print(msg, file=sys.stderr)

    start = time.time()

    # 1. Load image
    if isinstance(image, (str, Path)):
        img = Image.open(image)
        log(f"Loaded: {image} ({img.width}x{img.height})")
    else:
        img = image

    # 2. Preprocess
    log(f"Preprocessing: upscale={config.preprocess.upscale_factor}x, "
        f"contrast={config.preprocess.contrast}")
    processed = preprocess_image(img, config.preprocess)
    log(f"  Result: {processed.width}x{processed.height}")

    # 3. Generate crop regions (full image + crops)
    regions = crop_regions(processed, config.crop)
    region_names = ["full"]
    if config.crop.strategy == "columns":
        for i in range(config.crop.num_columns):
            region_names.append(f"col_{i}")
    elif config.crop.strategy == "grid":
        n = config.crop.num_columns
        for r in range(n):
            for c in range(n):
                region_names.append(f"grid_{r}_{c}")
    # Pad names if needed
    while len(region_names) < len(regions):
        region_names.append(f"region_{len(region_names)}")

    log(f"Regions: {len(regions)} ({', '.join(region_names[:len(regions)])})")

    # 4. Run OCR on each region, multiple passes
    client = OllamaOCR(config.ollama)
    all_results: list[OCRResult] = []

    for region_idx, (region_img, rname) in enumerate(zip(regions, region_names)):
        for pass_num in range(config.passes_per_crop):
            label = f"{rname}_p{pass_num}" if config.passes_per_crop > 1 else rname
            log(f"  OCR: {label}...")

            result = client.run(region_img, region_name=label)
            all_results.append(result)

            log(f"    {result.elapsed:.1f}s, {result.eval_count} tokens, "
                f"{result.tokens_per_sec:.0f} tok/s, {len(result.text)} chars")

    # 5. Merge results
    log(f"Merging {len(all_results)} results...")
    merged = merge_results(
        all_results,
        similarity_threshold=config.merge_similarity_threshold,
        min_line_length=config.min_line_length,
    )
    total_elapsed = time.time() - start

    log(f"Done: {len(merged.text)} chars, confidence={merged.confidence:.1%}, "
        f"total={total_elapsed:.1f}s")

    return PipelineResult(
        merged=merged,
        pass_results=all_results,
        total_elapsed=total_elapsed,
        num_passes=len(all_results),
    )
