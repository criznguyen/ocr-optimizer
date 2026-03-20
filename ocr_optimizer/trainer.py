"""Train neural-memory brain from OCR'd document images.

Pipeline:
1. Scan directory for image files
2. Run each through ocr-optimize pipeline (sync)
3. Encode extracted text into neural-memory brain as pinned knowledge (async)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig, load_config
from .pipeline import run_pipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

DEFAULT_BRAIN_DB = Path.home() / ".neuralmemory" / "brains" / "default.db"


@dataclass
class TrainResult:
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    total_chars: int = 0
    total_elapsed: float = 0.0
    encoded_fibers: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _find_images(path: Path, recursive: bool = True) -> list[Path]:
    """Find all image files in a directory."""
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    pattern = "**/*" if recursive else "*"
    images = []
    for f in path.glob(pattern):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return sorted(images)


async def _train_async(
    images: list[Path],
    brain_db: Path,
    domain_tag: str,
    ocr_config: PipelineConfig,
    verbose: bool,
    min_confidence: float,
    min_chars: int,
) -> TrainResult:
    """Core training loop — runs inside asyncio.run()."""
    from neural_memory import Brain, BrainConfig, MemoryEncoder
    from neural_memory.storage.sqlite_store import SQLiteStorage

    def log(msg: str):
        if verbose:
            print(msg, file=sys.stderr)

    # Setup storage and brain
    brain_db.parent.mkdir(parents=True, exist_ok=True)
    storage = SQLiteStorage(str(brain_db))
    await storage.initialize()

    brain = await storage.find_brain_by_name("default")
    if brain is None:
        brain = Brain.create("default", config=BrainConfig())
        await storage.save_brain(brain)

    storage.set_brain(brain.id)
    config = brain.config or BrainConfig()
    encoder = MemoryEncoder(storage=storage, config=config)

    log(f"Brain ready: {brain.name} (id={brain.id[:8]}...)")

    result = TrainResult(total_images=len(images))
    start = time.time()

    for i, img_path in enumerate(images, 1):
        log(f"\n[{i}/{len(images)}] {img_path.name}")

        try:
            # OCR (sync — runs Ollama HTTP calls)
            ocr_result = run_pipeline(img_path, ocr_config, verbose=False)
            text = ocr_result.text.strip()
            confidence = ocr_result.confidence

            log(f"  OCR: {len(text)} chars, confidence={confidence:.1%}, "
                f"time={ocr_result.total_elapsed:.1f}s")

            if confidence < min_confidence:
                log(f"  SKIP: confidence {confidence:.1%} < {min_confidence:.1%}")
                result.failed += 1
                result.errors.append((str(img_path), f"low confidence: {confidence:.1%}"))
                continue

            if len(text) < min_chars:
                log(f"  SKIP: too short ({len(text)} chars < {min_chars})")
                result.failed += 1
                result.errors.append((str(img_path), f"too short: {len(text)} chars"))
                continue

            # Prepare content with source metadata
            content = (
                f"[Source: {img_path.name} | OCR confidence: {confidence:.0%}]\n\n"
                f"{text}"
            )

            # Encode into brain (async)
            await encoder.encode(
                content=content,
                timestamp=datetime.now(),
                metadata={
                    "source_file": str(img_path),
                    "source_type": "ocr",
                    "ocr_confidence": round(confidence, 4),
                    "ocr_chars": len(text),
                    "ocr_passes": ocr_result.num_passes,
                },
                tags={domain_tag, "ocr", "trained", img_path.suffix.lstrip(".")},
                language="auto",
                skip_conflicts=True,
                skip_time_neurons=True,
                initial_stage="episodic",
                salience_ceiling=0.7,
            )

            result.successful += 1
            result.total_chars += len(text)
            result.encoded_fibers += 1
            log(f"  Encoded: {len(text)} chars -> brain")

        except Exception as e:
            result.failed += 1
            result.errors.append((str(img_path), str(e)))
            log(f"  ERROR: {e}")

    result.total_elapsed = time.time() - start

    # Print stats from brain
    brain = await storage.get_brain(brain.id)
    log(f"\n{'=' * 50}")
    log(f"Training complete: {result.successful}/{result.total_images} images")
    log(f"  Total chars encoded: {result.total_chars:,}")
    log(f"  Fibers created: {result.encoded_fibers}")
    log(f"  Brain neurons: {brain.neuron_count}")
    log(f"  Brain synapses: {brain.synapse_count}")
    log(f"  Brain fibers: {brain.fiber_count}")
    log(f"  Failed: {result.failed}")
    log(f"  Time: {result.total_elapsed:.1f}s")

    return result


def train_brain(
    source: str | Path,
    brain_db: str | Path | None = None,
    domain_tag: str = "ocr-document",
    ocr_config: PipelineConfig | None = None,
    recursive: bool = True,
    verbose: bool = False,
    min_confidence: float = 0.3,
    min_chars: int = 20,
) -> TrainResult:
    """OCR images and train neural-memory brain with extracted text.

    Args:
        source: Path to image file or directory of images.
        brain_db: Path to neural-memory brain .db file.
        domain_tag: Tag for categorizing trained memories.
        ocr_config: OCR pipeline config. Uses defaults if None.
        recursive: Scan subdirectories.
        verbose: Print progress.
        min_confidence: Minimum OCR confidence to accept (0-1).
        min_chars: Minimum text length to accept.

    Returns:
        TrainResult with statistics.
    """
    source = Path(source)
    brain_db = Path(brain_db) if brain_db else DEFAULT_BRAIN_DB
    ocr_config = ocr_config or load_config()

    images = _find_images(source, recursive)
    if not images:
        if verbose:
            print(f"No images found in {source}", file=sys.stderr)
        return TrainResult()

    if verbose:
        print(f"Found {len(images)} images in {source}", file=sys.stderr)
        print(f"Connecting to brain: {brain_db}", file=sys.stderr)

    return asyncio.run(_train_async(
        images=images,
        brain_db=brain_db,
        domain_tag=domain_tag,
        ocr_config=ocr_config,
        verbose=verbose,
        min_confidence=min_confidence,
        min_chars=min_chars,
    ))
