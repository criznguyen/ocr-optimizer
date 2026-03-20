"""Configuration models for OCR Optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "glm-ocr:latest"
    prompt: str = "OCR this image. Extract all text exactly as shown."
    timeout: int = 180


@dataclass
class PreprocessConfig:
    upscale_factor: int = 2
    contrast: float = 1.3
    sharpen_passes: int = 0
    sharpness: float = 1.0
    binarize: bool = False
    binarize_threshold: int = 140


@dataclass
class CropConfig:
    strategy: str = "columns"  # none, columns, grid
    num_columns: int = 2
    overlap_px: int = 20


@dataclass
class PipelineConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    passes_per_crop: int = 1
    merge_similarity_threshold: float = 0.6
    min_line_length: int = 10


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        # Try default locations
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if path is None or not Path(path).exists():
        return PipelineConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    ollama = OllamaConfig(**raw.get("ollama", {}))
    preprocess = PreprocessConfig(**raw.get("preprocess", {}))
    crop = CropConfig(**raw.get("crop", {}))
    pipeline = raw.get("pipeline", {})

    return PipelineConfig(
        ollama=ollama,
        preprocess=preprocess,
        crop=crop,
        passes_per_crop=pipeline.get("passes_per_crop", 1),
        merge_similarity_threshold=pipeline.get("merge_similarity_threshold", 0.6),
        min_line_length=pipeline.get("min_line_length", 10),
    )
