"""Smart region cropping for multi-pass OCR."""

from __future__ import annotations

from PIL import Image

from .config import CropConfig


def crop_columns(img: Image.Image, n_cols: int, overlap_px: int = 20) -> list[Image.Image]:
    """Split image into vertical column strips with overlap."""
    if n_cols <= 1:
        return [img.copy()]

    col_width = img.width // n_cols
    crops = []

    for i in range(n_cols):
        x_start = max(0, i * col_width - overlap_px)
        x_end = min(img.width, (i + 1) * col_width + overlap_px)
        crop = img.crop((x_start, 0, x_end, img.height))
        crops.append(crop)

    return crops


def crop_grid(img: Image.Image, rows: int, cols: int, overlap_px: int = 20) -> list[Image.Image]:
    """Split image into a grid of regions with overlap."""
    if rows <= 1 and cols <= 1:
        return [img.copy()]

    cell_w = img.width // cols
    cell_h = img.height // rows
    crops = []

    for r in range(rows):
        for c in range(cols):
            x_start = max(0, c * cell_w - overlap_px)
            x_end = min(img.width, (c + 1) * cell_w + overlap_px)
            y_start = max(0, r * cell_h - overlap_px)
            y_end = min(img.height, (r + 1) * cell_h + overlap_px)
            crops.append(img.crop((x_start, y_start, x_end, y_end)))

    return crops


def crop_regions(img: Image.Image, config: CropConfig) -> list[Image.Image]:
    """Generate crop regions based on config strategy.

    Returns list of cropped images. Always includes full image as first element.
    """
    if config.strategy == "none":
        return [img.copy()]

    # Always include full image first
    regions = [img.copy()]

    if config.strategy == "columns":
        regions.extend(crop_columns(img, config.num_columns, config.overlap_px))
    elif config.strategy == "grid":
        # For grid, use num_columns as both rows and cols
        regions.extend(crop_grid(img, config.num_columns, config.num_columns, config.overlap_px))

    return regions
