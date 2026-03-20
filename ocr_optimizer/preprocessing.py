"""Image preprocessing for OCR optimization."""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

from .config import PreprocessConfig

# Map string names to PIL resampling methods
_RESAMPLE = {
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


def upscale(img: Image.Image, factor: int, method: str = "bicubic") -> Image.Image:
    """Resize image by a given factor."""
    if factor <= 1:
        return img.copy()
    resample = _RESAMPLE.get(method, Image.BICUBIC)
    return img.resize((img.width * factor, img.height * factor), resample)


def enhance_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast. factor=1.0 is no change, >1.0 increases contrast."""
    if factor == 1.0:
        return img
    return ImageEnhance.Contrast(img).enhance(factor)


def enhance_sharpness(img: Image.Image, factor: float, passes: int) -> Image.Image:
    """Apply sharpening filter + sharpness enhancement."""
    result = img
    for _ in range(passes):
        result = result.filter(ImageFilter.SHARPEN)
    if factor != 1.0:
        result = ImageEnhance.Sharpness(result).enhance(factor)
    return result


def binarize(img: Image.Image, threshold: int = 140) -> Image.Image:
    """Convert to black & white using threshold."""
    gray = img.convert("L")
    bw = gray.point(lambda x: 0 if x < threshold else 255, "1")
    return bw.convert("RGB")


def preprocess_image(img: Image.Image, config: PreprocessConfig) -> Image.Image:
    """Apply full preprocessing chain."""
    result = upscale(img, config.upscale_factor)
    result = enhance_contrast(result, config.contrast)
    if config.sharpen_passes > 0 or config.sharpness != 1.0:
        result = enhance_sharpness(result, config.sharpness, config.sharpen_passes)
    if config.binarize:
        result = binarize(result, config.binarize_threshold)
    return result
