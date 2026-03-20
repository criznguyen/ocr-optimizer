"""Ollama API client for GLM-OCR."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import requests
from PIL import Image

from .config import OllamaConfig


@dataclass
class OCRResult:
    text: str
    elapsed: float
    eval_count: int
    tokens_per_sec: float
    region_name: str = ""


class OllamaOCR:
    """Wrapper around Ollama API for OCR inference."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self._url = f"{config.base_url}/api/generate"

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 PNG string."""
        buf = io.BytesIO()
        # Use JPEG for large images to reduce payload size
        if img.width * img.height > 2_000_000:
            img_rgb = img.convert("RGB") if img.mode != "RGB" else img
            img_rgb.save(buf, format="JPEG", quality=95)
        else:
            img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def run(self, image: Image.Image, prompt: str | None = None, region_name: str = "") -> OCRResult:
        """Run OCR on a PIL Image. Returns OCRResult.

        If the request fails (e.g. OOM), retries once with a downscaled image.
        """
        return self._run_with_retry(image, prompt or self.config.prompt, region_name)

    def _run_with_retry(self, image: Image.Image, prompt: str, region_name: str, attempt: int = 0) -> OCRResult:
        img_b64 = self._image_to_base64(image)
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }

        start = time.time()
        try:
            resp = requests.post(self._url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            if attempt < 1:
                # Retry with downscaled image (50%)
                smaller = image.resize(
                    (image.width // 2, image.height // 2), Image.BICUBIC
                )
                return self._run_with_retry(smaller, prompt, region_name, attempt + 1)
            raise

        elapsed = time.time() - start
        result = resp.json()
        text = result.get("response", "")
        eval_count = result.get("eval_count", 0)
        tps = eval_count / elapsed if elapsed > 0 else 0

        return OCRResult(
            text=text,
            elapsed=elapsed,
            eval_count=eval_count,
            tokens_per_sec=tps,
            region_name=region_name,
        )
