# ocr-optimizer

Multi-pass OCR pipeline using [GLM-OCR](https://huggingface.co/THUDM/glm-ocr) (1.1B params) via [Ollama](https://ollama.com). Preprocesses images, splits into overlapping column crops, runs OCR on each region, and merges results with a smart line-level alignment algorithm.

Optionally trains [neural-memory](https://github.com/criznguyen/neural-memory) with extracted text.

## How it works

```
Image → Preprocess (upscale 2x, contrast, sharpen)
      → Crop (full + N columns with overlap)
      → OCR each region (GLM-OCR via Ollama)
      → Smart Merge (line alignment + voting + missing line recovery)
      → Text output (+ optional neural-memory training)
```

**Key findings from benchmarking:**
- Upscale 2x is the single most impactful technique (0/8 → 7/8 accuracy on real newspaper scan)
- Same-model multi-crop beats multi-model ensemble on 8GB VRAM
- Prompt engineering has minimal effect on dedicated OCR models

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with GLM-OCR model:
  ```bash
  ollama pull glm-ocr:latest
  ```

## Installation

```bash
pip install -e .
```

## Usage

### Single image OCR

```bash
# Basic — uses default config
ocr-optimize ocr image.png

# Custom config
ocr-optimize ocr image.png -c config.yaml

# Save output to file
ocr-optimize ocr image.png -o output.txt

# Disable preprocessing or multi-pass
ocr-optimize ocr image.png --no-preprocess --no-multipass
```

### Batch OCR + neural-memory training

```bash
# Train from a directory of images
ocr-optimize train /path/to/images/ --brain-db ~/.neural-memory/brain.db

# With custom domain tag and confidence threshold
ocr-optimize train /path/to/images/ --brain-db brain.db --domain my-docs --min-confidence 0.5
```

## Configuration

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "glm-ocr:latest"
  timeout: 180

preprocess:
  upscale_factor: 2       # bicubic upscale
  contrast: 1.3
  sharpness: 1.5
  binarize: false          # Otsu threshold (use for very noisy scans)

crop:
  strategy: "columns"      # columns | grid | none
  num_columns: 2
  overlap_px: 20           # overlap between crops to avoid cutting text

pipeline:
  multi_pass: true
  preprocess: true
```

## Python API

```python
from ocr_optimizer import load_config, run_pipeline

config = load_config("config.yaml")
result = run_pipeline("document.png", config)

print(result.text)
print(f"Confidence: {result.confidence:.1%}")
print(f"Passes: {result.num_passes}")
```

## Performance

| Image type | Accuracy | Speed (RTX 2070) |
|-----------|----------|-------------------|
| Clean scan, upscale 2x | 7/8 fields | ~3-5s |
| Multi-pass (full + 2 crops) | 7/8 fields | ~8-12s |
| Raw (no preprocessing) | 0/8 fields | ~2s |

## Project structure

```
ocr_optimizer/
├── config.py          # Dataclass configs + YAML loader
├── preprocessing.py   # Upscale, contrast, sharpness, binarize
├── cropper.py         # Column/grid crop with overlap
├── ocr_client.py      # Ollama API client (auto-retry on OOM)
├── merge.py           # Smart merge: line alignment + voting
├── pipeline.py        # Orchestrator: preprocess → crop → OCR → merge
├── trainer.py         # Neural-memory training integration
└── cli.py             # CLI entry point
```

## License

MIT
