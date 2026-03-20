# OCR Optimizer — Project Report

## Overview

**ocr-optimizer** is a multi-pass OCR optimization pipeline using GLM-OCR (1.1B params) via Ollama. It preprocesses images, splits them into overlapping column crops, runs OCR on each region, and merges results using a smart line-level alignment algorithm.

- **Version**: 0.1.0
- **Language**: Python 3.11+
- **Total LOC**: 1,050 (9 files)
- **CLI entry point**: `ocr-optimize`

---

## Architecture

```
Image Input
    │
    ▼
┌─────────────┐
│ Preprocess   │  upscale 2x (bicubic), contrast 1.3, sharpness 1.5, optional binarize
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Cropper      │  full image + N column crops (20px overlap)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ OCR Client   │  GLM-OCR via Ollama API (localhost:11434)
│              │  auto-retry with downscale on OOM
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Smart Merge  │  line-level alignment, voting, missing line recovery
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Trainer      │  neural-memory encode (optional)
└─────────────┘
```

---

## Codebase

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 8 | Public exports |
| `config.py` | 78 | Dataclass configs: `OllamaConfig`, `PreprocessConfig`, `CropConfig`, `PipelineConfig` |
| `preprocessing.py` | 59 | Image upscale, contrast, sharpness, binarization |
| `cropper.py` | 65 | Column/grid crop with overlap |
| `ocr_client.py` | 85 | `OllamaOCR` — base64 image → text via Ollama `/api/generate`. Auto-downscale retry. JPEG for >2M pixels |
| `merge.py` | 218 | Smart merge: `difflib.SequenceMatcher` line alignment, best-line voting, missing line recovery, substring dedup |
| `pipeline.py` | 122 | `run_pipeline()` orchestrator: preprocess → crop → multi-pass OCR → merge |
| `trainer.py` | 210 | Neural-memory training: single event loop for aiosqlite, Brain create/find, async encode |
| `cli.py` | 205 | Subcommands: `ocr` (single image), `train` (batch OCR → neural-memory) |

---

## Performance Benchmarks

### Preprocessing Impact (stock_gs200.jpg — real newspaper scan)

| Technique | Accuracy (out of 8 key fields) |
|-----------|-------------------------------|
| Raw (no preprocessing) | 0/8 |
| Upscale 2x only | **7/8** |
| Upscale 2x + contrast | 7/8 |
| Upscale 2x + sharpen | 7/8 |
| Binarize (Otsu) | 1/8 |
| Crop columns only | 0/8 |

**Key finding**: Upscale 2x is the single most impactful technique. "Blackbaud" consistently misread — fundamental model limitation at 1.1B params.

### Multi-Pass vs Single-Pass

| Strategy | Accuracy |
|----------|----------|
| Single model (GLM-OCR) full image | 7/8 |
| Multi-model ensemble (GLM-OCR + Granite + Moondream + MiniCPM-V) | 4/8 |
| Same-model multi-crop (full + 2 columns, smart merge) | **7/8** |

**Key finding**: Same-model multi-crop beats multi-model ensemble on 8GB VRAM. Different models output incompatible formats making merge unreliable.

### OCR Speed (RTX 2070, 8GB VRAM)

| Image | Time |
|-------|------|
| Single pass (upscale 2x) | ~3-5s |
| Multi-pass (full + 2 crops) | ~8-12s |
| Full pipeline (preprocess + OCR + merge) | ~10-15s |

### Neural-Memory Training (batch of test images)

| Metric | Value |
|--------|-------|
| Neurons created | 173 |
| Synapses | 2,499 |
| Fibers | 4 |
| Training time (3 images) | ~2s |

---

## Dependencies

```
ocr-optimizer
├── Pillow >= 10.0        (image processing)
├── requests >= 2.28      (Ollama API calls)
├── pyyaml >= 6.0         (config loading)
└── [optional] neural-memory  (training integration)
```

**External**:
- Ollama server running locally with `glm-ocr:latest` model pulled

---

## Configuration

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "glm-ocr:latest"
  timeout: 180
preprocess:
  upscale_factor: 2
  contrast: 1.3
  sharpness: 1.5
  binarize: false
crop:
  strategy: "columns"    # columns | grid | none
  num_columns: 2
  overlap_px: 20
pipeline:
  multi_pass: true
  preprocess: true
```

---

## Known Limitations

1. **Single GPU constraint**: RTX 2070 8GB VRAM limits to 1 concurrent OCR call
2. **GLM-OCR 1.1B accuracy ceiling**: Some proper nouns consistently misread (e.g., "Blackbaud")
3. **No built-in PDF support** — handled by doc-harvester via pdf2image
4. **Prompt engineering has minimal effect** on dedicated OCR models like GLM-OCR

---

## Roadmap

- [ ] Add support for larger OCR models when VRAM allows (e.g., Qwen2-VL 7B)
- [ ] Layout detection (table vs paragraph vs header) for smarter cropping
- [ ] Confidence scoring per line (not just per pass)
- [ ] Caching layer for repeated images
- [ ] Unit tests
