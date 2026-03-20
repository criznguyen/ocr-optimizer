"""OCR Optimizer - Multi-pass OCR with smart merge using GLM-OCR via Ollama."""

__version__ = "0.1.0"

from .config import PipelineConfig, load_config
from .pipeline import run_pipeline
from .trainer import train_brain
