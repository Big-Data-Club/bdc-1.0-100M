"""BDC-1.0-100M: a compact research codebase for decoder-only LM pretraining."""

from .config import ModelConfig, RunConfig, TrainingConfig, load_config
from .model import BDCModel

__all__ = ["BDCModel", "ModelConfig", "TrainingConfig", "RunConfig", "load_config"]
__version__ = "1.0.0"
