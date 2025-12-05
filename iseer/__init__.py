"""
Iseer: A Novel Mamba × MoE Hybrid Language Model
Built from scratch.
"""

from iseer.model.config import IseerConfig
from iseer.model.iseer import Iseer
from iseer.tokenizer.bpe import BPETokenizer

__version__ = "0.1.0"
__all__ = ["Iseer", "IseerConfig", "BPETokenizer"]
