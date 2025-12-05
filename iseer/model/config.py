"""
Iseer Model Configuration.

Defines hyperparameters for different model sizes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IseerConfig:
    """Configuration for Iseer model."""
    
    # Vocabulary
    vocab_size: int = 32000
    
    # Model dimensions
    d_model: int = 512
    n_layers: int = 12
    
    # Mamba SSM parameters
    d_state: int = 16          # SSM state dimension
    d_conv: int = 4            # Convolution kernel width
    expand: int = 2            # FFN expansion factor
    
    # Mixture of Experts
    n_experts: int = 8
    top_k: int = 2
    expert_dim: Optional[int] = None  # Defaults to d_model * expand
    
    # Training
    max_seq_len: int = 2048
    dropout: float = 0.0
    
    # Load balancing
    moe_aux_loss_coef: float = 0.01
    
    # Initialization
    initializer_range: float = 0.02
    
    def __post_init__(self):
        if self.expert_dim is None:
            self.expert_dim = self.d_model * self.expand


# Predefined configurations
ISEER_SM = IseerConfig(
    vocab_size=32000,
    d_model=256,
    n_layers=8,
    d_state=16,
    d_conv=4,
    expand=2,
    n_experts=4,
    top_k=2,
    max_seq_len=2048,
)

ISEER_MD = IseerConfig(
    vocab_size=32000,
    d_model=512,
    n_layers=12,
    d_state=16,
    d_conv=4,
    expand=2,
    n_experts=8,
    top_k=2,
    max_seq_len=4096,
)

ISEER_LG = IseerConfig(
    vocab_size=32000,
    d_model=768,
    n_layers=16,
    d_state=16,
    d_conv=4,
    expand=2,
    n_experts=16,
    top_k=2,
    max_seq_len=4096,
)
