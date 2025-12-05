"""
Core layers for Iseer architecture.

Includes:
- RMSNorm (Root Mean Square Layer Normalization)
- FFN (Feed-Forward Network for experts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler and faster than LayerNorm - no mean subtraction.
    Used in LLaMA, Mamba, and now Iseer.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class FFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.
    
    Used as expert networks in MoE layer.
    SwiGLU = Swish(xW) * (xV) - better than ReLU/GELU.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        # SwiGLU has 3 projections instead of 2
        # To match parameter count, use 2/3 * 4 * d_model ≈ 2.67 * d_model
        d_ff_adjusted = int(2 * d_ff / 3)
        
        self.w1 = nn.Linear(d_model, d_ff_adjusted, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff_adjusted, d_model, bias=False)  # Down
        self.w3 = nn.Linear(d_model, d_ff_adjusted, bias=False)  # Up
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(xW1) * (xW3), then project back
        return self.dropout(
            self.w2(F.silu(self.w1(x)) * self.w3(x))
        )


class Embedding(nn.Module):
    """
    Token + Position embeddings.
    
    Uses learned position embeddings (not RoPE for simplicity).
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        
        return self.dropout(tok_emb + pos_emb)
