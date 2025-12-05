"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

This is a from-scratch implementation of the Mamba block.
Reference: https://arxiv.org/abs/2312.00752

Key innovations:
- Input-dependent (selective) state space parameters
- Hardware-efficient selective scan
- O(n) complexity instead of O(n²) attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

# Try to import Triton for custom kernels
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class MambaBlock(nn.Module):
    """
    Single Mamba block.
    
    Architecture:
    1. Input projection (expand to d_inner)
    2. Causal 1D convolution (local context)
    3. Selective SSM (the core innovation)
    4. Output projection
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Convolution kernel width
        expand: Expansion factor for inner dimension
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Input projection: x -> (z, x) where z is for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal 1D convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise
        )
        
        # SSM parameters projection
        # Projects x to: delta, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Delta (timestep) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt_proj bias for proper timescale
        dt_init_std = 1.0 / math.sqrt(self.d_inner)
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        # A parameter (state transition matrix) - learned
        # Initialize to negative values (for stability after exp)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection: split into x and z (gate)
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Causal convolution
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Causal: truncate future
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # SSM
        y = self._ssm(x_conv)
        
        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        return self.dropout(y)
    
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective State Space Model.
        
        h[k] = Ā·h[k-1] + B̄·x[k]
        y[k] = C·h[k] + D·x[k]
        
        Where Ā, B̄ are discretized from continuous A, B using
        input-dependent timestep delta.
        """
        B, L, D = x.shape
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Project x to get delta, B, C (selective parameters)
        x_proj = self.x_proj(x)  # (B, L, d_state * 2 + 1)
        
        # Split projections
        delta = x_proj[:, :, :1]  # (B, L, 1)
        B_sel = x_proj[:, :, 1:self.d_state + 1]  # (B, L, d_state)
        C_sel = x_proj[:, :, self.d_state + 1:]  # (B, L, d_state)
        
        # Compute delta (timestep) with softplus for positivity
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Discretize A and B
        # Ā = exp(delta * A) for each position
        # B̄ = delta * B
        
        # Run selective scan
        y = self._selective_scan(
            x, delta, A, B_sel, C_sel, self.D
        )
        
        return y
    
    def _selective_scan(
        self,
        x: torch.Tensor,      # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,      # (N,)
        B: torch.Tensor,      # (B, L, N)
        C: torch.Tensor,      # (B, L, N)
        D: torch.Tensor,      # (D,)
    ) -> torch.Tensor:
        """
        Selective scan implementation.
        
        This is the core of Mamba - a parallel scan over the sequence.
        For production, this would use Triton kernels for speed.
        Here we use a simple but correct sequential implementation.
        """
        B_batch, L, d_inner = x.shape
        N = A.shape[0]
        
        # Expand A for broadcasting: (D, N)
        A = repeat(A, 'n -> d n', d=d_inner)
        
        # Initialize state
        h = torch.zeros(B_batch, d_inner, N, device=x.device, dtype=x.dtype)
        
        # Output accumulator
        ys = []
        
        for i in range(L):
            # Get current inputs
            x_i = x[:, i, :]  # (B, D)
            delta_i = delta[:, i, :]  # (B, D)
            B_i = B[:, i, :]  # (B, N)
            C_i = C[:, i, :]  # (B, N)
            
            # Discretize for this timestep
            # Ā = exp(delta * A)
            delta_A = delta_i.unsqueeze(-1) * A  # (B, D, N)
            A_bar = torch.exp(delta_A)
            
            # B̄ = delta * B (broadcast over D dimension)
            B_bar = delta_i.unsqueeze(-1) * B_i.unsqueeze(1)  # (B, D, N)
            
            # State update: h = Ā * h + B̄ * x
            h = A_bar * h + B_bar * x_i.unsqueeze(-1)
            
            # Output: y = C * h + D * x
            y_i = torch.einsum('bdn,bn->bd', h, C_i) + D * x_i
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y


if HAS_TRITON:
    @triton.jit
    def selective_scan_kernel(
        # Pointers to tensors
        x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
        out_ptr, h_ptr,
        # Dimensions
        B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
        # Block size
        BLOCK: tl.constexpr,
    ):
        """
        Triton kernel for selective scan.
        TODO: Implement parallel scan for maximum speed.
        """
        pid = tl.program_id(0)
        # Kernel implementation would go here
        pass
