"""
Mixture of Experts (MoE) Layer.

Implements sparse expert routing with load balancing.
Only top-k experts are activated per token = efficient!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from iseer.model.layers import FFN


class Router(nn.Module):
    """
    Expert router with top-k selection.
    
    Learns which experts to route each token to.
    Uses softmax over expert scores, selects top-k.
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            weights: Routing weights for top-k experts (B, L, top_k)
            indices: Indices of selected experts (B, L, top_k)
            router_logits: Raw logits for load balancing loss (B, L, n_experts)
        """
        # Compute routing scores
        router_logits = self.gate(x)  # (B, L, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize weights to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return weights, indices, router_logits


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Contains multiple FFN experts, routes tokens to top-k of them.
    Includes load balancing loss to prevent expert collapse.
    
    Args:
        d_model: Model dimension
        n_experts: Number of expert networks
        top_k: Number of experts to route to per token
        expert_dim: Hidden dimension for expert FFNs
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        expert_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Router
        self.router = Router(d_model, n_experts, top_k)
        
        # Expert networks (all identical architecture, different weights)
        self.experts = nn.ModuleList([
            FFN(d_model, expert_dim, dropout)
            for _ in range(n_experts)
        ])
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            output: Combined expert outputs (B, L, D)
            aux_loss: Load balancing auxiliary loss (scalar)
        """
        B, L, D = x.shape
        
        # Route tokens to experts
        weights, indices, router_logits = self.router(x)
        
        # Compute expert outputs
        # For simplicity, we process each expert separately
        # Production would use batched parallel dispatch
        
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # (B, L)
            expert_weight = weights[:, :, k:k+1]  # (B, L, 1)
            
            for e in range(self.n_experts):
                # Find tokens routed to this expert
                mask = (expert_idx == e)  # (B, L)
                
                if mask.any():
                    # Get tokens for this expert
                    # Flatten for processing
                    flat_x = x.view(-1, D)  # (B*L, D)
                    flat_mask = mask.view(-1)  # (B*L,)
                    
                    # Process through expert
                    expert_input = flat_x[flat_mask]  # (num_tokens, D)
                    expert_output = self.experts[e](expert_input)
                    
                    # Weight and add to output
                    flat_weight = expert_weight.view(-1, 1)[flat_mask]  # (num_tokens, 1)
                    weighted_output = expert_output * flat_weight
                    
                    # Scatter back
                    flat_output = output.view(-1, D)
                    flat_output[flat_mask] += weighted_output
        
        # Compute load balancing loss
        aux_loss = self._load_balancing_loss(router_logits, indices)
        
        return output, aux_loss
    
    def _load_balancing_loss(
        self,
        router_logits: torch.Tensor,  # (B, L, n_experts)
        indices: torch.Tensor,  # (B, L, top_k)
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages uniform expert utilization, prevents collapse.
        
        L_aux = α * n_experts * Σ_i (f_i * P_i)
        
        where:
        - f_i = fraction of tokens routed to expert i
        - P_i = mean routing probability for expert i
        """
        B, L, _ = router_logits.shape
        n_tokens = B * L
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # (B, L, n_experts)
        
        # f_i: fraction of tokens routed to each expert
        # Count how many times each expert is in top-k
        expert_counts = torch.zeros(self.n_experts, device=router_logits.device)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k].view(-1)  # (B*L,)
            expert_counts.scatter_add_(
                0, 
                expert_idx, 
                torch.ones_like(expert_idx, dtype=torch.float)
            )
        
        # Normalize to get fraction
        f = expert_counts / (n_tokens * self.top_k)
        
        # P_i: mean routing probability for each expert
        P = router_probs.mean(dim=[0, 1])  # (n_experts,)
        
        # Load balancing loss
        aux_loss = self.n_experts * (f * P).sum()
        
        return aux_loss
