"""
Iseer: Full Language Model Architecture.

Combines Mamba SSM + Mixture of Experts into a complete LLM.

This is the main model file - the culmination of:
- Mamba blocks for O(n) sequence modeling
- MoE for sparse, efficient computation
- RMSNorm, embeddings, and output head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from iseer.model.config import IseerConfig
from iseer.model.mamba import MambaBlock
from iseer.model.moe import MoELayer
from iseer.model.layers import RMSNorm, Embedding


class IseerBlock(nn.Module):
    """
    Single Iseer block: Mamba + MoE.
    
    Architecture:
        x → RMSNorm → Mamba → + → RMSNorm → MoE → + → out
            └──────────────────┘   └──────────────┘
              (residual)             (residual)
    """
    
    def __init__(self, config: IseerConfig):
        super().__init__()
        
        # Mamba path
        self.norm1 = RMSNorm(config.d_model)
        self.mamba = MambaBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=config.dropout,
        )
        
        # MoE path
        self.norm2 = RMSNorm(config.d_model)
        self.moe = MoELayer(
            d_model=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
            expert_dim=config.expert_dim,
            dropout=config.dropout,
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input (B, L, D)
            
        Returns:
            output: Output (B, L, D)
            aux_loss: MoE load balancing loss
        """
        # Mamba with residual
        x = x + self.mamba(self.norm1(x))
        
        # MoE with residual
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, aux_loss


class Iseer(nn.Module):
    """
    Iseer Language Model.
    
    A novel architecture combining:
    - Mamba (State Space Models) for linear-time sequence modeling
    - Mixture of Experts for sparse, efficient computation
    
    Built from scratch. No HuggingFace.
    
    Args:
        config: IseerConfig with model hyperparameters
    """
    
    def __init__(self, config: IseerConfig):
        super().__init__()
        self.config = config
        
        # Token + position embeddings
        self.embed = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        
        # Iseer blocks
        self.blocks = nn.ModuleList([
            IseerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.d_model)
        
        # Output head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed.token_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Iseer model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (B, L)
            labels: Target token IDs for loss computation (B, L)
            
        Returns:
            logits: Output logits (B, L, vocab_size)
            loss: Language modeling loss (if labels provided)
            aux_loss: Total MoE auxiliary loss
        """
        # Embeddings
        x = self.embed(input_ids)
        
        # Pass through blocks
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Final norm and output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            
            # Add auxiliary loss
            loss = loss + self.config.moe_aux_loss_coef * total_aux_loss
        
        return logits, loss, total_aux_loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens (B, L)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token IDs (B, L + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max length if needed
            idx_cond = input_ids
            if input_ids.size(1) > self.config.max_seq_len:
                idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and active parameters.
        
        Returns:
            total_params: All parameters
            active_params: Parameters active per forward pass (approx)
        """
        total = sum(p.numel() for p in self.parameters())
        
        # Estimate active params (only top_k experts used)
        expert_params = sum(
            p.numel() 
            for block in self.blocks 
            for expert in block.moe.experts 
            for p in expert.parameters()
        )
        active_expert_params = expert_params * (self.config.top_k / self.config.n_experts)
        
        non_expert_params = total - expert_params
        active = int(non_expert_params + active_expert_params)
        
        return total, active


# Quick test
if __name__ == "__main__":
    from iseer.model.config import ISEER_SM
    
    print("Testing Iseer model...")
    
    # Create model
    model = Iseer(ISEER_SM)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, ISEER_SM.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    logits, loss, aux_loss = model(input_ids, labels)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Aux loss: {aux_loss:.4f}")
    
    # Test generation
    prompt = torch.randint(0, ISEER_SM.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    # Count parameters
    total, active = model.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Active parameters: {active:,}")
    print(f"Efficiency: {active/total*100:.1f}% active")
    
    print("\n✓ All tests passed!")
