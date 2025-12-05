"""
Trainer for Iseer model.

Features:
- Mixed precision (fp16/bf16) training
- Gradient accumulation
- Gradient checkpointing (optional)
- Learning rate scheduling
- Logging and checkpointing
- Wandb integration (optional)
"""

import math
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Batching
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Precision
    mixed_precision: bool = True
    dtype: str = "float16"  # float16 or bfloat16
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 10
    output_dir: str = "checkpoints"
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "iseer"
    wandb_run_name: Optional[str] = None


class Trainer:
    """
    Trainer for Iseer language model.
    
    Handles the full training loop with:
    - Mixed precision
    - Gradient accumulation
    - LR scheduling
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Training on {self.device}")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # LR Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        self.autocast_dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16
        
        # State
        self.global_step = 0
        self.epoch = 0
        
        # Output dir
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
    
    def _create_scheduler(self):
        """Create cosine LR scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            # Cosine decay
            progress = (step - self.config.warmup_steps) / (
                self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self) -> Dict[str, float]:
        """Run the full training loop."""
        self.model.train()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        step_start_time = time.time()
        
        data_iter = iter(self.train_dataloader)
        
        for step in range(self.config.max_steps):
            self.global_step = step
            
            # Accumulate gradients
            accumulated_loss = 0.0
            accumulated_aux = 0.0
            
            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                    self.epoch += 1
                
                loss, aux_loss = self._train_step(batch)
                accumulated_loss += loss
                accumulated_aux += aux_loss
            
            # Average over accumulation steps
            accumulated_loss /= self.config.gradient_accumulation_steps
            accumulated_aux /= self.config.gradient_accumulation_steps
            
            # Gradient clipping and optimizer step
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += accumulated_loss
            total_aux_loss += accumulated_aux
            
            # Logging
            if (step + 1) % self.config.log_steps == 0:
                elapsed = time.time() - step_start_time
                avg_loss = total_loss / self.config.log_steps
                avg_aux = total_aux_loss / self.config.log_steps
                lr = self.scheduler.get_last_lr()[0]
                
                print(
                    f"Step {step + 1}/{self.config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | Aux: {avg_aux:.4f} | "
                    f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
                )
                
                if self.config.use_wandb and HAS_WANDB:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/aux_loss": avg_aux,
                        "train/lr": lr,
                        "train/step": step + 1,
                    })
                
                total_loss = 0.0
                total_aux_loss = 0.0
                step_start_time = time.time()
            
            # Evaluation
            if self.eval_dataloader and (step + 1) % self.config.eval_steps == 0:
                eval_loss = self.evaluate()
                print(f"  Eval Loss: {eval_loss:.4f}")
                if self.config.use_wandb and HAS_WANDB:
                    wandb.log({"eval/loss": eval_loss, "train/step": step + 1})
                self.model.train()
            
            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{step + 1}")
        
        # Final save
        self.save_checkpoint("final")
        
        return {"final_loss": accumulated_loss}
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step with mixed precision."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        with autocast(dtype=self.autocast_dtype, enabled=self.config.mixed_precision):
            logits, loss, aux_loss = self.model(input_ids, labels=labels)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        return loss.item(), aux_loss if isinstance(aux_loss, float) else aux_loss.item()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with autocast(dtype=self.autocast_dtype, enabled=self.config.mixed_precision):
                _, loss, _ = self.model(input_ids, labels=labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.output_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": vars(self.config),
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from {path}")
