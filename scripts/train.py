"""
Pre-training Script for Iseer.

Usage:
    python scripts/train.py --config configs/iseer_sm.yaml
    
Or with command line args:
    python scripts/train.py --model-size sm --batch-size 8 --max-steps 10000
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from iseer.model.config import ISEER_SM, ISEER_MD
from iseer.model.iseer import Iseer
from iseer.tokenizer.bpe import BPETokenizer
from iseer.data.dataset import TextDataset, create_dataloader, load_training_texts
from iseer.training.trainer import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Iseer Language Model")
    
    # Model
    parser.add_argument("--model-size", choices=["sm", "md"], default="sm")
    parser.add_argument("--tokenizer", default="iseer/tokenizer/vocab.json")
    
    # Data
    parser.add_argument("--dict-path", default="../train/ProjectShobdo/data/dictionary.json")
    parser.add_argument("--data-files", nargs="*", help="Additional text files")
    parser.add_argument("--seq-len", type=int, default=512)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    
    # Precision
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    
    # Logging
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="iseer")
    parser.add_argument("--run-name", default=None)
    
    args = parser.parse_args()
    
    # Device check
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"   Vocabulary size: {len(tokenizer):,}")
    
    # Load model
    print("\n2. Creating model...")
    config = ISEER_SM if args.model_size == "sm" else ISEER_MD
    config.vocab_size = len(tokenizer)  # Match tokenizer
    model = Iseer(config)
    
    total_params, active_params = model.count_parameters()
    print(f"   Total params: {total_params:,}")
    print(f"   Active params: {active_params:,}")
    
    # Load data
    print("\n3. Loading training data...")
    texts = load_training_texts(
        dict_path=args.dict_path,
        text_files=args.data_files,
    )
    
    # Create dataloader  
    print("\n4. Creating dataloader...")
    train_dataloader = create_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    
    # Training config
    train_config = TrainingConfig(
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=not args.no_mixed_precision,
        dtype=args.dtype,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
    )
    
    # Create trainer
    print("\n5. Starting training...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation}")
    print(f"   Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Mixed precision: {not args.no_mixed_precision} ({args.dtype})")
    print()
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=train_config,
    )
    
    # Train!
    trainer.train()
    
    print("\n✓ Training complete!")
    print(f"  Checkpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
