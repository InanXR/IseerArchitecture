"""
Tokenizer Training Script.

Trains a BPE tokenizer on Bengali + English data.
Includes: Shobdo dictionary, CC-100 Bengali, Wikipedia, FineWeb-Edu.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Iterator

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iseer.tokenizer.bpe import BPETokenizer
from iseer.data.dictionary import load_shobdo_dictionary, dictionary_to_text


def load_dictionary_texts(dict_path: str) -> List[str]:
    """Load and convert Shobdo dictionary to training texts."""
    print(f"Loading dictionary from {dict_path}...")
    entries = load_shobdo_dictionary(dict_path)
    texts = list(dictionary_to_text(entries))
    print(f"  Converted {len(texts):,} dictionary entries")
    return texts


def load_huggingface_texts(dataset_name: str, lang: str = None, split: str = "train", max_samples: int = 50000) -> List[str]:
    """Load texts from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Warning: datasets library not installed. Skipping HuggingFace data.")
        return []
    
    print(f"Loading {dataset_name} (lang={lang})...")
    try:
        if dataset_name == "cc100" and lang:
            ds = load_dataset("cc100", lang=lang, split=split, streaming=True)
        elif dataset_name == "wikipedia" and lang:
            ds = load_dataset("wikipedia", f"20231101.{lang}", split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        
        texts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            text = item.get("text", "")
            if len(text) > 50:  # Skip very short texts
                texts.append(text[:2000])  # Truncate very long texts
        
        print(f"  Loaded {len(texts):,} samples")
        return texts
    
    except Exception as e:
        print(f"  Warning: Failed to load {dataset_name}: {e}")
        return []


def train_tokenizer(
    output_path: str,
    vocab_size: int = 32000,
    dict_path: str = None,
    use_hf_data: bool = True,
    max_hf_samples: int = 25000,
):
    """
    Train the BPE tokenizer.
    
    Args:
        output_path: Where to save the trained tokenizer
        vocab_size: Target vocabulary size
        dict_path: Path to Shobdo dictionary JSON
        use_hf_data: Whether to load HuggingFace datasets
        max_hf_samples: Max samples per HuggingFace dataset
    """
    all_texts = []
    
    # 1. Load Shobdo dictionary (Bengali)
    if dict_path and Path(dict_path).exists():
        dict_texts = load_dictionary_texts(dict_path)
        all_texts.extend(dict_texts)
    else:
        print("Warning: Dictionary path not found, skipping")
    
    # 2. Load HuggingFace datasets
    if use_hf_data:
        # Bengali
        bn_texts = load_huggingface_texts("cc100", lang="bn", max_samples=max_hf_samples)
        all_texts.extend(bn_texts)
        
        # English  
        en_texts = load_huggingface_texts("cc100", lang="en", max_samples=max_hf_samples)
        all_texts.extend(en_texts)
    
    print(f"\nTotal training texts: {len(all_texts):,}")
    
    if len(all_texts) < 1000:
        print("\nWarning: Limited data available. Training on dictionary only.")
    
    # 3. Train tokenizer
    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(all_texts, min_frequency=2, verbose=True)
    
    # 4. Save tokenizer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    
    # 5. Test
    print("\n" + "="*60)
    print("TESTING TOKENIZER")
    print("="*60)
    
    test_texts = [
        "Hello, World!",
        "বাংলাদেশ একটি সুন্দর দেশ।",
        "অকপট মানে সৎ এবং সরল।",
        "The Iseer architecture combines Mamba with MoE.",
    ]
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        match = "✓" if text == decoded else "✗"
        print(f"\n{match} Original: {text}")
        print(f"  Tokens:  {len(ids)} ids")
        print(f"  Decoded: {decoded}")
    
    print(f"\n✓ Tokenizer saved to {output_path}")
    print(f"  Vocabulary size: {len(tokenizer):,}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Iseer BPE Tokenizer")
    parser.add_argument(
        "--output", "-o",
        default="iseer/tokenizer/vocab.json",
        help="Output path for tokenizer"
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=32000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--dict-path", "-d",
        default="../train/ProjectShobdo/data/dictionary.json",
        help="Path to Shobdo dictionary"
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Skip HuggingFace datasets (use dictionary only)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25000,
        help="Max samples per HuggingFace dataset"
    )
    
    args = parser.parse_args()
    
    train_tokenizer(
        output_path=args.output,
        vocab_size=args.vocab_size,
        dict_path=args.dict_path,
        use_hf_data=not args.no_hf,
        max_hf_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
