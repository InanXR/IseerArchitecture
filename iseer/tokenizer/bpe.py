"""
BPE Tokenizer - Built from scratch. OPTIMIZED VERSION.

Byte-Pair Encoding tokenizer that handles Bengali + English text.
No HuggingFace - pure Python implementation.

Optimizations:
- Pre-tokenize into words (massive speedup)
- Track word frequencies to avoid redundant counting
- Use regex for word splitting
"""

import json
import re
import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set


# Simple pre-tokenization pattern - splits on whitespace boundaries
# Works with any Unicode text (Bengali, English, etc.)
SPLIT_PATTERN = re.compile(r'(\s+)', re.UNICODE)


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer - FAST IMPLEMENTATION.
    
    Key optimizations:
    - Pre-tokenize text into words first
    - Track word frequencies (count pairs per unique word × frequency)
    - Much faster than naive O(n²) per merge
    """
    
    # Special tokens
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab: Dict[bytes, int] = {}
        self.inverse_vocab: Dict[int, bytes] = {}
        
        # Special token IDs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        self._init_base_vocab()
    
    def _init_base_vocab(self):
        """Initialize vocabulary with special tokens and all bytes."""
        special_tokens = [
            self.PAD_TOKEN.encode('utf-8'),
            self.UNK_TOKEN.encode('utf-8'),
            self.BOS_TOKEN.encode('utf-8'),
            self.EOS_TOKEN.encode('utf-8'),
        ]
        
        idx = 0
        for token in special_tokens:
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            idx += 1
        
        # Add all single bytes (0-255)
        for i in range(256):
            byte_token = bytes([i])
            if byte_token not in self.vocab:
                self.vocab[byte_token] = idx
                self.inverse_vocab[idx] = byte_token
                idx += 1
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into words/tokens before BPE."""
        # Split on whitespace but keep non-whitespace chunks as tokens
        parts = SPLIT_PATTERN.split(text)
        return [p for p in parts if p]  # Remove empty strings
    
    def train(
        self, 
        texts: List[str], 
        min_frequency: int = 2,
        verbose: bool = True
    ) -> None:
        """
        Train the BPE tokenizer on a corpus - OPTIMIZED.
        
        Key insight: Instead of processing each text separately,
        we pre-tokenize into words and count word frequencies.
        Then we only process unique words, weighted by frequency.
        """
        if verbose:
            print("Pre-tokenizing corpus...")
        
        # Step 1: Pre-tokenize and count word frequencies
        word_freqs: Dict[str, int] = defaultdict(int)
        for text in texts:
            for word in self._pre_tokenize(text):
                word_freqs[word] += 1
        
        if verbose:
            print(f"  Found {len(word_freqs):,} unique words")
        
        # Step 2: Convert words to byte sequences
        # word_bytes[word_as_tuple] = frequency
        word_bytes: Dict[Tuple[bytes, ...], int] = {}
        for word, freq in word_freqs.items():
            # Convert to tuple of single bytes
            byte_seq = tuple(bytes([b]) for b in word.encode('utf-8'))
            if byte_seq in word_bytes:
                word_bytes[byte_seq] += freq
            else:
                word_bytes[byte_seq] = freq
        
        if verbose:
            print(f"  Converted to {len(word_bytes):,} unique byte sequences")
        
        # Step 3: BPE training loop
        num_merges = self.vocab_size - len(self.vocab)
        
        if verbose:
            print(f"Learning {num_merges:,} merges...")
        
        for merge_idx in range(num_merges):
            # Count pair frequencies across all words (weighted by word freq)
            pair_freqs: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
            
            for word_seq, word_freq in word_bytes.items():
                for i in range(len(word_seq) - 1):
                    pair = (word_seq[i], word_seq[i + 1])
                    pair_freqs[pair] += word_freq
            
            if not pair_freqs:
                if verbose:
                    print(f"  No more pairs to merge at iteration {merge_idx}")
                break
            
            # Find best pair
            best_pair = max(pair_freqs.keys(), key=lambda p: pair_freqs[p])
            best_freq = pair_freqs[best_pair]
            
            if best_freq < min_frequency:
                if verbose:
                    print(f"  Best pair freq {best_freq} < {min_frequency}, stopping")
                break
            
            # Merge this pair in all words
            new_word_bytes: Dict[Tuple[bytes, ...], int] = {}
            merged_token = best_pair[0] + best_pair[1]
            
            for word_seq, word_freq in word_bytes.items():
                new_seq = self._merge_pair_tuple(word_seq, best_pair, merged_token)
                if new_seq in new_word_bytes:
                    new_word_bytes[new_seq] += word_freq
                else:
                    new_word_bytes[new_seq] = word_freq
            
            word_bytes = new_word_bytes
            
            # Add to vocabulary
            self.merges.append(best_pair)
            if merged_token not in self.vocab:
                new_idx = len(self.vocab)
                self.vocab[merged_token] = new_idx
                self.inverse_vocab[new_idx] = merged_token
            
            if verbose and (merge_idx + 1) % 1000 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}: freq={best_freq}")
        
        if verbose:
            print(f"Training complete. Vocabulary size: {len(self.vocab):,}")
    
    def _merge_pair_tuple(
        self,
        seq: Tuple[bytes, ...],
        pair: Tuple[bytes, bytes],
        merged: bytes
    ) -> Tuple[bytes, ...]:
        """Merge all occurrences of pair in sequence."""
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return tuple(new_seq)
    
    def _merge_pair(
        self, 
        tokens: List[bytes], 
        pair: Tuple[bytes, bytes]
    ) -> List[bytes]:
        """Merge all occurrences of a pair in the token list."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1 
                and tokens[i] == pair[0] 
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = [bytes([b]) for b in text.encode('utf-8')]
        
        # Apply merges in order
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.unk_id)
        
        if add_special_tokens:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        
        byte_sequence = b""
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.inverse_vocab:
                byte_sequence += self.inverse_vocab[token_id]
        
        try:
            return byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            return byte_sequence.decode('utf-8', errors='replace')
    
    def save(self, path: str) -> None:
        """Save tokenizer to JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": [
                [p[0].hex(), p[1].hex()] for p in self.merges
            ],
            "vocab": {
                k.hex(): v for k, v in self.vocab.items()
            },
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        
        tokenizer.merges = [
            (bytes.fromhex(p[0]), bytes.fromhex(p[1])) 
            for p in data["merges"]
        ]
        tokenizer.vocab = {
            bytes.fromhex(k): v for k, v in data["vocab"].items()
        }
        tokenizer.inverse_vocab = {
            v: k for k, v in tokenizer.vocab.items()
        }
        
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


# Quick test
if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=500)
    
    test_texts = [
        "Hello, World! This is a test.",
        "বাংলাদেশ একটি সুন্দর দেশ।",
        "The quick brown fox jumps over the lazy dog.",
        "আমি বাংলায় কথা বলি।",
    ]
    
    print("Training tokenizer...")
    tokenizer.train(test_texts * 100, verbose=True)
    
    print("\nTesting encode/decode:")
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"Original: {text}")
        print(f"IDs: {ids[:20]}...")
        print(f"Decoded: {decoded}")
        print(f"Match: {text == decoded}")
        print()
