"""
Dataset and DataLoader for Iseer training.

Handles:
- Loading text from various sources
- Tokenization with our BPE tokenizer
- Chunking into fixed-length sequences
- Efficient batching with DataLoader
"""

import json
import random
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class TextDataset(Dataset):
    """
    Simple text dataset for pre-training.
    
    Loads texts, tokenizes them, and chunks into fixed-length sequences.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int = 512,
        overlap: int = 64,
    ):
        """
        Args:
            texts: List of training texts
            tokenizer: BPE tokenizer instance
            seq_len: Sequence length for training
            overlap: Overlap between consecutive chunks
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.overlap = overlap
        
        # Tokenize and chunk all texts
        self.chunks = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            # Create overlapping chunks
            stride = seq_len - overlap
            for i in range(0, len(ids) - seq_len + 1, stride):
                chunk = ids[i:i + seq_len]
                if len(chunk) == seq_len:
                    self.chunks.append(chunk)
        
        print(f"Created {len(self.chunks):,} training chunks")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        # For causal LM: input = tokens[:-1], target = tokens[1:]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large-scale training.
    
    Yields chunks on-the-fly without loading everything into memory.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 512,
        shuffle_buffer: int = 10000,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        
        # Load and process file(s)
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = list(self.data_path.glob("*.txt")) + list(self.data_path.glob("*.json"))
        
        for file_path in files:
            for text in self._read_file(file_path):
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                
                # Chunk the tokens
                for i in range(0, len(ids) - self.seq_len + 1, self.seq_len):
                    chunk = ids[i:i + self.seq_len]
                    if len(chunk) == self.seq_len:
                        buffer.append(chunk)
                    
                    # Shuffle and yield when buffer is full
                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        for item in buffer:
                            yield self._chunk_to_sample(item)
                        buffer = []
        
        # Yield remaining items
        random.shuffle(buffer)
        for item in buffer:
            yield self._chunk_to_sample(item)
    
    def _read_file(self, path: Path) -> Iterator[str]:
        """Read texts from file."""
        if path.suffix == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            yield item
                        elif isinstance(item, dict) and "text" in item:
                            yield item["text"]
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    
    def _chunk_to_sample(self, chunk: List[int]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = TextDataset(texts, tokenizer, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_training_texts(
    dict_path: Optional[str] = None,
    text_files: Optional[List[str]] = None,
    max_texts: Optional[int] = None,
) -> List[str]:
    """Load training texts from various sources."""
    texts = []
    
    # Load from dictionary
    if dict_path and Path(dict_path).exists():
        from iseer.data.dictionary import load_shobdo_dictionary, dictionary_to_text
        entries = load_shobdo_dictionary(dict_path)
        dict_texts = list(dictionary_to_text(entries))
        texts.extend(dict_texts)
        print(f"Loaded {len(dict_texts):,} texts from dictionary")
    
    # Load from text files
    if text_files:
        for file_path in text_files:
            path = Path(file_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    file_texts = [line.strip() for line in f if line.strip()]
                texts.extend(file_texts)
                print(f"Loaded {len(file_texts):,} texts from {file_path}")
    
    # Limit if needed
    if max_texts and len(texts) > max_texts:
        random.shuffle(texts)
        texts = texts[:max_texts]
    
    print(f"Total training texts: {len(texts):,}")
    return texts
