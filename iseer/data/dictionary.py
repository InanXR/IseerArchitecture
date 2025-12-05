"""
Shobdo Dictionary Processing.

Converts the Bengali dictionary JSON into training text.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterator


def load_shobdo_dictionary(path: str) -> List[Dict[str, Any]]:
    """Load the Shobdo dictionary from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def entry_to_text(entry: Dict[str, Any]) -> str:
    """
    Convert a single dictionary entry to natural language text.
    
    Example output:
    "অকপট (অকপট) একটি সংস্কৃত শব্দ। এর অর্থ অমায়িক, কপটতাহীন।
    ইংরেজিতে Candid বা Sincere। উদাহরণ: তিনি অকপট স্বীকারোক্তি দিয়ে সবার মন জয় করলেন।"
    """
    parts = []
    
    word = entry.get("word", "")
    pronunciation = entry.get("pronunciation", "")
    
    # Word and pronunciation
    if pronunciation and pronunciation != word:
        parts.append(f"{word} ({pronunciation})")
    else:
        parts.append(word)
    
    # Etymology
    etymology = entry.get("etymology")
    if etymology and isinstance(etymology, dict):
        source = etymology.get("source_language", "")
        if source:
            parts.append(f"একটি {source} শব্দ।")
    
    # Part of speech
    pos = entry.get("part_of_speech")
    if pos:
        parts.append(f"পদ: {pos}।")
    
    # Meanings
    meanings = entry.get("meanings", [])
    if meanings:
        meaning_text = "; ".join(meanings[:3])  # Limit to 3 meanings
        parts.append(f"অর্থ: {meaning_text}")
    
    # English translation
    english = entry.get("english_translation")
    if english:
        parts.append(f"ইংরেজিতে: {english}।")
    
    # Examples
    examples = entry.get("examples", [])
    if examples:
        parts.append(f"উদাহরণ: {examples[0]}")
    
    return " ".join(parts)


def dictionary_to_text(entries: List[Dict[str, Any]]) -> Iterator[str]:
    """Convert all dictionary entries to training text."""
    for entry in entries:
        try:
            text = entry_to_text(entry)
            if text.strip():
                yield text
        except Exception:
            continue  # Skip malformed entries


def load_and_convert(path: str) -> List[str]:
    """Load dictionary and convert to training texts."""
    entries = load_shobdo_dictionary(path)
    return list(dictionary_to_text(entries))


if __name__ == "__main__":
    import sys
    
    # Test with sample
    path = sys.argv[1] if len(sys.argv) > 1 else "../../train/ProjectShobdo/data/dictionary.json"
    
    print("Loading dictionary...")
    entries = load_shobdo_dictionary(path)
    print(f"Loaded {len(entries):,} entries")
    
    print("\nSample conversions:")
    for i, text in enumerate(dictionary_to_text(entries[:5])):
        print(f"\n{i+1}. {text}")
