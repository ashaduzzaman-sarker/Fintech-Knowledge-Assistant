# chunking.py
import tiktoken
from typing import List, Dict

def get_tokenizer(model_name: str = "gpt-4o-mini"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, max_tokens: int = 800, overlap: int = 128, model_name: str = "gpt-4o-mini") -> List[Dict]:
    enc = get_tokenizer(model_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(tokens):
        end = start + max_tokens
        slice_tokens = tokens[start:end]
        chunk_text = enc.decode(slice_tokens)
        chunks.append({
            "text": chunk_text,
            "start_token": start,
            "end_token": min(end, len(tokens)),
            "chunk_id": f"chunk_{chunk_idx}_{start}"
        })
        chunk_idx += 1
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def chunk_document(doc: Dict, max_tokens: int = 800, overlap: int = 128, model_name: str = "gpt-4o-mini"):
    text = doc.get("text", "")
    chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap, model_name=model_name)
    for c in chunks:
        c.update({
            "source_id": doc.get("id"),
            "title": doc.get("title"),
            "metadata": doc.get("metadata", {})
        })
    return chunks
