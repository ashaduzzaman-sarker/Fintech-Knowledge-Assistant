# hf_ingestion.py
from datasets import load_dataset
from typing import List, Dict

def hf_to_docs(dataset_name: str, split: str = "train", text_field: str = "text", title_field: str = None, max_examples: int = None) -> List[Dict]:
    """
    Convert HF dataset into list of docs suitable for RAG ingestion.
    """
    ds = load_dataset(dataset_name, split=split)
    docs = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        # guard if text_field missing
        if text_field not in row:
            # try common alternatives
            possibles = [k for k in row.keys() if isinstance(row[k], str)]
            if not possibles:
                continue
            text = str(row[possibles[0]])
        else:
            text = str(row[text_field])
        title = str(row[title_field]) if title_field and title_field in row else f"{dataset_name}_{i}"
        metadata = {k: str(v) for k, v in row.items() if k not in [text_field, title_field]}
        docs.append({
            "id": f"{dataset_name}_{split}_{i}",
            "title": title,
            "text": text,
            "metadata": metadata
        })
    return docs

if __name__ == "__main__":
    # small smoke test
    docs = hf_to_docs("banking77", split="train", text_field="text", max_examples=5)
    for d in docs:
        print(d["id"], d["title"], d["text"][:120], "...")
