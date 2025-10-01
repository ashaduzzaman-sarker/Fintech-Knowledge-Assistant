# ingestion.py
import pdfplumber
import os
from typing import List, Dict
import pandas as pd

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_csv(path: str, text_column: str = "text") -> List[Dict]:
    df = pd.read_csv(path)
    docs = []
    for i, row in df.iterrows():
        docs.append({
            "id": f"csv_{i}",
            "title": str(row.get('title', f"row_{i}")),
            "text": str(row[text_column]),
            "metadata": row.to_dict()
        })
    return docs

def read_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)

def ingest_file(path: str, title: str = None) -> Dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = read_pdf(path)
    elif ext == ".txt":
        text = read_txt(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return {"id": os.path.basename(path), "title": title or os.path.basename(path), "text": text, "metadata": {"source_path": path}}
