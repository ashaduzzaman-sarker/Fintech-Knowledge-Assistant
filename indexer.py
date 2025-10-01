# indexer.py
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from chunking import chunk_document

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "fintech-knowledge")
BATCH_UPSERT = int(os.environ.get("BATCH_UPSERT", 64))
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_CHUNK_TOKENS = int(os.environ.get("MAX_CHUNK_TOKENS", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 128))

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def ensure_index(index_name: str, dim: int):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dim, metric="cosine", pod_type="p1")
    return pinecone.Index(index_name)

def embed_texts(texts: List[str]):
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return emb.embed_documents(texts)

def upsert_documents(docs: List[Dict]):
    """
    docs: list of {id, title, text, metadata}
    """
    # chunk
    all_chunks = []
    for doc in docs:
        c = chunk_document(doc, max_tokens=MAX_CHUNK_TOKENS, overlap=CHUNK_OVERLAP)
        all_chunks.extend(c)
    if not all_chunks:
        return {"status": "no_chunks", "indexed_chunks": 0}

    texts = [c["text"] for c in all_chunks]
    ids = [f"{c['source_id']}::{c['chunk_id']}" for c in all_chunks]

    vectors = embed_texts(texts)
    dim = len(vectors[0])
    index = ensure_index(INDEX_NAME, dim)

    batch = []
    for idx, vec in enumerate(vectors):
        meta = {
            "source_id": all_chunks[idx]["source_id"],
            "title": all_chunks[idx]["title"],
            "start_token": all_chunks[idx]["start_token"],
            "end_token": all_chunks[idx]["end_token"],
            **all_chunks[idx]["metadata"]
        }
        batch.append((ids[idx], vec, meta))
        if len(batch) >= BATCH_UPSERT:
            index.upsert(vectors=batch)
            batch = []
    if batch:
        index.upsert(vectors=batch)

    return {"status": "ok", "indexed_chunks": len(ids)}
