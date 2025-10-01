# api.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import sqlite3
from ingestion import ingest_file, read_csv
from hf_ingestion import hf_to_docs
from indexer import upsert_documents
from agents import answer_pipeline
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
DB_PATH = "storage.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.post("/ingest/file")
async def ingest_endpoint(file: UploadFile = File(...)):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    doc = ingest_file(path)
    res = upsert_documents([doc])
    return JSONResponse(res)

@app.post("/ingest/csv")
async def ingest_csv_endpoint(file: UploadFile = File(...), text_column: str = Form("text")):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    docs = read_csv(path, text_column)
    res = upsert_documents(docs)
    return JSONResponse(res)

@app.post("/ingest/hf")
async def ingest_hf_endpoint(dataset: str = Form(...), split: str = Form("train"), text_field: str = Form("text"), max_examples: int = Form(None)):
    docs = hf_to_docs(dataset, split=split, text_field=text_field, max_examples=max_examples)
    res = upsert_documents(docs)
    return JSONResponse(res)

@app.get("/query")
async def query_endpoint(q: str, k: int = 5):
    result = answer_pipeline(q, k=k)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO queries (query, response) VALUES (?, ?)", (q, result["answer"]))
    conn.commit()
    conn.close()
    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
