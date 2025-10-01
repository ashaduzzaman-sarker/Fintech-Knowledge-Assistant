# Fintech RAG Assistant (LangChain + Pinecone + HF datasets)

## Setup
1. Copy `.env.example` → `.env` and fill keys.
2. `pip install -r requirements.txt`
3. Start API:
   `uvicorn api:app --reload`
4. Start Streamlit (optional):
   `streamlit run streamlit_app.py`

## Ingest from Hugging Face
POST to `/ingest/hf` with form fields:
- `dataset` (e.g., `banking77`)
- `split` (default `train`)
- `text_field` (e.g., `text`)
- `max_examples` (optional integer)

## Query
GET `/query?q=...&k=5`

## Notes
- Adjust chunk sizes and overlap in `.env`.
- Ensure Pinecone index exists (created automatically by indexer if not present).
