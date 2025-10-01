# streamlit_app.py
import streamlit as st
import requests

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FinTech RAG", layout="centered")
st.title("FinTech Knowledge Assistant — RAG (Hugging Face ingestion)")

st.sidebar.header("Ingest")
ingest_mode = st.sidebar.selectbox("Ingest from", ["Upload file", "CSV upload", "Hugging Face dataset"])
if ingest_mode == "Upload file":
    uploaded = st.sidebar.file_uploader("Upload PDF/TXT", type=["pdf","txt"])
    if uploaded and st.sidebar.button("Ingest file"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        res = requests.post(f"{API_URL}/ingest/file", files=files)
        st.sidebar.write(res.json())
elif ingest_mode == "CSV upload":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    text_col = st.sidebar.text_input("Text column name", "text")
    if uploaded and st.sidebar.button("Ingest CSV"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        res = requests.post(f"{API_URL}/ingest/csv", files=files, data={"text_column": text_col})
        st.sidebar.write(res.json())
else:
    dataset = st.sidebar.text_input("HF dataset (e.g. banking77, financial_phrasebank)", "banking77")
    split = st.sidebar.text_input("split", "train")
    text_field = st.sidebar.text_input("text_field", "text")
    max_examples = st.sidebar.number_input("max_examples (0 = all)", 0, 100000, 0)
    if st.sidebar.button("Ingest HF dataset"):
        payload = {"dataset": dataset, "split": split, "text_field": text_field}
        if max_examples > 0:
            payload["max_examples"] = int(max_examples)
        res = requests.post(f"{API_URL}/ingest/hf", data=payload)
        st.sidebar.write(res.json())

st.header("Ask a question")
q = st.text_input("Query")
k = st.slider("Top-K retrieval", 1, 20, 5)
if st.button("Ask"):
    resp = requests.get(f"{API_URL}/query", params={"q": q, "k": k}).json()
    st.subheader("Answer")
    st.write(resp.get("answer"))
    st.subheader("Summary")
    st.write(resp.get("summary"))
    st.subheader("Sources")
    for s in resp.get("documents", []):
        st.write(s)
