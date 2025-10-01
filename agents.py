# agents.py
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict
from retriever import get_langchain_retriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.0))

def build_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

SYNTH_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "You are a careful FinTech knowledge assistant. Use ONLY the provided context to answer. "
        "If the context doesn't contain the answer, say 'I don't know'.\n\nContext:\n{context}\n\n"
        "Answer the query concisely and then provide a JSON array called Sources listing objects with "
        "`title` and `source_id` for provenance.\n\nQuery: {query}\n\nAnswer:"
    )
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="Summarize the following fintech documents into concise bullets for a quick overview:\n\n{context}\n\nSummary:"
)

def retrieve(query: str, k: int = 5):
    retriever = get_langchain_retriever(k=k)
    docs = retriever.get_relevant_documents(query)
    return docs

def synthesize_answer(query: str, docs: List[Dict]):
    llm = build_llm()
    # Keep provenance short: title | source_id
    context = "\n\n---\n\n".join([f"[{d.metadata.get('title','no-title')} | {d.metadata.get('source_id')}] {d.page_content}" for d in docs])
    chain = LLMChain(llm=llm, prompt=SYNTH_PROMPT)
    resp = chain.run({"query": query, "context": context})
    return resp

def summarize(docs: List[Dict]):
    llm = build_llm()
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)
    return chain.run({"context": context})

def answer_pipeline(query: str, k: int = 5):
    docs = retrieve(query, k=k)
    answer = synthesize_answer(query, docs)
    summary = summarize(docs)
    provenance = [{"title": d.metadata.get("title"), "source_id": d.metadata.get("source_id")} for d in docs]
    return {"query": query, "answer": answer, "summary": summary, "documents": provenance}
