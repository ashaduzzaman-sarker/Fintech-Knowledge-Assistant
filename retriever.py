# retriever.py
import os
from dotenv import load_dotenv
load_dotenv()

import pinecone
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "fintech-knowledge")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def get_langchain_retriever(k: int = 5):
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    index = pinecone.Index(INDEX_NAME)
    vectorstore = LC_Pinecone(index, emb.embed_query, text_key="text")
    retriever = vectorstore.as_retriever(search_type="cosine", search_kwargs={"k": k})
    return retriever
