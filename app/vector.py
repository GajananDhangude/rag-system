from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

VECTOR_DIR = "data/vector_store"
COLLECTION = "rag_system"

os.makedirs(VECTOR_DIR, exist_ok=True)


embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR
    )


def add_to_chroma(chunks: list[Document]):

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    

    existing_items = vector_store.get(ids=chunk_ids)
    existing_ids = set(existing_items['ids'])

    new_chunks = []
    new_ids = []

    for chunk , chunk_id in zip(chunks , chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new chunks...")
        vector_store.add_documents(documents=new_chunks, ids=new_ids)
    else:
        print(" All chunks already exist. Skipping...")

