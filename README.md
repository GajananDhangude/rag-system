This repository contains a Retrieval-Augmented Generation (RAG) system built with a production mindset. It supports multi-format document ingestion, metadata-aware vector search, hybrid retrieval (semantic + keyword), LLM-based reranking, and citation-grounded answers — all exposed through a clean FastAPI backend.



#Project Structure 

app/
loaders.py # PDF, TXT, DOCX loaders
chunking.py # Chunking + metadata injection
vector.py # ChromaDB + embeddings + deduplication
rag_query.py # Hybrid RAG pipeline + citations
api.py # FastAPI service


.env
requirements.txt
README.md


#RAG Pipeline 

RAG Pipeline Flow

Document Upload

File is saved locally

Parsed using format-specific loaders

Split into overlapping chunks

Metadata added: source, page, chunk_id, timestamp

Vector Ingestion

Embeddings generated using sentence-transformers/all-MiniLM-L6-v2

Stored in ChromaDB with deduplication

Query Processing

Semantic vector search

Keyword boosting (hybrid search)

LLM-based reranking

Answer generation with citations


# installation

1 clone repo 

2 create virtual enviroment

3 install dependencies
  pip install -r requirements.txt

4 setup your GROQ_API_KEY = ""

5 run the server 

uvicorn app.api:app --reload