from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uvicorn

from app.chunking import ingest_file
from app.vector import add_to_chroma
from app.rag_query import query_rag

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG API", version="1.0")

# Track the last uploaded file globally for convenience
latest_source = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    source: str | None = None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    global latest_source
    
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".pdf", ".txt", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")


    save_path = os.path.join(UPLOAD_DIR, f"{file.filename}")

    if os.path.exists(save_path):
        yield "The file already exists in our local storage"
    
    else:
        yield "Path is Clear . Saving File."


    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest into vector DB
    chunks = ingest_file(save_path)
    add_to_chroma(chunks)

    # Update the active source so the user doesn't have to provide it in /query
    if chunks:
        latest_source = chunks[0].metadata.get("source")

    return {
        "message": "Document uploaded & indexed",
        "source": latest_source
    }

@app.post("/query")
def query_rag_api(req: QueryRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required")

    target_source =latest_source
    
    if not target_source:
        raise HTTPException(status_code=400, detail="No active document. Please upload one first.")

    # Call your RAG function
    answer = query_rag(req.question, target_source)

    return {
        "answer": answer['responce'],
        # "context_used":answer['contexts'],
        "source_used": target_source
    }



if __name__ =="__main__":

    uvicorn.run(
        "app.api:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False
    )