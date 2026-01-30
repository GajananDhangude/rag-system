from app.vector import add_to_chroma
from app.chunking import ingest_file
from app.rag_query import query_rag


def ingest_and_set_active(file_path):
    global active_source
    # 1. Split and process your chunks
    chunks = ingest_file(file_path)
    

    if chunks:
        active_source = chunks[0].metadata.get('source')
    
    add_to_chroma(chunks)
    print(f" Active Document set to: {active_source}")

    return active_source


if __name__ =="__main__":

    file_path = "data/attention-is-all-you-need-Paper.pdf"

    source = ingest_and_set_active(file_path)

    query_text = input("\n Enter your query:")

    responce = query_rag(query_text , source)

    print(responce)

