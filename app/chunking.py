import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vector import vector_store
from app.loaders import process_docs

def ingest_file(file_path):
    start = time.time()

    document = process_docs(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=60
    )

    chunks = splitter.split_documents(document)

    # documents = []
    for i , chunk in enumerate(chunks):
        source = chunk.metadata.get('source' , 'unknown')
        page = chunk.metadata.get('page')
        # chunk.metadata['chunk_id'] = f"{source}:{page}:{i}"

        clean_metadata = {
            'source': source,
            'page': page,
            'chunk_id': f"{source}:{page}:{i}"
        }
        if 'title' in chunk.metadata:
            clean_metadata['title'] = chunk.metadata['title']

        chunk.metadata = clean_metadata


    return chunks


    # vector_store.add_documents(documents)


