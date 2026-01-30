from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader


def process_docs(file_path:str):

    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".txt"):
        return load_txt(file_path)
    elif file_path.endswith(".docx"):
        return load_docx(file_path)
    
    else:
        raise ValueError("Document Type is not Supported")

def load_pdf(file_path:str):
    loader = PyMuPDFLoader(
        file_path,
        mode="page",
        extract_tables="markdown"
    )

    docs = loader.load()
    return docs

def load_txt(file_path:str):
    try:
        loader = TextLoader(file_path , encoding="utf-8")
        docs = loader.load()

        return docs
    except UnicodeDecodeError:
        loader = TextLoader(file_path , encoding="latin-1")
        docs = loader.load()

        return docs
        
def load_docx(file_path:str):
    try:
        loader = Docx2txtLoader(file_path)

        docs = loader.load()
        return docs
    
    except Exception as e :
        raise ValueError(f"Error extracting DoCX text : {str(e)}")
    
