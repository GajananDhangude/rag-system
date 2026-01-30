from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from app.vector import vector_store
from app.vector import add_to_chroma
from app.chunking import ingest_file
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_rag(quert_text:str , source:str):

    retriever = vector_store.as_retriever(
        search_kwargs = {
            "k":3,
            "filter":{"source":source}
        }
    )

    # retrieved_docs = retriever.invoke(quert_text)


    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering based ONLY on the provided context.
Context:
{context}
Question:
{question}
Answer clearly and concisely:
""")
    

    rag_chain = (
        {
            "context":retriever | format_docs,
            "question":RunnablePassthrough()

        }
        |prompt
        |llm
        |StrOutputParser()
    )

    responce = rag_chain.invoke(quert_text)

    return {
        "responce":responce,
        # "contexts":[doc.page_content for doc in retrieved_docs],
    }

