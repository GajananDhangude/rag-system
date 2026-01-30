from langchain_community.document_loaders import DirectoryLoader
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness , answer_relevancy , context_recall , context_precision
from app.rag_query import query_rag
from app.test_data_eval import test_data
from datasets import Dataset
import os
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from ragas.llms import LangchainLLMWrapper

ragas_eval_llm = OpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_wrapper = LangchainLLMWrapper(ragas_eval_llm)


path = "data/uploads/b673bbc5-3158-4ae2-ac64-5bdab4a3e2f9_attention-is-all-you-need-Paper.pdf"


def rag_evaluation(test_data:list[dict]):


    rag_results = []

    source = path

    for item in test_data:
        question = item['question']

        result = query_rag(question , source)

        rag_results.append({
            "question":question,
            "answer": result['responce'],
            "contexts":result['contexts'],
            "ground_truths":item["ground_truth"]
        })

    results_df = pd.DataFrame(rag_results)
    results_df.rename(columns={"ground_truths": "reference"}, inplace=True)

    ragas_dataset = Dataset.from_pandas(results_df)

    score = evaluate(
        ragas_dataset,
        metrics=[faithfulness , answer_relevancy , context_recall , context_precision],
        llm=llm_wrapper,
    )

    return score.to_pandas()


if __name__ =="__main__":

    score_df = rag_evaluation(test_data)
    print("\n--- Manual Ragas Evaluation Results ---")
    print(score_df)
    print("\n--- Average Scores ---")
    print(score_df.mean(numeric_only=True))