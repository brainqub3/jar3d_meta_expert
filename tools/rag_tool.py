import sys
import os
import tempfile
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
from typing import Dict, List
from termcolor import colored
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from flashrank import Ranker, RerankRequest
from llmsherpa.readers import LayoutPDFReader
from langchain.schema import Document
from config.load_configs import load_config


config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
load_config(config_path)


def rag_tool(url: str, query: str) -> List[Dict[str, str]]:
    try:
        print(colored(f"\n\nStarting rag_tool with URL: {url} and query: {query}\n\n", "green"))
        llmsherpa_api_url = os.environ.get('LLM_SHERPA_SERVER')
        if not llmsherpa_api_url:
            raise ValueError("LLM_SHERPA_SERVER environment variable is not set")
        print(colored(f"\n\nUsing LLM Sherpa API URL: {llmsherpa_api_url}\n\n", "green"))

        reader = LayoutPDFReader(llmsherpa_api_url)
        print(colored("\n\nURL reader initialized\n\n", "green"))
        doc = reader.read_pdf(url)
        print(colored("\n\nURL Contents red successfully\n\n", "green"))

        corpus = []
        for chunk in doc.chunks():
            documents = Document(
                page_content=chunk.to_context_text(),
                metadata={"source": url}
            )
            corpus.append(documents)

        print(colored(f"\n\nCreated corpus with {len(corpus)} documents\n\n", "green"))

        embeddings = FastEmbedEmbeddings(model_name='jinaai/jina-embeddings-v2-small-en', max_length=512)
        texts = [doc.page_content for doc in corpus]
        metadata = [doc.metadata for doc in corpus]

        print(colored("\n\nCreating FAISS index...\n\n", "green"))
        retriever = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata).as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        print(colored(f"\n\nRetrieved {len(docs)} documents\n\n", "green"))

        passages = []
        for idx, doc in enumerate(docs, start=1):
            passage = {
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            passages.append(passage)

        print(colored("\n\nRe-ranking documents...\n\n", "green"))
        ranker = Ranker(cache_dir=tempfile.mkdtemp())
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerankrequest)
        print(colored("\n\nRe-ranking complete\n\n", "green"))

        return results
    except Exception as e:
        print(colored(f"Error in rag_tool: {str(e)}", "red"))
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    url = "https://arxiv.org/pdf/2108.00573.pdf"
    query = "What is MusiQue?"
    results = rag_tool(url, query)
    print(f"\n\n RERANKED DOCS: {results}")

    # query = "What is MusiQue?"
    # docs = reranker_retriever.invoke(query)
    # print(results)
    # print(f"\n\n RERANKED DOCS: {results}")