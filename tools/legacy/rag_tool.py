import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import tempfile
import requests
from urllib3.exceptions import ProtocolError
from requests.exceptions import ConnectionError, Timeout, RequestException
from typing import Dict, List
from termcolor import colored
from bs4 import BeautifulSoup
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from flashrank import Ranker, RerankRequest
from llmsherpa.readers import LayoutPDFReader
from langchain.schema import Document
from config.load_configs import load_config
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage
from fake_useragent import UserAgent
from config.load_configs import load_config


config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
load_config(config_path)

ua = UserAgent()
os.environ["USER_AGENT"] = ua.random

def scraper(url: str) -> dict:
    print(colored(f"\n\n RAG tool failed, starting basic scraping with URL: {url}\n\n", "green"))
    try:
        print(colored(f"\n\nStarting HTML scraper with URL: {url}\n\n", "green"))
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from relevant tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div'])
        content = ' '.join([element.get_text(strip=True) for element in text_elements])
        
        return {"source": url, "content": content}
    except Exception as e:
        try:
            print(colored(f"\n\nStarting PDF scraper with URL: {url}\n\n", "green"))
            loader = PyPDFLoader(url)
            pages = loader.load_and_split()
            content = "\n".join([page.page_content for page in pages])
            return {"source": url, "content": content}
        except Exception as e:
            return {"source": url, "content": f"Error scraping document: {str(e)}"}


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
        retriever = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata).as_retriever(search_kwargs={"k": 30})
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

        try: 
            print(colored("Falling back to scraper method...", "yellow"))
            results = scraper(url)
            return results
        except Exception as e:
            return []

if __name__ == "__main__":
    url = "https://www.accuweather.com/en/gb/london/ec4a-2/current-weather/328328#google_vignette"
    query = "MMLU scores"
    results = rag_tool(url, query)
    print(f"\n\n RESULTS: {results}")
    
    # Print the content of the first result for verification
    if results:
        print("\n\nContent of the first result:")
        # print(results[0]['text'][:500] + "...")
        print(results)  # Print first 500 characters