import sys
import os
import numpy as np
import traceback
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
from langchain_community.vectorstores.utils import DistanceStrategy
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

def intelligent_chunking(url: str) -> List[Dict[str, str]]:
    try:
        print(colored(f"\n\nStarting Intelligent Chunking with LLM Sherpa\n\n", "green"))
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
        return corpus
    
    except Exception as e:        
        print(colored(f"Error in Intelligent Chunking: {str(e)}", "red"))
        traceback.print_exc()
        return Document(page_content="Error in Intelligent Chunking", metadata={"source": url}) 

def index_and_rank(corpus: List[Document], query: str) -> List[Dict[str, str]]:
    print(colored("\n\nStarting indexing and ranking with FastEmbeddings and FAISS\n\n", "green"))
    embeddings = FastEmbedEmbeddings(model_name='jinaai/jina-embeddings-v2-small-en', max_length=512)
    texts = [doc.page_content for doc in corpus]
    metadata = [doc.metadata for doc in corpus]

    print(colored("\n\nCreating FAISS index...\n\n", "green"))
    # faiss = FAISS(distance_strategy=DistanceStrategy.COSINE, embedding_function=embeddings, )
    retriever = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata).as_retriever(search_kwargs={"k": 40, "distance_strategy": DistanceStrategy.COSINE})
    docs = retriever.invoke(query)
    print(colored(f"\n\nRetrieved {len(docs)} documents\n\n", "green"))

    passages = []
    for idx, doc in enumerate(docs, start=1):
        try:
            passage = {
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            passages.append(passage)
        except Exception as e:
            print(colored(f"Error in indexing and ranking: {str(e)}", "red"))
            traceback.print_exc()
            passages.append({"id": idx, "text": "Error in indexing and ranking", "meta": {"source": "unknown"}})

    print(colored("\n\nRe-ranking documents...\n\n", "green"))
    ranker = Ranker(cache_dir=tempfile.mkdtemp())
    rerankrequest = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerankrequest)
    print(colored("\n\nRe-ranking complete\n\n", "green"))

    # Calculate the set percentile
    scores = [result['score'] for result in results]
    percentile_threshold = 50
    percentile = np.percentile(scores, percentile_threshold)

    # Keep only the results above the set percentile
    top_results = [result for result in results if result['score'] > percentile]

    # Convert results to the desired format
    final_results = [
        {
            "text": result['text'],
            "meta": result['meta'],
            "score": result['score']
        }
        for result in top_results
    ]

    # Sort final results by score in descending order
    final_results.sort(key=lambda x: x["score"], reverse=True)

    print(colored(f"\n\nKept {len(final_results)} results above the {percentile_threshold}th percentile (score >= {percentile:.4f})\n\n", "green"))

    return final_results

    
def run_rag(urls: List[str], query: str) -> List[Dict[str, str]]:
    corpus = []
    for url in urls:
        chunks = intelligent_chunking(url)
        corpus.extend(chunks)

    ranked_docs = index_and_rank(corpus, query)
    return ranked_docs

if __name__ == "__main__":
    url1 = "https://warrendale-wagyu.co.uk/?gad_source=1&gclid=CjwKCAjw5Ky1BhAgEiwA5jGujrmS_oFv7e5rxqrLUAQX2ITyPYz5pvbyTT5u7-8POJDoXoPhg0fRWBoCDtgQAvD_BwE"
    url2 = "https://saffronalley.com/collections/premium-wagyu?gad_source=1&gclid=CjwKCAjw5Ky1BhAgEiwA5jGujqF5004NDJrWRkRWhQqvZ869qlXiBOHrxpb1-mgjgD88hd8dmTMGkRoCey0QAvD_BwE"
    url3 = "https://www.finefoodspecialist.co.uk/wagyu-beef"
    # url4 = "https://www.accuweather.com/en/gb/london/ec4a-2/current-weather/328328#google_vignette"
    query = """{
    "🎯": "Find the cheapest available A5 wagyu beef from butchers in London, with their website and contact details",
    "📋": [
        "Identify butchers in London selling A5 wagyu beef",
        "Compare prices of A5 wagyu beef across identified butchers",
        "Determine the cheapest option available",
        "Collect website URLs for each identified butcher",
        "Gather contact details (phone number, email) for each butcher",
        "Include operating hours for each butcher",
        "Add customer ratings or reviews if available",
        "Present information in a table format for easy comparison"
    ],
    "👍🏼": ["Cost-effectiveness", "Quality assurance", "Local sourcing", "Ease of contact", "Online presence", "Convenience", "Customer satisfaction"],
    "🔧": "Enhance requirements with additional details for informed decision-making",
    "🧭": [
        "1. Research butchers in London specializing in A5 wagyu beef",
        "2. Create a list of butchers with their A5 wagyu beef prices",
        "3. Compile website URLs for each butcher",
        "4. Collect phone numbers and email addresses for direct contact",
        "5. Gather operating hours for each butcher",
        "6. Find and include customer ratings or reviews where available",
        "7. Organize all information in a table format for easy comparison",
        "8. Verify the accuracy and currency of all collected information",
        "9. Sort the table by price to highlight the cheapest option"
    ],
}"""

    urls = [url1, url2, url3]
    results = run_rag(urls, query)

    # print(f"\n\n RESULTS: {results}")
